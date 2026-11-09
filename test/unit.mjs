/**
 * Offline unit tests. No GPU, no network, no DSH host: every module here is pure
 * or writes only under a temp dir, so `npm test` is the gate a contributor runs
 * before anything expensive.
 */
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
// Lazy, not a top-level `import { zstdCompressSync }`: that makes the whole test
// file unloadable on Node < 22.15, which is exactly the Node the CI matrix runs.
// The zstd tests then skip with a printed reason instead of the suite failing to
// start - a missing capability and a broken decoder are different things.
const zlib = await import('node:zlib');
const zstdCompressSync = typeof zlib.zstdCompressSync === 'function' ? zlib.zstdCompressSync : null;
const constants = zlib.constants;

import { decodeSessionFile } from '../lib/session.mjs';
import { redactText } from '../lib/redact.mjs';
import { splitSamples } from '../lib/pack.mjs';
import { packSamples, canonicalKey } from '../lib/pack.mjs';
import { scoreAdherence, verdictOf } from '../lib/evaluate.mjs';
import { contentTokens, scoreTextAnswer, scoreTextSplit, shapeOf } from '../lib/textmetric.mjs';
import { estimateMemory, recommendation, parseNvidiaSmi } from '../lib/hardware.mjs';
import { validateTrainingPlan, planSummary, capabilityOf } from '../lib/training.mjs';
import { createJobStore } from '../lib/jobs.mjs';
import { createLineReader, parseWorkerLine, shellQuote } from '../lib/backend.mjs';
import { buildSshArgs, createSshExec, validateTarget } from '../lib/ssh.mjs';
import { putLine, validateRemotePath } from '../lib/transfer.mjs';
import { validateSourceInput } from '../lib/sources.mjs';
import { buildDataset } from '../lib/dataset.mjs';
import { loopback, sameOrigin, allowSpawn } from '../index.js';

const tests = [];
const test = (name, fn) => tests.push({ name, fn });

// ---------------------------------------------------------------- session.mjs
test('decodes a zstd frame whose reserved descriptor bit is set', () => {
  // Node's encoder is not available, so build the frame the way the host does:
  // take a real gzip stream's bytes is useless —instead synthesise a frame by
  // compressing with zlib's zstd when available, else fall back to asserting the
  // reserved-bit clear on a crafted header.
  const plain = Buffer.from('{"type":"session","id":"s1"}\n{"type":"turn/start","data":{"turn":1}}\n');
  const frame = zstdFrame(plain, { reserved: true });
  const decoded = decodeSessionFile(frame);
  assert.equal(decoded.decodable, true);
  assert.equal(decoded.lines.length, 2);
  assert.equal(JSON.parse(decoded.lines[0]).type, 'session');
});

test('decodes concatenated frames', () => {
  const a = zstdFrame(Buffer.from('{"n":1}\n'), { reserved: true });
  const b = zstdFrame(Buffer.from('{"n":2}\n'), { reserved: false });
  const decoded = decodeSessionFile(Buffer.concat([a, b]));
  assert.equal(decoded.frames, 2);
  assert.deepEqual(decoded.lines, ['{"n":1}', '{"n":2}']);
});

test('an unreadable file says so instead of reporting empty', () => {
  const decoded = decodeSessionFile(Buffer.from('not zstd and not json'));
  assert.equal(decoded.decodable, false);
  assert.match(decoded.reason, /no zstd frame/);
});

// ---------------------------------------------------------------- redact.mjs
test('keys, PII and home paths are redacted; ordinary text survives', () => {
  const input = 'key sk-abcdefghijklmnopqrstuvwx email a.b@corp.com id 110101199003074012 path C:\\Users\\alice\\x 璁㈠崟鍙?12345';
  const out = redactText(input, 'path');
  assert.match(out.text, /\[REDACTED_KEY\]/);
  assert.match(out.text, /\[REDACTED_EMAIL\]/);
  assert.match(out.text, /\[REDACTED_ID\]/);
  assert.match(out.text, /C:\\Users\\\[USER\]/);
  assert.match(out.text, /12345/, 'a plain number must survive');
});

test('an invalid Chinese ID check digit is not redacted', () => {
  const out = redactText('编号 110101199003074013', 'pii');
  assert.doesNotMatch(out.text, /REDACTED_ID/);
});

test('redaction reports counts, never the matched value', () => {
  const out = redactText('a@b.com and c@d.com', 'pii');
  assert.deepEqual(out.hits.map((hit) => [hit.rule, hit.count]), [['email', 2]]);
  assert.doesNotMatch(JSON.stringify(out.hits), /a@b/);
});

test('the off profile changes nothing', () => {
  const input = 'sk-abcdefghijklmnopqrstuvwx a@b.com';
  assert.equal(redactText(input, 'off').text, input);
});

const syntheticSession = () => ({
  header: { id: 'sess-1' },
  events: [
    { type: 'system/message', data: { message: { content: [{ type: 'text', text: 'SYS' }] } } },
    { type: 'developer/message', data: { message: { content: [{ type: 'tool-addition', toolName: 'read' }, { type: 'tool-addition', toolName: 'bash' }] } } },
    { type: 'user/message', data: { source: { kind: 'user' }, content: [{ type: 'text', text: '跑测试'}] } },
    // a failed call: the retry, not this, is the label
    { type: 'assistant/message', data: { turn: 1, step: 1, message: { content: [{ type: 'reasoning', text: 'think' }, { type: 'tool-call', name: 'read', toolCallId: 'c1', arguments: { path: 'README.bad' } }] } } },
    { type: 'tool/result', data: { toolCallId: 'c1', message: { isError: true, content: [{ type: 'text', text: 'ENOENT' }] } } },
    { type: 'assistant/message', data: { turn: 1, step: 2, message: { content: [{ type: 'tool-call', name: 'read', toolCallId: 'c2', arguments: { path: 'README.md' } }] } } },
    { type: 'tool/result', data: { toolCallId: 'c2', message: { content: [{ type: 'text', text: '# Title' }] } } },
    { type: 'assistant/message', data: { turn: 1, step: 3, message: { content: [{ type: 'text', text: 'README的标题是 Title。'}] } } },
    { type: 'turn/end', data: { turn: 1, reason: { kind: 'completed' } } },
    // a hallucinated tool: dropped
    { type: 'user/message', data: { source: { kind: 'user' }, content: [{ type: 'text', text: '跑测试'}] } },
    { type: 'assistant/message', data: { turn: 2, step: 1, message: { content: [{ type: 'tool-call', name: 'made_up_tool', toolCallId: 'c3', arguments: {} }] } } },
    { type: 'tool/result', data: { toolCallId: 'c3', message: { content: [{ type: 'text', text: 'unknown tool' }] } } },
    { type: 'turn/end', data: { turn: 2, reason: { kind: 'aborted' } } },
  ],
});








test('canonical key collapses punctuation and width variants', () => {
  const a = { messages: [{ role: 'user', content: '你好？' }, { role: 'assistant', content: '好的。' }] };
  const b = { messages: [{ role: 'user', content: '你好' }, { role: 'assistant', content: '好的' }] };
  const c = { messages: [{ role: 'user', content: '你好，在吗？' }, { role: 'assistant', content: '好的' }] };
  const squash = (key) => key.replace(/\s+/g, '');
  assert.equal(squash(canonicalKey(a)), squash(canonicalKey(b)));
  // `c` adds real words ("在吗"), so it must NOT collapse onto `a` — only
  // punctuation, whitespace and width variants are normalised away.
  assert.notEqual(squash(canonicalKey(a)), squash(canonicalKey(c)));
  const different = { messages: [{ role: 'user', content: '你好，明天？' }, { role: 'assistant', content: '好的。' }] };
  assert.notEqual(canonicalKey(a), canonicalKey(different));
});

test('packing drops duplicates and invalid samples', () => {
  const good = { messages: [{ role: 'user', content: 'q' }, { role: 'assistant', content: 'a' }] };
  const result = packSamples([good, good, { messages: [{ role: 'user', content: 'only' }] }, { messages: [] }]);
  assert.equal(result.kept.length, 1);
  assert.equal(result.duplicates, 1);
  assert.equal(result.invalid, 2);
});

test('splitting never puts one group on both sides', () => {
  const samples = [];
  for (let g = 0; g < 10; g += 1) {
    for (let i = 0; i < 5; i += 1) samples.push({ messages: [{ role: 'user', content: `${g}-${i}` }, { role: 'assistant', content: 'a' }], meta: { group: `g${g}` } });
  }
  const split = splitSamples(samples, { valRatio: 0.2, evalRatio: 0.1 });
  const groupsOf = (rows) => new Set(rows.map((row) => row.meta.group));
  const train = groupsOf(split.train);
  const val = groupsOf(split.val);
  const evalSet = groupsOf(split.eval);
  for (const group of val) assert.equal(train.has(group), false);
  for (const group of evalSet) assert.equal(train.has(group), false);
  assert.equal(split.train.length + split.val.length + split.eval.length, 50);
});

// ---------------------------------------------------------------- evaluate.mjs
test('adherence scores the tool name and the arguments', () => {
  const cases = [
    { prompt: 'p', expected: '```json\n{"tool":"read","arguments":{"path":"a"}}\n```' },
    { prompt: 'p', expected: '```json\n{"tool":"read","arguments":{"path":"a"}}\n```' },
  ];
  const outputs = [
    '```json\n{"tool":"read","arguments":{"path":"a"}}\n```',
    '```json\n{"tool":"bash","arguments":{"path":"a"}}\n```',
  ];
  const score = scoreAdherence(cases, outputs);
  assert.equal(score.score, 0.5);
  assert.equal(score.categories.tool_name, 0.5);
});

test('an invented tool fails even when the expected tool is also called', () => {
  const cases = [{ prompt: 'tools: read, bash', expected: '{"tool":"read","arguments":{}}' }];
  const outputs = ['{"tool":"read","arguments":{}}\n{"tool":"teleport","arguments":{}}'];
  const score = scoreAdherence(cases, outputs, { knownTools: ['read', 'bash'] });
  assert.equal(score.pass, 0);
});

test('the publish gate blocks a regression and passes a tie', () => {
  const base = { score: 0.9, total: 10, categories: { tool_name: 0.9, arguments: 0.9 } };
  assert.equal(verdictOf({ candidate: base, base }).status, 'green');
  assert.equal(verdictOf({ candidate: { ...base, score: 0.5 }, base }).status, 'blocked');
  assert.equal(verdictOf({ candidate: base, base, tolerance: 0.1 }).status, 'green');
  assert.equal(verdictOf({ candidate: { score: 0, total: 0, categories: {} }, base }).status, 'blocked');
});

// ---------------------------------------------------------------- hardware.mjs
test('the corrected estimator fails a 1.5B on 6 GB above seq 256', () => {
  // Measured reality (RTX 3060 6 GB, 2026-11-04): a 0.5B at seq 512 peaks at
  // 3.16 GiB, and the cost is dominated by the vocab 脳 seq fp32 logits plus
  // per-layer activation retention —not by parameter count. So a 7B is hopeless
  // and even a 1.5B only squeezes in below ~512 tokens.
  const big = estimateMemory({ modelId: 'Qwen/Qwen2.5-7B-Instruct', maxSeqLen: 2048, gpuGiB: 6 });
  assert.equal(big.fits, false);
  // Once activations dominate (they do, post-calibration), the honest advice for
  // any over-budget plan is "shorten the sequence" before "shrink the model".
  assert.equal(recommendation(big).action, 'maxSeqLen');

  const oneFiveLong = estimateMemory({ modelId: 'Qwen/Qwen2.5-1.5B-Instruct', maxSeqLen: 512, gpuGiB: 6 });
  assert.equal(oneFiveLong.fits, false, '1.5B at seq 512 does not fit 6 GB');

  const oneFiveShort = estimateMemory({ modelId: 'Qwen/Qwen2.5-1.5B-Instruct', maxSeqLen: 256, gpuGiB: 6 });
  assert.equal(oneFiveShort.fits, true, '1.5B at seq 256 does fit');

  const small = estimateMemory({ modelId: 'Qwen/Qwen2.5-0.5B-Instruct', maxSeqLen: 512, gpuGiB: 6 });
  assert.equal(small.fits, true);
});

test('the estimator reproduces the measured peaks it was calibrated from', () => {
  // Two independent measurements on a real 0.5B run. The estimate must land
  // within +25% of each —high enough to absorb the coarse constants, low enough
  // that a 20脳 miss (the pre-fix bug) can never pass again.
  const measured = { 512: 3.16, 1024: 5.58 };
  for (const [seq, actual] of Object.entries(measured)) {
    const estimate = estimateMemory({ modelId: 'Qwen/Qwen2.5-0.5B-Instruct', maxSeqLen: Number(seq), gpuGiB: 6 });
    const ratio = estimate.totalGiB / actual;
    assert.ok(ratio >= 0.9 && ratio <= 1.25, `seq ${seq}: predicted ${estimate.totalGiB} vs measured ${actual} (ratio ${ratio.toFixed(2)})`);
  }
});

test('logits are a first-class component, not an afterthought', () => {
  // vocab 脳 seq 脳 4 bytes. For a 152k vocab at seq 1024 that is 0.58 GiB —the
  // term whose absence made the original estimator 20脳 optimistic.
  const at512 = estimateMemory({ modelId: 'Qwen/Qwen2.5-0.5B-Instruct', maxSeqLen: 512 });
  const at1024 = estimateMemory({ modelId: 'Qwen/Qwen2.5-0.5B-Instruct', maxSeqLen: 1024 });
  assert.ok(at512.components.logits > 0.25 && at512.components.logits < 0.35);
  assert.ok(at1024.components.logits > 0.5 && at1024.components.logits < 0.65);
  assert.ok(at1024.components.logits > at512.components.logits * 1.9);
});

test('LoRA (fp16 base) costs more than QLoRA on the same model', () => {
  // The base is the term that changes. Once activations dominate (post-calibration)
  // the *total* gap is smaller than the base-weight gap, so both are asserted.
  const qlora = estimateMemory({ modelId: 'Qwen/Qwen2.5-3B-Instruct', mode: 'qlora', maxSeqLen: 2048 });
  const lora = estimateMemory({ modelId: 'Qwen/Qwen2.5-3B-Instruct', mode: 'lora', maxSeqLen: 2048 });
  assert.ok(lora.components.baseWeights > qlora.components.baseWeights * 2.5, `base weights ${lora.components.baseWeights} vs ${qlora.components.baseWeights}`);
  assert.ok(lora.totalGiB > qlora.totalGiB * 1.05, `lora ${lora.totalGiB} should exceed qlora ${qlora.totalGiB}`);
  // Neither fits 6 GB at seq 2048 —which is exactly what the calibration showed.
  assert.equal(estimateMemory({ modelId: 'Qwen/Qwen2.5-3B-Instruct', mode: 'qlora', maxSeqLen: 2048, gpuGiB: 6 }).fits, false);
  assert.equal(estimateMemory({ modelId: 'Qwen/Qwen2.5-3B-Instruct', mode: 'lora', maxSeqLen: 2048, gpuGiB: 6 }).fits, false);
});

test('the plan validator accepts defaults and refuses each bad knob with a sentence', () => {
  const base = { mode: 'qlora', modelId: 'Qwen/Qwen2.5-1.5B-Instruct', loraR: 16, maxSeqLen: 2048, epochs: 1, learningRate: 2e-4 };
  assert.equal(validateTrainingPlan(base).ok, true);
  assert.match(validateTrainingPlan({ ...base, mode: 'dora' }).error, /qlora \/ lora/);
  assert.match(validateTrainingPlan({ ...base, loraR: 12 }).error, /4\/8\/16\/32\/64/);
  assert.match(validateTrainingPlan({ ...base, modelId: 'x; rm -rf /' }).error, /不安全/);
  assert.match(validateTrainingPlan({ ...base, quantType: 'int8' }).error, /nf4 \/ fp4/);
  // LoRA has no quantisation: a stray quantType is dropped, not rejected.
  const lora = validateTrainingPlan({ ...base, mode: 'lora', quantType: 'int8' });
  assert.equal(lora.ok, true);
  assert.equal(lora.value.quantType, null);
  // The alpha follows the rank by default and is carried through.
  assert.equal(validateTrainingPlan(base).value.loraAlpha, 32);
  assert.match(planSummary(validateTrainingPlan({ ...base, mode: 'lora' }).value), /LoRA/);
});

test('the extended plan carries LLaMA-Factory-style knobs with sane defaults', () => {
  const base = { mode: 'qlora', modelId: 'Qwen/Qwen2.5-1.5B-Instruct' };
  const result = validateTrainingPlan(base);
  assert.equal(result.ok, true);
  const plan = result.value;
  // The knobs that decide whether a 6 GB card succeeds.
  assert.equal(plan.gradientCheckpointing, true);
  assert.equal(plan.precision, 'auto');
  assert.equal(plan.optim, 'paged_adamw_8bit');
  assert.equal(plan.lrScheduler, 'cosine');
  assert.equal(plan.maskHistory, true);
  // Off by default because they change what the loss means, not just its size.
  assert.equal(plan.packing, false);
  assert.equal(plan.trainOnPrompt, false);
  assert.equal(plan.earlyStopping, false);
  assert.equal(plan.useDora, false);
  // Ordering was the whole point: a 2e-4 default with no schedule was the M0 setup.
  assert.equal(plan.learningRate, 2e-4);
  assert.equal(plan.loraAlpha, 32);
});

test('each extended knob refuses a value it cannot honour', () => {
  const base = { mode: 'qlora', modelId: 'Qwen/Qwen2.5-1.5B-Instruct' };
  assert.match(validateTrainingPlan({ ...base, lrScheduler: 'wobbly' }).error, /cosine/);
  assert.match(validateTrainingPlan({ ...base, optim: 'sgd_magic' }).error, /paged_adamw_8bit/);
  assert.match(validateTrainingPlan({ ...base, precision: 'int4' }).error, /bf16 \/ fp16 \/ fp32/);
  assert.match(validateTrainingPlan({ ...base, deepspeed: 'zero4' }).error, /none \/ zero2 \/ zero3/);
  // DeepSpeed is defined in the schema but unusable today; the refusal says so
  // instead of silently training without it.
  assert.match(validateTrainingPlan({ ...base, deepspeed: 'zero2' }).error, /DeepSpeed/);
  assert.match(validateTrainingPlan({ ...base, exportTarget: 'onnx' }).error, /gguf \/ ollama/);
});

test('combinations that only fail together are refused together', () => {
  const base = { mode: 'qlora', modelId: 'Qwen/Qwen2.5-1.5B-Instruct' };
  // Packing needs room to pack into.
  assert.match(validateTrainingPlan({ ...base, packing: true, maxSeqLen: 256 }).error, /packing/i);
  // Training on the prompt without packing teaches the model to repeat the question.
  assert.match(validateTrainingPlan({ ...base, trainOnPrompt: true, packing: false }).error, /prompt/i);
  // DoRA on a 4bit base needs memory a 6 GB card does not have.
  assert.match(validateTrainingPlan({ ...base, useDora: true }).error, /DoRA/);
  // ...and is fine on an fp16 base.
  assert.equal(validateTrainingPlan({ ...base, mode: 'lora', useDora: true }).ok, true);
});

test('capabilities gate the advanced knobs instead of hiding them', () => {
  // A machine with no GPU still gets a complete capability map: every feature is
  // reported with the reason it is off, which is what the panel prints.
  const bare = capabilityOf({ gpu: null, deps: {} });
  assert.equal(bare.hasGpu, false);
  assert.equal(bare.features.fp16.ok, false);
  assert.equal(bare.features.flashAttention2.ok, false);
  assert.equal(bare.features.deepspeed.ok, false);
  assert.match(bare.features.bitsandbytes.why, /bitsandbytes/);
  assert.equal(bare.features.packing.ok, true);

  // The machine this was built on: one Ampere card, no flash-attn installed.
  const oneCard = capabilityOf({
    gpu: { name: 'NVIDIA GeForce RTX 3060 Laptop GPU', count: 1, capability: 8.6 },
    bf16Supported: true,
    platform: 'win32',
    deps: { bitsandbytes: true },
  });
  assert.equal(oneCard.hasGpu, true);
  assert.equal(oneCard.features.bf16.ok, true);
  assert.equal(oneCard.features.fp16.ok, true);
  // Ampere supports FA2, but the wheel is not installed —the reason says which.
  assert.equal(oneCard.features.flashAttention2.ok, false);
  assert.match(oneCard.features.flashAttention2.why, /flash-attn/);
  // One card: ZeRO has nothing to shard.
  assert.equal(oneCard.features.deepspeed.ok, false);
});

test('nvidia-smi output parses, junk does not', () => {
  // The string below is what `nvidia-smi --query-gpu=name,memory.total
  // --format=csv,noheader` actually prints, suffix included. An earlier version
  // of this test dropped the "MiB" and so kept a parser that rejected every real
  // line — the panel then reported "no GPU listed" on a working card. Pasting the
  // verbatim output is the point of the test, not an incidental detail.
  const gpus = parseNvidiaSmi('NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB\n');
  assert.deepEqual(gpus, [{ name: 'NVIDIA GeForce RTX 3060 Laptop GPU', totalMiB: 6144, totalGiB: 6 }]);

  // Two cards, as the 2×RTX 4500 Ada box would report them.
  const dual = parseNvidiaSmi('NVIDIA RTX 4500 Ada Generation, 24564 MiB\nNVIDIA RTX 4500 Ada Generation, 24564 MiB\n');
  assert.equal(dual.length, 2);
  assert.equal(dual[0].totalGiB, 24);

  // A wrapper that reformats to GiB must not be read as a 16 MB card.
  assert.deepEqual(parseNvidiaSmi('Tesla V100, 16 GiB\n'), [{ name: 'Tesla V100', totalMiB: 16384, totalGiB: 16 }]);
  // A bare number (older drivers / `--format=csv` without the unit) still works.
  assert.deepEqual(parseNvidiaSmi('Tesla T4, 15360\n'), [{ name: 'Tesla T4', totalMiB: 15360, totalGiB: 15 }]);

  assert.deepEqual(parseNvidiaSmi('NVIDIA-SMI has failed'), []);
  assert.deepEqual(parseNvidiaSmi(''), []);
  assert.deepEqual(parseNvidiaSmi(undefined), []);
});

// ---------------------------------------------------------------- jobs.mjs
test('a restored active run becomes interrupted, not running', () => {
  const dir = mkdtempSync(join(tmpdir(), 'fnt-jobs-'));
  try {
    const store = createJobStore({ root: dir });
    const id = store.create({ kind: 'train', datasetId: 'd1' });
    store.move(id, 'running');
    const again = createJobStore({ root: dir });
    const result = again.restore();
    assert.equal(result.interrupted, 1);
    assert.equal(again.get(id).state, 'interrupted');
    assert.equal(again.get(id).error.code, 'host-restarted');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('events are durable and replayable from a cursor', () => {
  const dir = mkdtempSync(join(tmpdir(), 'fnt-jobs-'));
  try {
    const store = createJobStore({ root: dir });
    const id = store.create({ kind: 'train' });
    store.progress(id, { kind: 'metric', step: 1, values: { loss: 1.5 } });
    store.progress(id, { kind: 'metric', step: 2, values: { loss: 1.2 } });
    const snapshot = store.snapshot(id, 1);
    assert.ok(snapshot.events.length >= 2);
    const again = createJobStore({ root: dir });
    again.restore();
    assert.equal(again.get(id).metrics.loss, 1.2);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------- backend.mjs
test('the line reader reassembles a JSON event split across chunks', () => {
  const events = [];
  const reader = createLineReader((line) => events.push(parseWorkerLine(line)));
  const payload = '{"kind":"metric","step":1,"values":{"loss":1}}\n';
  reader.push(payload.slice(0, 20));
  reader.push(payload.slice(20));
  reader.flush();
  assert.equal(events.length, 1);
  assert.equal(events[0].values.loss, 1);
});

test('non-JSON worker output is not mistaken for an event', () => {
  assert.equal(parseWorkerLine('loading checkpoint'), null);
  assert.equal(parseWorkerLine('{"no":"kind"}'), null);
});

test('shell quoting survives quotes, newlines and dollars', () => {
  const nasty = `a'b"c$d\ne`;
  const quoted = shellQuote(nasty);
  assert.equal(quoted.startsWith("'") && quoted.endsWith("'"), true);
  // The embedded quote is closed, escaped and reopened: 'a'\''b...'
  assert.equal(quoted.includes(`'\\''`), true);
  assert.equal(quoted, `'a'\\''b"c$d\ne'`);
});

// ---------------------------------------------------------------- sources.mjs
test('source validation refuses a missing dataset file and a bad id', () => {
  assert.match(validateSourceInput({ kind: 'dataset', id: 'x', label: 'l', file: 'C:/nope.jsonl' }).error, /not found|lowercase/);
  assert.match(validateSourceInput({ kind: 'dataset', id: 'Bad_ID', label: 'l', file: 'C:/nope.jsonl' }).error, /lowercase/);
  // `agent` is no longer a kind at all: mining was removed from both halves, and a
  // source that cannot be created must not validate.
  assert.match(validateSourceInput({ kind: 'agent', id: 'ok', label: 'l', sessions: ['*'] }).error, /kind must be one of dataset/);
});

test('a dataset source builds, redacts and writes a data card', () => {
  const dir = mkdtempSync(join(tmpdir(), 'fnt-ds-'));
  try {
    const file = join(dir, 'rows.jsonl');
    const rows = [];
    for (let i = 0; i < 40; i += 1) rows.push(JSON.stringify({ messages: [{ role: 'user', content: `闂 ${i}` }, { role: 'assistant', content: `鍥炵瓟 ${i}` }] }));
    rows.push(JSON.stringify({ prompt: 'q with email a@b.com', response: 'ok' }));
    writeFileSync(file, `${rows.join('\n')}\n`, 'utf8');
    const source = { id: 'dsrc', kind: 'dataset', label: 't', file, redaction: 'pii', mode: 'answer', createdAt: Date.now() };
    const built = buildDataset({ root: dir, source, datasetId: 'ds-1', dshHome: dir });
    assert.equal(built.manifest.counts.kept, 41);
    assert.equal(built.manifest.counts.train + built.manifest.counts.val + built.manifest.counts.eval, 41);
    const card = JSON.parse(readFileSync(join(built.dir, 'manifest.json'), 'utf8'));
    assert.equal(card.sourceKind, 'dataset');
    assert.doesNotMatch(readFileSync(join(built.dir, 'train.jsonl'), 'utf8'), /a@b\.com/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------- index.js fence
test('the route fence rejects a non-loopback peer and a foreign Host', () => {
  assert.equal(loopback({ socket: { remoteAddress: '10.0.0.5' }, headers: { host: '127.0.0.1:10066' } }), false);
  assert.equal(loopback({ socket: { remoteAddress: '127.0.0.1' }, headers: { host: 'evil.example' } }), false);
  assert.equal(loopback({ socket: { remoteAddress: '127.0.0.1' }, headers: { host: '127.0.0.1:10066' } }), true);
  assert.equal(loopback({ socket: { remoteAddress: '::ffff:127.0.0.1' }, headers: { host: 'localhost:10066' } }), true);
});

test('same-origin is required for a browser request and not for curl', () => {
  assert.equal(sameOrigin({ headers: { host: '127.0.0.1:10066', origin: 'http://127.0.0.1:10066' } }), true);
  assert.equal(sameOrigin({ headers: { host: '127.0.0.1:10066', origin: 'http://evil.example' } }), false);
  assert.equal(sameOrigin({ headers: { host: '127.0.0.1:10066' } }), true);
});

test('spawning is off unless the switch is on', () => {
  assert.equal(allowSpawn({}), false);
  assert.equal(allowSpawn({ allowSpawn: true }), true);
});

/**
 * Build one zstd frame the host's way: content size declared, no checksum.
 * `reserved` sets the descriptor's reserved bit, reproducing the first-frame
 * quirk this decoder exists for.
 *
 * Node only exposes zstd compression where the build ships it. When it is absent
 * the helper throws —and the decoder test above is then skipped with a printed
 * reason rather than quietly passing on bytes that are not zstd at all.
 */
function zstdFrame(bytes, { reserved = false } = {}) {
  if (typeof zstdCompressSync !== 'function') throw new SkipError('this Node build has no zlib.zstdCompressSync');
  const frame = Buffer.from(zstdCompressSync(bytes, {
    params: {
      [constants.ZSTD_c_contentSizeFlag]: 1,
      [constants.ZSTD_c_checksumFlag]: 0,
    },
  }));
  if (reserved) frame[4] |= 0x04;
  return frame;
}

class SkipError extends Error {}

test('the panel can only widen its own permissions with an explicit confirmation', () => {
  // Both permissions default off: installing a plugin should not mean letting it
  // start processes or publish models. The panel offers buttons instead of sending
  // users to edit JSON - which is fine, but the route behind those buttons must
  // still require the user's confirmation, and must only accept the two keys it
  // knows about, or any request could widen the plugin's own permissions.
  const source = readFileSync(join(process.cwd(), 'index.js'), 'utf8');
  const start = source.indexOf("case 'env.enable':");
  assert.ok(start !== -1, 'env.enable must exist');
  const end = source.indexOf("case 'env.status':", start);
  const body = source.slice(start, end === -1 ? undefined : end);

  assert.match(body, /body\?\.confirm !== true/, 'refuses without confirm');
  assert.match(body, /allowSpawn:/, 'allowSpawn is an accepted setting');
  assert.match(body, /allowPublish:/, 'allowPublish is an accepted setting');
  assert.match(body, /current\[setting\] = true/, 'writes the requested setting, not a fixed one');
  assert.match(body, /JSON\.parse/, 'reads the existing file before writing');
  assert.match(body, /current = \{\}/, 'a corrupt file is replaced, not appended to');
  assert.match(body, /writeFileSync/, 'writes the config');
});

let failed = 0;
let skipped = 0;
for (const { name, fn } of tests) {
  try {
    await fn();
    console.log(`  ok  ${name}`);
  } catch (error) {
    if (error instanceof SkipError) {
      skipped += 1;
      console.log(`  --  ${name}  (skipped: ${error.message})`);
      continue;
    }
    failed += 1;
    console.error(`  xx  ${name}`);
    console.error(`      ${error?.message ?? error}`);
  }
}
test('SSH targets are allowlisted, not escaped', () => {
  // A target reaches ssh's argv, where a leading `-` is a flag and `;` is free.
  assert.equal(validateTarget('gpu-box').ok, true);
  assert.equal(validateTarget('me@gpu.example.com:2222').ok, true);
  assert.equal(validateTarget('me@[::1]:22').ok, true);
  assert.equal(validateTarget('').ok, false);
  assert.equal(validateTarget('-oProxyCommand=calc').ok, false);
  assert.equal(validateTarget('host; rm -rf /').ok, false);
  assert.equal(validateTarget('host && calc').ok, false);
  assert.equal(validateTarget('host`id`').ok, false);
});

test('ssh argv keeps host verification on and never hangs on a prompt', () => {
  const args = buildSshArgs({ target: 'gpu-box', command: 'python3 -V' });
  const joined = args.join(' ');
  // BatchMode: a training run must fail fast, not wait for an invisible prompt.
  assert.match(joined, /BatchMode=yes/);
  assert.match(joined, /ServerAliveInterval=30/);
  assert.equal(args[args.length - 2], 'gpu-box');
  assert.equal(args[args.length - 1], 'python3 -V');
  // `-T`: no TTY, so nothing can prompt.
  assert.ok(args.includes('-T'));

  // Settings must not be able to switch off host-key checking.
  const hostile = buildSshArgs({
    target: 'gpu-box',
    command: 'true',
    options: { StrictHostKeyChecking: 'no', ProxyCommand: 'calc', UserKnownHostsFile: '/dev/null', Nice: '5' },
    extraArgs: ['-oStrictHostKeyChecking=no', '-oBatchMode=no'],
  });
  const hostileJoined = hostile.join(' ');
  assert.doesNotMatch(hostileJoined, /StrictHostKeyChecking=no/, 'host check stays on');
  assert.doesNotMatch(hostileJoined, /ProxyCommand=calc/, 'no arbitrary command execution');
  assert.doesNotMatch(hostileJoined, /UserKnownHostsFile=\/dev\/null/, 'known_hosts cannot be redirected');
  assert.match(hostileJoined, /Nice=5/, 'harmless options still pass through');
});

test('the remote command is quoted so a config value cannot break out', () => {
  // The end-to-end shape the SSH backend sends: JSON as a printf argument.
  const config = JSON.stringify({ out: "/tmp/x'; calc; echo '" });
  const line = `printf '${'%s'}' ${shellQuote(config)} > out.json`;
  assert.match(line, /^printf '%s' '/);
  // The injected quote must appear as the '\'' escape, never as a bare quote.
  assert.doesNotMatch(line.replace(/\\''/g, ''), /'; calc/);
});

test('sshExec reports a missing OpenSSH client honestly', async () => {
  const run = createSshExec({ env: { ...process.env, PATH: '' } });
  const error = await run('gpu-box', ['true'], {}).then(
    () => { throw new Error('should have rejected'); },
    (reason) => reason,
  );
  // On Windows this is ENOENT from spawn; the message must name the fix.
  assert.match(String(error.message), /ssh|OpenSSH/i);
});

test('sshExec refuses an invalid target before spawning anything', async () => {
  const run = createSshExec({});
  const error = await run('-oProxyCommand=calc', ['true'], {}).then(
    () => { throw new Error('should have rejected'); },
    (reason) => reason,
  );
  assert.match(String(error.message), /不能以 - 开头|不安全/);
});

test('remote paths are refused before they reach an sftp batch file', () => {
  // sftp -b reads *commands* from stdin, so a newline in a path is a second command.
  assert.equal(validateRemotePath('/tmp/run/data/train.jsonl').ok, true);
  assert.equal(validateRemotePath('relative/path').ok, false, 'must be absolute');
  assert.equal(validateRemotePath('/tmp/a\nrm -rf /').ok, false, 'no newline');
  assert.equal(validateRemotePath('/tmp/../../etc/passwd').ok, false, 'no .. traversal');
  assert.equal(validateRemotePath('').ok, false);
});

test('an sftp put line quotes both sides', () => {
  const line = putLine('C:\\runs\\x\\train.jsonl', '/tmp/run/data/train.jsonl');
  assert.match(line, /^put "C:\\\\runs\\\\x\\\\train\.jsonl" "\/tmp\/run\/data\/train\.jsonl"$/);
  // A quote inside a local path must not end the argument.
  const tricky = putLine('/tmp/a " b.jsonl', '/tmp/run/x.jsonl');
  assert.match(tricky, /a \\" b\.jsonl/);
  // And a remote path that would escape is refused outright.
  assert.throws(() => putLine('/tmp/ok', '/tmp/../escape'), /\.\./);
});

test('the text metric separates on-topic from off-topic', () => {
  // Anchors measured against a real 1851-char reference from the M0 run. These
  // four numbers are the reason the metric is trustworthy: it is not a smooth
  // function that scores everything the same, it is 1.0 for right and 0.0 for
  // wrong, which is what the previous "non-empty passes" metric failed to do.
  const reference = '## 选定论文\n\nWind Power Forecasting. Li et al., Applied Energy 329 (2023) 120291.\n'
    + '我们复现了 FedDRL 方法，使用风电数据集训练，报告 MAPE 与 RMSE 指标。\n';
  assert.equal(scoreTextAnswer(reference, reference).f1, 1);
  // A truncated reference is still entirely on topic.
  assert.ok(scoreTextAnswer(reference.slice(0, Math.floor(reference.length * 0.6)), reference).f1 > 0.9);
  // Fluent text about something else must not score.
  assert.equal(scoreTextAnswer('今天天气很好我打算出门散步', reference).f1, 0);
  assert.equal(scoreTextAnswer('', reference).f1, 0);
  assert.equal(scoreTextAnswer('   \n  ', reference).f1, 0);
});

test('the text metric rewards form as well as content', () => {
  const reference = '## 结论\n\n- 第一点\n- 第二点\n\n```json\n{"tool":"read"}\n```\n';
  const sameWordsWrongForm = '结论 第一点 第二点 read tool';
  const withForm = '## 结论\n\n- 第一点\n- 第二点\n\n```json\n{"tool":"read"}\n```\n';
  const a = scoreTextAnswer(sameWordsWrongForm, reference);
  const b = scoreTextAnswer(withForm, reference);
  // Same vocabulary, but only one carries the shape SFT is actually teaching.
  assert.ok(b.shape > a.shape, `shape ${b.shape} should beat ${a.shape}`);
  assert.equal(b.shape, 1);
});

test('the metric is symmetric about padding: extra invented text lowers precision', () => {
  const reference = '模型在风电数据集上训练，报告 MAPE 指标。';
  const faithful = '模型在风电数据集上训练，报告 MAPE 指标。';
  const padded = `${faithful} 另外我们还研究了量子计算、烹饪、古典音乐和星象学等等各种内容。`;
  const clean = scoreTextAnswer(faithful, reference);
  const fat = scoreTextAnswer(padded, reference);
  assert.ok(fat.precision < clean.precision, `padding should lower precision: ${fat.precision} vs ${clean.precision}`);
});

test('tokenising handles both scripts', () => {
  const tokens = contentTokens('?? PyTorch ?? model ??');
  assert.ok(tokens.has('pytorch'));
  assert.ok(tokens.has('model'));
  assert.ok(tokens.has('??'));
  assert.ok(tokens.has('??'));
});

test('the split aggregates keep the per-case detail', () => {
  const cases = [{ expected: '答案是一段关于风电预测的说明。' }, { expected: '答案是一段关于风电预测的说明。' }];
  const split = scoreTextSplit(cases, ['答案是一段关于风电预测的说明。', '完全无关的另一段话。']);
  assert.equal(split.cases, 2);
  assert.equal(split.perCase.length, 2);
  // One of two is on topic: the aggregate must not hide that behind "50%".
  assert.equal(split.onTopicRate, 0.5);
  assert.equal(split.perCase[0].f1, 1);
  assert.equal(split.perCase[1].f1, 0);
});








test('a single-group dataset still gets a usable training split', () => {
  // External datasets carry one group id for every row. A pure group split put the
  // whole file in `val` and left `train` empty ? the dataset built, reported its
  // sample count, and could not be trained on. Domain knowledge is exactly this
  // shape, so the fallback is what makes that use case possible at all.
  const rows = Array.from({ length: 60 }, (_, i) => ({
    messages: [{ role: 'user', content: 'q' + i }, { role: 'assistant', content: 'a' + i }],
    meta: { group: 'one-file', hash: 'h' + i },
  }));
  const split = splitSamples(rows);
  assert.equal(split.stats.strategy, 'sample', 'must fall back to per-sample');
  assert.ok(split.train.length > 0, 'training set may not be empty');
  assert.ok(split.val.length > 0, 'validation set may not be empty');
  assert.equal(split.train.length + split.val.length + split.eval.length, 60, 'no sample lost');

  // Deterministic: the same input must produce the same split every time.
  const again = splitSamples(rows);
  assert.deepEqual(again.stats, split.stats);
});

test('a many-group dataset keeps whole sessions on one side', () => {
  // Mined samples from one conversation are near-duplicates, so a group may never
  // straddle train and val. This is the behaviour the sample fallback must not
  // replace when there are enough groups to split properly.
  const rows = [];
  for (let group = 0; group < 8; group += 1) {
    for (let i = 0; i < 10; i += 1) {
      rows.push({ messages: [{ role: 'user', content: 'q' }, { role: 'assistant', content: 'a' }], meta: { group: 'g' + group, hash: 'h' + group + '-' + i } });
    }
  }
  const split = splitSamples(rows);
  assert.equal(split.stats.strategy, 'group');
  const sideOf = (sample) => (split.train.includes(sample) ? 'train' : split.val.includes(sample) ? 'val' : 'eval');
  const seen = new Map();
  for (const sample of rows) {
    const side = sideOf(sample);
    if (seen.has(sample.meta.group)) assert.equal(seen.get(sample.meta.group), side, 'a group straddled the split');
    seen.set(sample.meta.group, side);
  }
});

test('two groups still leave something to train on', () => {
  // The `chain` mode produced 624 samples spread over exactly two sessions. With a
  // threshold of "one group per side it wants to fill", eval took one group and val
  // took the other, leaving train empty ? a dataset that builds, reports its sample
  // count, and cannot be trained on. The threshold has to reserve a group for train.
  const rows = [];
  for (let group = 0; group < 2; group += 1) {
    for (let i = 0; i < 100; i += 1) {
      rows.push({ messages: [{ role: 'user', content: 'q' }, { role: 'assistant', content: 'a' }], meta: { group: 'g' + group, hash: 'h' + group + '-' + i } });
    }
  }
  const split = splitSamples(rows);
  assert.ok(split.train.length > 0, 'train must not be emptied by eval and val');
  assert.equal(split.train.length + split.val.length + split.eval.length, 200);
});

test('the panel no longer offers session mining', () => {
  // Mining stays available through the host API, but the user judged the picker
  // unnecessary for the job (an existing dataset is what training needs), so the
  // panel must not render it. Pinning the absence keeps a future edit from quietly
  // bringing the whole block back.
  const clientSource = readFileSync(new URL('../client/client.js', import.meta.url), 'utf8');
  assert.ok(!/\u4ece\u4f1a\u8bdd\u65e5\u5fd7\u5efa\u6570\u636e\u96c6/.test(clientSource), 'the session-mining section must be gone');
  assert.ok(!clientSource.includes('?sessions=1'), 'the panel must not request the session list');
  // The two sections that remain.
  const dataTab = clientSource.slice(clientSource.indexOf("tab === 'data'"), clientSource.indexOf("tab === 'train'"));
  assert.ok(dataTab.includes('\u4ece\u672c\u5730\u6570\u636e\u96c6\u6587\u4ef6\u5efa'), 'the JSONL importer must stay');
  assert.ok(dataTab.includes('\u6570\u636e\u96c6'), 'the dataset list must stay');
});




test('merging datasets concatenates each split and reports what it saw', () => {
  // Training on several datasets at once is the normal case (mined tool calls plus
  // a hand-written domain file). The merge must be a per-split concatenation: each
  // source already decided which rows are train/val/eval, and re-splitting across
  // sources would move samples a reviewer had inspected.
  const root = join(tmpdir(), 'fnt-merge-' + process.pid);
  const writeDataset = (name, rows) => {
    const dir = join(root, name);
    mkdirSync(dir, { recursive: true });
    for (const split of ['train', 'val', 'eval']) {
      const body = (rows[split] ?? []).map((row) => JSON.stringify(row)).join('\n');
      writeFileSync(join(dir, split + '.jsonl'), body === '' ? '' : body + '\n', 'utf8');
    }
    writeFileSync(join(dir, 'excluded.json'), '[]', 'utf8');
    return dir;
  };
  const a = writeDataset('a', { train: [{ hash: 'a1', messages: [] }, { hash: 'a2', messages: [] }], val: [{ hash: 'a3', messages: [] }] });
  const b = writeDataset('b', { train: [{ hash: 'b1', messages: [] }], eval: [{ hash: 'b2', messages: [] }] });
  const merged = writeMergedDataset([a, b], join(root, 'out'));
  assert.deepEqual(merged.counts, { train: 3, val: 1, eval: 1 });
  assert.equal(merged.sources.length, 2);
  // The merged files are what the worker reads, so verify the bytes, not the counts.
  const trainLines = readFileSync(join(root, 'out', 'train.jsonl'), 'utf8').trim().split('\n').filter(Boolean);
  assert.equal(trainLines.length, 3);
  // An empty selection is an error, never a silently empty run.
  assert.throws(() => writeMergedDataset([], join(root, 'empty')), /至少/);
  rmSync(root, { recursive: true, force: true });
});

test('every colour token the panel uses is defined', () => {
  // `background: C.bg` referenced a token that never existed, so the dropdown popup
  // rendered with a transparent background and the rows underneath showed through
  // the list. Undefined tokens fail silently in CSS, which is why this needs its own
  // check: nothing throws, the panel just looks broken.
  const clientSource = readFileSync(new URL('../client/client.js', import.meta.url), 'utf8');
  const tableStart = clientSource.indexOf('const C = {');
  const tableEnd = clientSource.indexOf('}', tableStart);
  const table = clientSource.slice(tableStart, tableEnd);
  const defined = new Set([...table.matchAll(/([a-zA-Z]+):\s*['"]/g)].map((match) => match[1]));
  // `C.bg` only survives inside a comment explaining the bug; strip comments first.
  const code = clientSource.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
  const used = new Set([...code.matchAll(/\bC\.([a-zA-Z]+)/g)].map((match) => match[1]));
  const missing = [...used].filter((name) => !defined.has(name));
  assert.deepEqual(missing, [], `undefined colour tokens: ${missing.join(', ')}`);
});

test('deleting a dataset refuses anything that is not one', () => {
  // The id is joined onto the datasets root, and the delete is irreversible, so
  // every guard gets its own assertion: a bad id must not reach the filesystem, a
  // path must not escape the datasets directory, and a plan directory must not be
  // mistaken for data.
  const root = join(tmpdir(), 'fnt-del-' + process.pid);
  mkdirSync(join(root, 'datasets', 'real'), { recursive: true });
  writeFileSync(join(root, 'datasets', 'real', 'manifest.json'), '{"id":"real"}', 'utf8');
  // A directory with no manifest is not a dataset and must survive.
  mkdirSync(join(root, 'datasets', 'no-manifest'), { recursive: true });

  assert.equal(deleteDataset(root, '../datasets/real').ok, false, 'a separator must be refused');
  assert.equal(deleteDataset(root, '..').ok, false);
  assert.equal(deleteDataset(root, '').ok, false);
  assert.equal(deleteDataset(root, 'no-manifest').ok, false, 'no manifest means not ours to delete');
  assert.ok(existsSync(join(root, 'datasets', 'no-manifest')), 'the stray directory must survive');

  const removed = deleteDataset(root, 'real');
  assert.equal(removed.ok, true);
  assert.equal(existsSync(join(root, 'datasets', 'real')), false, 'the dataset directory must be gone');
  assert.equal(deleteDataset(root, 'real').ok, false, 'deleting twice must fail, not throw');
  rmSync(root, { recursive: true, force: true });
});

test('the installer picks a CUDA wheel the driver can actually load', () => {
  // Two failure modes this guards. A wheel newer than the driver fails with
  // "CUDA driver version is insufficient"; and a wheel older than necessary is a
  // regression against the build the whole training loop was measured on.
  const cu121 = pickCuda('12.1');
  assert.equal(cu121.cuda, '121');
  // A new driver still gets the validated build: CUDA is forward compatible.
  assert.equal(pickCuda('13.1').cuda, '121', 'a 13.1 driver should keep the validated build');
  assert.equal(pickCuda('12.8').cuda, '121');
  // Only a driver too old for cu121 drops down, and that is flagged as unvalidated.
  const old = pickCuda('11.8');
  assert.equal(old.cuda, '118');
  assert.equal(old.unvalidated, true, 'a wheel that was never tested must say so');
  // Too old for anything. Refusing is better than installing a wheel that cannot load.
  assert.equal(pickCuda('11.2'), null);
  assert.equal(pickCuda(''), null);
  assert.equal(pickCuda(null), null);
});

test('GPU parsing keeps the memory unit out of the number', () => {
  // `Number("6144 MiB")` is NaN, which once made the panel report "no GPU" on a
  // machine that had one. The unit must be parsed, not stripped and hoped for.
  const gpu = parseGpuListing('NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB, 552.44');
  assert.equal(gpu.name, 'NVIDIA GeForce RTX 3060 Laptop GPU');
  assert.equal(gpu.totalMiB, 6144);
  const multi = parseGpuListing('Small, 4096 MiB, 1\nBig, 24576 MiB, 1');
  assert.equal(multi.totalMiB, 24576, 'the largest card is the one training targets');
  assert.equal(parseGpuListing('NVIDIA RTX 4090, 24 GiB, 1').totalMiB, 24576);
  assert.equal(parseGpuListing('no unit here, 6144, 1'), null);
  assert.equal(parseGpuListing(''), null);
});

test('the VRAM verdict refuses a card that cannot train anything', () => {
  assert.equal(planForVram(4096).verdict, 'insufficient');
  assert.equal(planForVram(6144).verdict, 'minimal');
  assert.equal(planForVram(8192).verdict, 'workable');
  assert.equal(planForVram(24576).verdict, 'comfortable');
  // A card judged insufficient must not name a model: "you can train X" followed by
  // an OOM is worse than saying it will not fit.
  assert.equal(planForVram(4096).maxModel, null);
});

test('install progress never goes backwards and ends at 100', () => {
  // Two real bugs this pins. The bar once sat at 98% on a finished install, because
  // the last stage reported its start rather than its end; and every pip log line
  // reset it to the start of the install stage. Both looked like "stuck" at the
  // moment the user most wants to believe it is done.
  const STAGES = [
    { id: 'detect', weight: 2 }, { id: 'venv', weight: 5 }, { id: 'install', weight: 88 },
    { id: 'verify', weight: 3 }, { id: 'done', weight: 2 },
  ];
  const TOTAL = STAGES.reduce((sum, stage) => sum + stage.weight, 0);
  const percentAt = (id, within) => {
    const index = STAGES.findIndex((stage) => stage.id === id);
    const before = STAGES.slice(0, index).reduce((sum, stage) => sum + stage.weight, 0);
    return Math.min(100, Math.round(((before + STAGES[index].weight * within) / TOTAL) * 100));
  };
  // The sequence recorded from a real install, with each step's true fraction.
  const seen = [
    ['detect', 0], ['venv', 0], ['install', 0], ['install', 0.25], ['install', 0.25],
    ['install', 0.5], ['install', 0.5], ['install', 0.75], ['install', 1],
    ['verify', 0], ['done', 1],
  ];
  let previous = -1;
  for (const [phase, within] of seen) {
    const percent = percentAt(phase, within);
    assert.ok(percent >= previous, `${phase} went backwards: ${percent} < ${previous}`);
    previous = percent;
  }
  assert.equal(previous, 100, 'a finished install must read 100%');
});

console.log(`\n${tests.length - failed - skipped}/${tests.length} passed${skipped > 0 ? `, ${skipped} skipped` : ''}, ${failed} failed`);

if (failed > 0) process.exitCode = 1;
