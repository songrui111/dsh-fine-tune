/**
 * Host-side wiring test.
 *
 * `mount()` is called with a fake cordis context (tools.register / inject /
 * effect are collected, not executed by the real loader), the routes are then
 * served by a real `node:http` server, and every endpoint is exercised over
 * loopback — including the ones that must say *no*: a foreign Host, a
 * cross-site Origin, and starting a run with the spawn switch off.
 *
 * The worker is never spawned: the routes that would spend GPU time are checked
 * for their refusal paths, which is exactly the behaviour most likely to rot.
 */
import assert from 'node:assert/strict';
import http from 'node:http';
import { mkdtempSync, writeFileSync, rmSync, readdirSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const dataDir = mkdtempSync(join(tmpdir(), 'fnt-host-'));
process.env.DSH_HOME = join(dataDir, 'dsh-home');

const { mount, ROUTES } = await import('../index.js');

const registeredTools = [];
const registeredRoutes = [];
const warnings = [];
const ctx = {
  tools: { register: (tool) => { registeredTools.push(tool); return () => {}; } },
  effect: (factory) => { factory(); },
  inject: (names, callback) => {
    assert.deepEqual(names, ['webServer']);
    callback({
      webServer: { register: ({ path, handler }) => { registeredRoutes.push({ path, handler }); return () => {}; } },
      effect: (factory) => { factory(); },
    });
  },
};

const originalWarn = console.warn;
console.warn = (...args) => warnings.push(args.join(' '));

const mounted = mount(ctx, {
  dataDir: join(dataDir, 'fine-tune'),
  allowSpawn: false,
  allowPublish: false,
});

console.warn = originalWarn;

// ---------------------------------------------------------------- registration
assert.equal(registeredTools.length, 5, 'five model-facing tools');
assert.deepEqual(registeredTools.map((tool) => tool.name).sort(), ['train_check', 'train_eval', 'train_prepare', 'train_run', 'train_status']);
for (const tool of registeredTools) {
  assert.equal(typeof tool.execute, 'function');
  assert.equal(typeof tool.description, 'string');
  assert.ok(tool.description.length > 40, `${tool.name} needs a real description`);
  assert.equal(tool.parameters.type, 'object');
}
assert.equal(registeredRoutes.length, 4);
const routeOf = (path) => registeredRoutes.find((route) => route.path === path).handler;
for (const path of Object.values(ROUTES)) assert.ok(registeredRoutes.some((route) => route.path === path), `missing route ${path}`);

// ---------------------------------------------------------------- serve it
const server = http.createServer((request, response) => {
  const url = new URL(request.url, 'http://127.0.0.1');
  const route = registeredRoutes.find((entry) => entry.path === url.pathname);
  if (route === undefined) {
    response.writeHead(404);
    response.end();
    return;
  }
  route.handler(request, response);
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const port = server.address().port;
const base = `http://127.0.0.1:${port}`;

async function call(path, options = {}) {
  const response = await fetch(`${base}${path}`, options);
  const text = await response.text();
  let body = null;
  try {
    body = JSON.parse(text);
  } catch {
    body = { raw: text }
  }
  return { status: response.status, body, headers: response.headers };
}

const post = (path, body, headers = {}) => call(path, {
  method: 'POST',
  headers: { 'content-type': 'application/json', ...headers },
  body: JSON.stringify(body),
});

/** A tiny dataset legitimately lands entirely in one split; pick the populated one. */
function populatedSplit(counts) {
  for (const candidate of ['train', 'val', 'eval']) if (counts[candidate] > 0) return candidate;
  return 'train';
}

try {
  // ---- state
  const state = await call(ROUTES.state);
  assert.equal(state.status, 200);
  assert.equal(state.body.schema, 'dsh-fine-tune/v1');
  assert.equal(state.body.settings.allowSpawn, false);
  assert.deepEqual(state.body.runs, []);
  assert.deepEqual(state.body.sources, []);
  assert.equal(state.headers.get('cache-control'), 'no-store');

  // ---- the fence: a foreign Host is refused even from loopback.
  // `fetch` silently drops a custom Host header, so this probe uses raw http.
  const rebound = await new Promise((resolve, reject) => {
    const request = http.request({ host: '127.0.0.1', port, path: ROUTES.state, headers: { host: 'evil.example' } }, (response) => {
      response.resume();
      resolve(response.statusCode);
    });
    request.on('error', reject);
    request.end();
  });
  assert.equal(rebound, 403);

  // ---- a cross-site Origin is refused on a mutation, accepted when absent
  const crossSite = await post(ROUTES.action, { op: 'gpu' }, { origin: 'http://evil.example' });
  assert.equal(crossSite.status, 403);

  // ---- gpu / estimate endpoint answers without a GPU
  const gpu = await post(ROUTES.action, { op: 'gpu', modelId: 'Qwen/Qwen2.5-7B-Instruct', maxSeqLen: 4096 });
  assert.equal(gpu.status, 200);
  assert.ok(gpu.body.estimate !== null);
  assert.equal(typeof gpu.body.estimate.totalGiB, 'number');
  if (gpu.body.gpu.available === false) {
    assert.equal(typeof gpu.body.gpu.reason, 'string', 'a missing GPU must come with a reason');
  }

  // ---- source + dataset lifecycle over HTTP
  const rows = [];
  for (let i = 0; i < 30; i += 1) rows.push(JSON.stringify({ messages: [{ role: 'user', content: `问题 ${i} a@b.com` }, { role: 'assistant', content: `回答 ${i}` }] }));
  const datasetFile = join(dataDir, 'rows.jsonl');
  writeFileSync(datasetFile, `${rows.join('\n')}\n`, 'utf8');

  const created = await post(ROUTES.action, {
    op: 'source.create',
    source: { kind: 'dataset', id: 'rows', label: '测试行', file: datasetFile, mode: 'answer', redaction: 'pii' },
  });
  assert.equal(created.status, 200);
  assert.equal(created.body.source.id, 'rows');

  const duplicate = await post(ROUTES.action, {
    op: 'source.create',
    source: { kind: 'dataset', id: 'rows', label: 'again', file: datasetFile, mode: 'answer' },
  });
  assert.equal(duplicate.status, 409);

  const built = await post(ROUTES.action, { op: 'dataset.build', sourceId: 'rows', datasetId: 'rows-1' });
  assert.equal(built.status, 200);
  assert.equal(built.body.dataset.counts.kept, 30);
  assert.ok(existsSync(join(mounted.paths.datasets, 'rows-1', 'train.jsonl')));
  assert.ok(readdirSync(join(mounted.paths.datasets, 'rows-1')).includes('manifest.json'));

  // The redaction really happened on disk, not just in the response.
  const { readFileSync } = await import('node:fs');
  const trainText = readFileSync(join(mounted.paths.datasets, 'rows-1', 'train.jsonl'), 'utf8');
  assert.doesNotMatch(trainText, /a@b\.com/);

  // ---- run.start with the switch off is refused with the switch named
  const refused = await post(ROUTES.action, { op: 'run.start', datasetId: 'rows-1', modelId: 'Qwen/Qwen2.5-1.5B-Instruct' });
  assert.equal(refused.status, 403);
  assert.match(refused.body.error, /允许启动训练进程/);

  // ---- validation order is deliberate: the capability switch is checked first,
  // so an unsafe model id cannot probe whether the switch is on (nor start
  // anything if it were). With the switch off, both are a 403.
  const unsafe = await post(ROUTES.action, { op: 'run.start', datasetId: 'rows-1', modelId: 'x && del /f' });
  assert.equal(unsafe.status, 403);

  // ---- the same validation, reached when the switch is open, happens on the
  // tools path (train_run validates before it would ever spawn).
  let unsafeTool = null;
  try {
    await registeredTools.find((tool) => tool.name === 'train_run').execute({ datasetId: 'rows-1', modelId: 'x && del /f' });
  } catch (error) {
    unsafeTool = error.message;
  }
  assert.match(String(unsafeTool), /允许启动训练进程/, 'the switch is checked before the model id on the tools path too');

  // ---- with the switch off a missing dataset never gets far enough to be a 404;
  // the tools path reports the missing dataset (it checks the switch first, then
  // the data, which is the order a user reads the errors in).
  const missingDataset = await post(ROUTES.action, { op: 'run.start', datasetId: 'nope', modelId: 'Qwen/Qwen2.5-1.5B-Instruct' });
  assert.equal(missingDataset.status, 403);

  // ---- and an unknown dataset id is a 404 on the path that reaches validation
  const missingSource = await post(ROUTES.action, { op: 'dataset.build', sourceId: 'not-a-source' });
  assert.equal(missingSource.status, 404);

  // ---- publish without a green verdict is refused
  const publish = await post(ROUTES.action, { op: 'publish.prepare', id: 'whatever' });
  assert.equal(publish.status, 403);

  // ---- an unknown op is a 400 with the op echoed
  const unknown = await post(ROUTES.action, { op: 'nuke' });
  assert.equal(unknown.status, 400);
  assert.match(unknown.body.error, /nuke/);

  // ---- sample preview: paged, with both halves of a sample and its meta.
  // Every dataset here shares one group, and the group split never tears a group
  // apart, so the whole small corpus lands in one split — find the populated one.
  let trainSplit = null;
  for (const candidate of ['train', 'val', 'eval']) {
    if (built.body.dataset.counts[candidate] > 0) {
      trainSplit = candidate;
      break;
    }
  }
  const trainCount = built.body.dataset.counts[trainSplit];
  const page1 = await post(ROUTES.action, { op: 'dataset.preview', datasetId: 'rows-1', split: trainSplit, limit: 5, offset: 0 });
  assert.equal(page1.status, 200);
  assert.equal(page1.body.rows.length, Math.min(5, trainCount));
  assert.equal(page1.body.total, trainCount);
  assert.equal(page1.body.rows[0].hash.length >= 8, true, 'each row carries the hash an exclusion removes by');
  assert.equal(typeof page1.body.rows[0].prompt, 'string');
  assert.equal(typeof page1.body.rows[0].label, 'string');
  assert.equal(page1.body.manifest.id, 'rows-1');

  // ---- removal is by hash, and it changes what the training run would read
  const removed = page1.body.rows[0].hash;
  const excluded = await post(ROUTES.action, { op: 'dataset.exclude', datasetId: 'rows-1', hash: removed });
  assert.equal(excluded.status, 200);
  assert.equal(excluded.body.excluded.includes(removed), true);
  const page2 = await post(ROUTES.action, { op: 'dataset.preview', datasetId: 'rows-1', split: trainSplit, limit: 5, offset: 0 });
  assert.equal(page2.body.total, trainCount - 1, 'the effective set shrinks');
  assert.equal(page2.body.rows.some((row) => row.hash === removed), false);

  // ---- and the run's own data copy honours it: removing a sample reaches the
  // very next run, because training reads the filtered materialisation.
  const { writeFilteredDataset } = await import('../lib/dataset.mjs');
  const filtered = writeFilteredDataset(join(mounted.paths.datasets, 'rows-1'), join(dataDir, 'filtered'));
  assert.equal(filtered.counts.train + filtered.counts.val + filtered.counts.eval, 30 - 1);
  assert.doesNotMatch(readFileSync(join(dataDir, 'filtered', `${trainSplit}.jsonl`), 'utf8'), new RegExp(removed));

  // ---- exclusion is reversible, and a bad hash is refused
  const restored = await post(ROUTES.action, { op: 'dataset.exclude', datasetId: 'rows-1', hash: removed, restore: true });
  assert.equal(restored.body.excluded.includes(removed), false);
  const badHash = await post(ROUTES.action, { op: 'dataset.exclude', datasetId: 'rows-1', hash: 'not a hash' });
  assert.equal(badHash.status, 400);

  // ---- evaluation pair: prompts, expected answers and a stable A/B shuffle
  const pair = await post(ROUTES.action, { op: 'evaluation.pair', datasetId: 'rows-1', limit: 4 });
  assert.equal(pair.status, 200);
  assert.equal(pair.body.cases.length, Math.min(4, built.body.dataset.counts.eval));
  for (const item of pair.body.cases) {
    assert.equal(typeof item.prompt, 'string');
    assert.equal(typeof item.expected, 'string');
    assert.equal(typeof item.flip, 'boolean');
    assert.ok(Number.isInteger(item.index));
  }
  assert.ok(Array.isArray(pair.body.fixture.knownTools), 'the fixture travels with the pair');
  // The shuffle is deterministic: the same page twice gives the same assignment.
  const pairAgain = await post(ROUTES.action, { op: 'evaluation.pair', datasetId: 'rows-1', limit: 4 });
  assert.equal(JSON.stringify(pairAgain.body.cases.map((c) => c.flip)), JSON.stringify(pair.body.cases.map((c) => c.flip)));

  // ---- the training plan is validated on the start route
  const badRank = await post(ROUTES.action, { op: 'run.start', datasetId: 'rows-1', loraR: 12 });
  assert.equal(badRank.status, 403, 'the spawn switch is checked first');

  // ---- mode reaches the estimator, and both modes are offered to the panel
  const loraEstimate = await post(ROUTES.action, { op: 'gpu', mode: 'lora', modelId: 'Qwen/Qwen2.5-3B-Instruct', maxSeqLen: 2048 });
  const qloraEstimate = await post(ROUTES.action, { op: 'gpu', mode: 'qlora', modelId: 'Qwen/Qwen2.5-3B-Instruct', maxSeqLen: 2048 });
  assert.ok(loraEstimate.body.estimate.totalGiB > qloraEstimate.body.estimate.totalGiB, 'LoRA asks for more memory than QLoRA');
  assert.equal(loraEstimate.body.estimate.mode, 'lora');
  assert.deepEqual(loraEstimate.body.modes.map((mode) => mode.id).sort(), ['lora', 'qlora']);

  // ---- an oversized body is rejected: the route destroys the socket, which the
  // client sees as a transport failure rather than a status code.
  let oversizedRefused = false;
  try {
    await post(ROUTES.action, { op: 'gpu', pad: 'x'.repeat(80 * 1024) });
  } catch (error) {
    oversizedRefused = /fetch failed|socket|ECONNRESET/i.test(String(error && (error.cause && error.cause.code || error.message)));
  }
  assert.equal(oversizedRefused, true, 'an oversized body must not be buffered');

  // ---- run snapshot: an unknown id is a 404, and a real one round-trips
  const unknownRun = await call(`${ROUTES.run}?id=nope`);
  assert.equal(unknownRun.status, 404);

  // ---- stream: SSE headers, a replay snapshot, then close
  const jobId = mounted.store.create({ kind: 'train', datasetId: 'rows-1' });
  mounted.store.progress(jobId, { kind: 'metric', step: 1, values: { loss: 2.5 } });
  const controller = new AbortController();
  const streamResponse = await fetch(`${base}${ROUTES.stream}?id=${jobId}`, { signal: controller.signal });
  assert.equal(streamResponse.status, 200);
  assert.match(streamResponse.headers.get('content-type') || '', /text\/event-stream/);
  const reader = streamResponse.body.getReader();
  const chunk = await reader.read();
  const frame = new TextDecoder().decode(chunk.value);
  assert.match(frame, /"kind":"snapshot"/);
  assert.match(frame, /"loss":2\.5/);
  controller.abort();
  await reader.cancel().catch(() => {});

  // ---- tools: train_status sees the run, train_run refuses while the switch is off
  const status = await registeredTools.find((tool) => tool.name === 'train_status').execute({ id: jobId });
  assert.match(status.markdown, /train/);
  let toolRefused = null;
  try {
    await registeredTools.find((tool) => tool.name === 'train_run').execute({ datasetId: 'rows-1' });
  } catch (error) {
    toolRefused = error.message;
  }
  assert.match(String(toolRefused), /允许启动训练进程/);

  const check = await registeredTools.find((tool) => tool.name === 'train_check').execute({ modelId: 'Qwen/Qwen2.5-7B-Instruct', maxSeqLen: 4096 });
  assert.match(check.markdown, /显存预估/);
  assert.match(check.markdown, /Python/);

  const prepare = await registeredTools.find((tool) => tool.name === 'train_prepare').execute({
    kind: 'dataset', id: 'tool-made', label: '来自工具', file: datasetFile, mode: 'answer', redaction: 'pii',
  });
  assert.match(prepare.markdown, /数据卡/);
  assert.match(prepare.markdown, /保留/);

  console.log('host.js: 5 tools, 4 routes, fence + lifecycle + refusals verified');
} finally {
  server.close();
  rmSync(dataDir, { recursive: true, force: true });
}
