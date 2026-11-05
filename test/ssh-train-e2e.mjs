/**
 * The full remote-run path, end to end, against a real SSH server on this machine.
 *
 * `ssh-e2e.mjs` proves the transport moves bytes. This proves the thing the user
 * actually cares about: a dataset goes out, the *real worker* trains behind a real
 * SSH channel on the local GPU, and the adapter comes back.
 *
 * It deliberately re-implements the orchestration that `run.start` performs
 * (stage → upload → upload worker → start → collect) rather than importing it,
 * because `run.start` lives inside the cordis plugin closure and needs a live
 * host. The steps are the same ones, in the same order, so a break in either
 * place shows up here.
 *
 * Usage:  node test/ssh-train-e2e.mjs [port]
 * Exit:   0 passed, 1 failed, 77 skipped (no GPU / no torch / no server).
 */

import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildSshArgs } from '../lib/ssh.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = dirname(here);
const port = Number(process.argv[2] ?? 2223);
const python = join(root, 'python', '.venv', 'Scripts', 'python.exe');
const serverScript = join(here, 'ssh-server.py');
const keyPath = join(root, '.test-keys', 'client-key');
const keyOptions = ['-i', keyPath, '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=NUL'];

const results = [];
const check = (name, ok, detail = '') => {
  results.push({ name, ok });
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name}${detail && !ok ? ` — ${detail}` : ''}`);
};

function remote(command, timeoutMs = 600_000) {
  const args = buildSshArgs({ target: '127.0.0.1', command, port });
  args.splice(args.lastIndexOf('127.0.0.1'), 0, ...keyOptions);
  return new Promise((resolve, reject) => {
    const child = spawn('ssh', args, { windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    const timer = setTimeout(() => { child.kill(); resolve({ code: null, stdout, stderr: `${stderr}\n(timeout)` }); }, timeoutMs);
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('close', (code) => { clearTimeout(timer); resolve({ code, stdout, stderr }); });
    child.on('error', reject);
  });
}

function sftpBatch(lines) {
  const args = ['-o', 'BatchMode=yes', '-P', String(port), ...keyOptions, '-b', '-', '127.0.0.1'];
  return new Promise((resolve, reject) => {
    const child = spawn('sftp', args, { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
    let stderr = '';
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.stdout.on('data', () => {});
    child.on('close', (code) => resolve({ code, stderr }));
    child.on('error', reject);
    child.stdin.end(`${lines.join('\n')}\n`);
  });
}

async function startServer() {
  return new Promise((resolve) => {
    // The worker the server launches needs a model cache. Point it at this
    // machine's cache ? a real remote has its own, which is exactly the setup
    // step the probe is meant to catch before a run starts.
    const child = spawn(python, [serverScript, String(port), '--train-test'], {
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, HF_HOME: process.env.DSH_TEST_HF_HOME ?? process.env.HF_HOME ?? '', HF_ENDPOINT: 'https://hf-mirror.com' },
    });
    let buffered = '';
    let settled = false;
    const finish = (value) => { if (!settled) { settled = true; resolve(value); } };
    child.stdout.on('data', (chunk) => {
      buffered += chunk;
      if (buffered.includes('READY')) finish({ ok: true, child, output: buffered });
    });
    child.stderr.on('data', (chunk) => { buffered += chunk; });
    child.on('error', () => finish({ ok: false, output: buffered }));
    child.on('close', () => finish({ ok: false, output: buffered }));
    setTimeout(() => finish({ ok: false, output: 'not ready' }), 30_000);
  });
}

// ---------------------------------------------------------------- preconditions
const torch = await new Promise((resolve) => {
  const child = spawn(python, ['-c', 'import torch,sys;print(1 if torch.cuda.is_available() else 0)'], { windowsHide: true });
  let out = '';
  child.stdout.on('data', (chunk) => { out += chunk; });
  child.on('error', () => resolve(null));
  child.on('close', (code) => resolve(code === 0 ? out.trim() === '1' : null));
});
if (torch !== true) {
  console.log('跳过：本机没有可用的 CUDA（这个测试要真的跑一次训练）。');
  process.exit(77);
}

const server = await startServer();
if (!server.ok) {
  console.log(`跳过：SSH 测试服务器起不来（${server.output.split('\n')[0]}）`);
  process.exit(77);
}

try {
  const work = mkdtempSync(join(tmpdir(), 'dsh-train-'));
  // The remote root the server chroots into is printed on startup.
  const serverRoot = /ROOT (.+)/.exec(server.output)?.[1]?.trim() ?? '';

  // ---- 1. stage a tiny dataset locally (same layout writeFilteredDataset makes)
  const localData = join(work, 'data');
  mkdirSync(localData, { recursive: true });
  // Three short samples: enough for train+val, small enough to finish in seconds.
  const samples = Array.from({ length: 6 }, (_, index) => JSON.stringify({
    messages: [
      { role: 'user', content: `第 ${index} 个问题：这个文件在哪` },
      { role: 'assistant', content: `调用 read 工具读取第 ${index} 个文件` },
    ],
  }));
  writeFileSync(join(localData, 'train.jsonl'), `${samples.slice(0, 4).join('\n')}\n`, 'utf8');
  writeFileSync(join(localData, 'val.jsonl'), `${samples.slice(4).join('\n')}\n`, 'utf8');
  check('准备本地数据集', readdirSync(localData).length === 2);

  // ---- 2. mkdirp + upload (what prepareRemoteRun does)
  const remoteRun = '/work/run-e2e';
  const made = await sftpBatch([`-mkdir "/work"`, `-mkdir "${remoteRun}"`, `-mkdir "${remoteRun}/data"`]);
  check('远程建目录', made.code === 0, made.stderr.split('\n').filter(Boolean).pop());

  const upload = await sftpBatch([
    `put "${join(localData, 'train.jsonl')}" "${remoteRun}/data/train.jsonl"`,
    `put "${join(localData, 'val.jsonl')}" "${remoteRun}/data/val.jsonl"`,
  ]);
  check('上传数据集', upload.code === 0, upload.stderr.split('\n').filter(Boolean).pop());

  // ---- 3. the worker config, written remotely (the backend does this too)
  //
  // `data` and `out` are **directories** the worker joins "train.jsonl" onto — it
  // is the worker's contract, and passing a file path there produces
  // `.../run-e2e\\train.jsonl`. The chroot root is a Windows path here, so it is
  // normalised once, up front, instead of being re-replaced at each use site.
  const toPosix = (value) => String(value).replace(/\\/g, '/');
  const remoteData = toPosix(`${serverRoot}${remoteRun}/data`);
  const remoteOut = toPosix(`${serverRoot}${remoteRun}/out`);
  const remoteConfig = toPosix(`${serverRoot}${remoteRun}/worker-config.json`);
  const config = {
    verb: 'train',
    model: 'Qwen/Qwen2.5-0.5B-Instruct',
    mode: 'qlora',
    data: remoteData,
    out: remoteOut,
    maxSeqLen: 256,
    loraR: 8,
    epochs: 1,
    batchSize: 1,
    gradAccum: 1,
    maxSteps: 3,
    learningRate: 2e-4,
    loggingSteps: 1,
  };
  const configText = JSON.stringify(config, null, 2);
  writeFileSync(join(work, 'worker-config.json'), configText, 'utf8');
  const cfgUpload = await sftpBatch([`put "${join(work, 'worker-config.json')}" "${remoteRun}/worker-config.json"`]);
  check('上传 worker 配置', cfgUpload.code === 0, cfgUpload.stderr.split('\n').filter(Boolean).pop());

  // ---- 4. run the real worker over SSH
  console.log('  … 正在通过 SSH 跑真实训练（约 1-3 分钟）');
  const trained = await remote(`python3 -m fnt_worker train --config "${remoteConfig}"`);
  const events = trained.stdout.split('\n').map((line) => {
    try { return JSON.parse(line); } catch { return null; }
  }).filter((value) => value !== null);
  const kinds = new Set(events.map((event) => event.kind));
  // A worker that starts and dies is the interesting case: print its own words
  // rather than a summary, because catching config drift is the whole point.
  const artifactEvent = events.find((event) => event.kind === 'artifact');
  if (artifactEvent) console.log(`      worker 报告的产物路径：${artifactEvent.path}`);
  const errorEvent = events.find((event) => event.kind === 'error');
  if (errorEvent) console.log(`      远端 worker 报错：${errorEvent.code}: ${String(errorEvent.message).slice(0, 280)}`);
  check('worker 通过 SSH 启动并产出事件', events.length > 0, `exit ${trained.code}`);
  check('收到 metric 事件（有训练在跑）', kinds.has('metric'), [...kinds].join(','));
  check('收到 artifact 事件（适配器已保存）', kinds.has('artifact'), [...kinds].join(','));
  check('worker 正常退出', trained.code === 0, (trained.stderr ?? '').split('\n').filter(Boolean).pop());

  // ---- 5. pull the adapter back (what collectRemoteArtifacts does)
  //
  // The adapter directory has a fixed file set, so the files are fetched by name
  // rather than listed first.
  //
  // Path flavour: this test runs the "remote" on the same Windows box, so the
  // worker reports an absolute host path (`C:/.../out\adapter`) while sftp speaks
  // chroot-relative paths (`/work/run-e2e/out/adapter`). On a real Linux remote
  // both are the same string. The translation below is the price of testing on
  // one machine and is confined to this file.
  const remoteAdapter = toPosix(artifactEvent?.path ?? `${serverRoot}${remoteRun}/out/adapter`)
    .replace(toPosix(serverRoot), '')
    .replace(/\\/g, '/');
  const localOut = join(work, 'out', 'adapter');
  mkdirSync(localOut, { recursive: true });
  const wanted = ['adapter_model.safetensors', 'adapter_config.json'];
  const pull = await sftpBatch(wanted.map((name) =>
    `get "${remoteAdapter}/${name}" "${join(localOut, name).replace(/\\/g, '/')}"`));
  check('拉回 adapter 文件', pull.code === 0 && existsSync(join(localOut, 'adapter_model.safetensors')),
    pull.stderr.split('\n').filter(Boolean).pop());

  if (existsSync(join(localOut, 'adapter_config.json'))) {
    const pulled = JSON.parse(readFileSync(join(localOut, 'adapter_config.json'), 'utf8'));
    check('拉回的 adapter 配置可解析且 rank 正确', pulled.r === 8, `r=${pulled.r}`);
  }
  if (existsSync(join(localOut, 'adapter_model.safetensors'))) {
    const size = readFileSync(join(localOut, 'adapter_model.safetensors')).length;
    check('adapter 权重非空', size > 1000, `${size} bytes`);
  }
} finally {
  server.child.kill();
}

const failed = results.filter((result) => !result.ok).length;
console.log(`\n${results.length - failed}/${results.length} passed, ${failed} failed`);
process.exitCode = failed > 0 ? 1 : 0;
