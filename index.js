/**
 * Fine-tuning bench — host entry.
 *
 * Four HTTP routes under `/fine-tune/`, an SSE stream for a running job, and five
 * model-facing tools. Nothing heavy happens at mount: the job store is reconciled
 * against disk (cheap, and it is what makes "interrupted" honest after a restart),
 * the worker's Python is probed lazily, and a broken plugin degrades to "the panel
 * is missing" — never to "DSH will not start".
 *
 * Every route is gated the way the host's own routes are: loopback only, the Host
 * header must name the loopback host (defeats DNS rebinding), and any mutating
 * request must prove same-origin. A plugin route sits outside the host's auth
 * gate, so this fence is not optional: `POST .../action {"op":"run.start"}` starts
 * a GPU process and `publish.prepare` is the door to Ollama.
 */
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync } from 'node:fs';
import { dirname, join, resolve, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { homedir } from 'node:os';
import { createRequire } from 'node:module';
import { createJobStore } from './lib/jobs.mjs';
import { createLocalBackend, createSshBackend } from './lib/backend.mjs';
import { createSshExec, validateTarget, shellQuote } from './lib/ssh.mjs';
import { createSftpTransfer, downloadArtifacts, uploadDataset } from './lib/transfer.mjs';
import { install } from './lib/install.mjs';
import { readReady, venvPaths } from './lib/bootstrap.mjs';
import { buildDataset, listDatasets, evaluationPair, previewPage, loadExclusions, saveExclusions, deleteDataset, writeFilteredDataset, writeMergedDataset } from './lib/dataset.mjs';
import { listSources, saveSource, loadSource, validateSourceInput } from './lib/sources.mjs';
import { estimateMemory, parseNvidiaSmi, recommendation } from './lib/hardware.mjs';
import { MODES, capabilityOf, planSummary, validateTrainingPlan } from './lib/training.mjs';
import { verdictOf } from './lib/evaluate.mjs';
import { createTools } from './lib/tools.mjs';

export const name = 'fine-tune';
export const inject = ['tools'];

const API = '/fine-tune/api';
export const ROUTES = {
  state: `${API}/state`,
  action: `${API}/action`,
  run: `${API}/run`,
  stream: `${API}/stream`,
};

/**
 * The settings schema, declared so a `cordis.patch.yml` `config:` block reaches
 * this plugin at all.
 *
 * Without a declared `Config`, DSH has nothing to validate against and the raw
 * config never arrives — `allowSpawn` then reads `undefined`, the switch is never
 * on, and the panel says "training cannot start" no matter what the profile
 * contains. That is exactly what happened before this was added: the profile had
 * `allowSpawn: true` and the plugin still refused.
 *
 * `@deepseek-ai/schemastery` is resolved **lazily and optionally** on purpose. It
 * ships with DSH, so it is present inside a profile, but it is absent in a plain
 * checkout (this repository has no node_modules of its own). An eager import
 * would make the module throw before any code runs and turn a missing dev
 * dependency into a plugin that silently fails to load; here it degrades to a
 * schema-less mount that still answers state requests.
 */
function loadSchema() {
  try {
    return createRequire(import.meta.url)('@deepseek-ai/schemastery');
  } catch {
    return null;
  }
}

const z = loadSchema();

/** The declared settings shape, or null when schemastery is unavailable. */
export const Config = z === null
  ? null
  : z.object({
    // The two switches. Both default off: starting a process costs GPU hours and
    // publishing a model is one-way, so neither should happen by accident.
    allowSpawn: z.boolean().default(false).description('允许启动训练进程（本地或远程）。默认关闭。'),
    allowPublish: z.boolean().default(false).description('允许把训练结果发布到 Ollama。默认关闭。'),

    // Local worker interpreter. Empty means "resolve from DSH_FINE_TUNE_PYTHON or
    // the running host", which is right when the plugin's own venv is unknown.
    pythonPath: z.string().default('').description('装了 torch 的 Python 解释器路径（本地训练用）。'),
    workerRoot: z.string().default('').description('fnt_worker 包所在目录；留空则用插件自带的 python/。'),

    // Where runs, datasets and outputs live. Empty means $DSH_HOME/fine-tune.
    dataDir: z.string().default('').description('插件数据目录；留空则用 $DSH_HOME/fine-tune。'),

    // Remote execution.
    sshTarget: z.string().default('').description('SSH 目标：[user@]host[:port] 或 ~/.ssh/config 别名。'),
    sshRoot: z.string().default('/tmp/dsh-fine-tune').description('远端工作目录。'),

    // Optional overrides.
    nvidiaSmi: z.string().default('').description('nvidia-smi 的可执行文件路径；留空则走 PATH。'),
    env: z.dict(z.string()).default({}).description('传给 worker 进程的额外环境变量（如 HF_HOME、HF_ENDPOINT）。'),
  });

/** Where the user can override settings without editing DSH's patch layer. */
export function configFileFor(dataDir) {
  return join(dataDir, 'config.json');
}

/** Read the override file, tolerating its absence and its mistakes. */
function readConfigFile(dataDir) {
  const path = configFileFor(dataDir);
  try {
    if (!existsSync(path)) return {};
    const parsed = JSON.parse(readFileSync(path, 'utf8'));
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
      console.warn(`fine-tune: ${path} 不是 JSON 对象，已忽略`);
      return {};
    }
    return parsed;
  } catch (error) {
    console.warn(`fine-tune: 无法读取 ${path}（${error?.message ?? error}），已忽略`);
    return {};
  }
}

/**
 * Resolve settings from three sources, later winning:
 *
 *   1. the schema defaults
 *   2. whatever DSH passes as the plugin config
 *   3. `$DSH_HOME/fine-tune/config.json`, when present
 *
 * The file is **last** on purpose. DSH hands the plugin a config object with the
 * schema defaults already filled in, so an earlier ordering let `allowSpawn: false`
 * (a default) overwrite `allowSpawn: true` (the user's explicit intent) and the
 * switch could never be turned on. A value the user writes must beat a value the
 * schema invented.
 *
 * The file exists because a `cordis.patch.yml` `config:` block did **not** reach
 * this plugin in practice: the profile said `allowSpawn: true` and the live host
 * still reported `false`. Rather than reverse-engineer DSH's patch-merging rules,
 * the plugin offers a settings file a user can edit without touching DSH internals
 * — and every refusal message names that exact file, so there is always one clear
 * thing to change.
 *
 * A malformed file is reported and ignored: a typo must never stop the plugin
 * from mounting, or the panel would vanish with no way to fix it from the UI.
 */
function resolveConfig(raw, dataDir) {
  let declared = {};
  if (Config !== null) {
    try {
      declared = typeof Config === 'function' ? Config() : {};
    } catch {
      declared = {};
    }
  }
  // `dataDir` and `workerRoot` may legitimately come from the caller and are not
  // something the file should second-guess; everything else follows the rule
  // "what the user wrote wins over what a default supplied".
  const file = dataDir === undefined ? {} : readConfigFile(dataDir);
  const { dataDir: _ignoredDir, ...fromFile } = file;
  const merged = { ...declared, ...(raw ?? {}), ...fromFile };
  if (dataDir !== undefined) merged.dataDir = dataDir;
  // The installer writes its venv under the data directory, so a user who ran it
  // gets a working pythonPath without editing any config file. An explicit
  // pythonPath still wins: someone who pointed at their own interpreter did so on
  // purpose.
  if (typeof merged.pythonPath !== 'string' || merged.pythonPath.trim() === '') {
    const bootstrapped = dataDir === undefined ? null : venvPaths(dataDir).python;
    if (bootstrapped !== null && existsSync(bootstrapped)) merged.pythonPath = bootstrapped;
  }
  return merged;
}

export function apply(ctx, rawConfig = {}) {
  try {
    // The data directory is resolved first: it decides where the override file is.
    const provisional = { ...(rawConfig ?? {}) };
    const dataDir = typeof provisional.dataDir === 'string' && provisional.dataDir.trim() !== ''
      ? resolve(provisional.dataDir.trim())
      : join(dshHomeOf(), 'fine-tune');
    mount(ctx, resolveConfig(provisional, dataDir));
  } catch (error) {
    console.warn('fine-tune: mount failed; the plugin stays dormant', error);
  }
}

export function mount(ctx, settings) {
  const root = resolveRoot(settings);
  mkdirSync(root, { recursive: true });
  const store = createJobStore({ root, log: warn });
  const restored = store.restore();
  if (restored.interrupted > 0) warn(`fine-tune: ${restored.interrupted} run(s) were interrupted before this mount; marked interrupted.`);

  const paths = {
    root,
    datasets: join(root, 'datasets'),
    runs: join(root, 'runs'),
    outputs: join(root, 'outputs'),
    downloads: join(root, 'downloads'),
  };
  // Every path the plugin writes to, created once at mount. Only `datasets` used to
  // be made here; a training run then wrote to `outputs/<id>`, which did not exist,
  // and the worker failed with PermissionError EACCES on a directory nobody had
  // created. Creating them all up front makes that class of failure impossible.
  for (const dir of [paths.datasets, paths.runs, paths.outputs, paths.downloads]) {
    mkdirSync(dir, { recursive: true });
  }

  let gpuCache = { time: 0, value: null };
  const invalidateGpu = () => { gpuCache = { time: 0, value: null }; };
  const gpu = () => probeGpu(settings, gpuCache, (next) => { gpuCache = next; });

  // Which optional libraries the worker's interpreter actually has, and whether
  // the card does bf16. Asking the worker is the only honest source: the host
  // cannot import torch, and `nvidia-smi` does not report bf16 support. Cached
  // for a couple of minutes — installing a wheel should not need a page reload,
  // but neither should every panel refresh spawn Python.
  let capabilityCache = { time: 0, value: null };
  const capabilityProbe = async () => {
    if (Date.now() - capabilityCache.time < 120_000 && capabilityCache.value !== null) return capabilityCache.value;
    const fallback = { bf16Supported: false, deps: { bitsandbytes: false, flashAttn: false, llamaCpp: false } };
    const pythonPath = pythonSettings().pythonPath;
    if (!pythonPath) return fallback;
    const code = [
      'import json,sys',
      'out={"bf16Supported":False,"deps":{}}',
      'try:',
      '    import torch',
      '    out["bf16Supported"]=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())',
      'except Exception: pass',
      'for name,mod in (("bitsandbytes","bitsandbytes"),("flashAttn","flash_attn"),("llamaCpp","llama_cpp")):',
      '    try:',
      '        __import__(mod); out["deps"][name]=True',
      '    except Exception: out["deps"][name]=False',
      'print(json.dumps(out))',
    ].join('\n');
    try {
      const stdout = await new Promise((resolve, reject) => {
        const child = spawn(pythonPath, ['-c', code], { windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
        let out = '';
        let err = '';
        child.stdout.on('data', (chunk) => { out += chunk; });
        child.stderr.on('data', (chunk) => { err += chunk; });
        const timer = setTimeout(() => { child.kill(); reject(new Error('probe timeout')); }, 20_000);
        child.on('error', (error) => { clearTimeout(timer); reject(error); });
        child.on('close', (code2) => {
          clearTimeout(timer);
          if (code2 === 0) resolve(out);
          else reject(new Error(err || `exit ${code2}`));
        });
      });
      const parsed = JSON.parse(stdout.trim().split('\n').pop() || '{}');
      const value = { bf16Supported: parsed.bf16Supported === true, deps: parsed.deps ?? {} };
      capabilityCache = { time: Date.now(), value };
      return value;
    } catch {
      return fallback;
    }
  };

  // The worker's Python: the configured interpreter, else the one running the
  // host when that is a real interpreter, else nothing — and then every path that
  // needs it says which setting to fill. (An Electron host's process.execPath is
  // its own binary, so this fallback only applies to a `node`-driven host.)
  const pythonSettings = () => ({
    ...settings,
    pythonPath: settings.pythonPath || process.env.DSH_FINE_TUNE_PYTHON || (/python/i.test(basename(process.execPath)) ? process.execPath : ''),
  });
  const workerRoot = () => (typeof settings.workerRoot === 'string' && settings.workerRoot.trim() !== ''
    ? settings.workerRoot.trim()
    : join(dirname(fileURLToPath(import.meta.url)), 'python'));

  const sshEnv = { ...process.env, ...(settings.env ?? {}) };
  const sshExec = createSshExec({ env: sshEnv });
  const transfer = createSftpTransfer({ env: sshEnv });

  const backends = {
    local: createLocalBackend({ settings: { ...pythonSettings(), workerRoot: workerRoot() }, root, log: warn }),
    ssh: createSshBackend({
      settings,
      root,
      log: warn,
      // Wraps the system `ssh`. Nothing is configured here that the user's own
      // ~/.ssh/config does not already provide — the target is whatever alias or
      // host they put in the settings, and host-key trust stays ssh's business.
      sshExec,
    }),
  };

  // The tools import their own filesystem helpers and take `root` explicitly;
  // this object is what they destructure. Keeping it explicit (rather than
  // passing closures they never call) is what makes a missing `root` a startup
  // error here instead of a `join(undefined, …)` inside a tool call.
  const helpers = {
    root,
    paths,
    settings,
    store,
    backends,
    dshHome: dshHomeOf,
    estimateMemory,
    gpu,
  };

  for (const tool of createTools(helpers)) {
    ctx.effect(() => ctx.tools.register(tool), `fine-tune: ${tool.name} tool`);
  }

  const runAction = (body) => action({ body, settings, paths, store, backends, gpu, invalidateGpu });

  try {
    if (typeof ctx.inject === 'function') {
      ctx.inject(['webServer'], (inner) => {
        const webServer = inner?.webServer;
        if (webServer === undefined || webServer === null || typeof webServer.register !== 'function') return;
        const publish = (path, handler, label) => {
          const disposer = webServer.register({ kind: 'exact', path, handler });
          if (typeof inner.effect === 'function') inner.effect(() => disposer, label);
        };
        const gate = (request, response, { mutate }) => {
          if (!loopback(request, settings)) {
            send(response, 403, { error: 'fine-tune: this panel is served to the local host only.' });
            return false;
          }
          if (mutate && !sameOrigin(request)) {
            send(response, 403, { error: 'fine-tune: a state-changing request must prove same-origin.' });
            return false;
          }
          return true;
        };

        publish(ROUTES.state, async (request, response) => {
          if (!gate(request, response, { mutate: false })) return;
          if (!allowGet(request, response)) return;
          try {
            const url = new URL(request.url ?? ROUTES.state, 'http://127.0.0.1');
            const gpuNow = url.searchParams.get('gpu') === '1' ? await probeGpu(settings, { time: 0, value: null }, () => {}) : gpuCache.value;
            send(response, 200, {
              schema: 'dsh-fine-tune/v1',
              settings: publicSettings(settings),
              runs: store.list({ limit: 30 }),
              sources: listSources(root),
              datasets: listDatasets(root),
              modes: Object.values(MODES),
                        gpu: gpuNow,
            });
          } catch (error) {
            send(response, 500, { error: error?.message ?? String(error) });
          }
        }, 'fine-tune: state route');

        publish(ROUTES.action, async (request, response) => {
          if (!gate(request, response, { mutate: true })) return;
          if (request?.method !== 'POST') {
            response.writeHead(405, { allow: 'POST' });
            response.end();
            return;
          }
          try {
            const body = await readJsonBody(request);
            const result = await runAction(body);
            send(response, result.status ?? 200, result.body ?? { ok: true });
          } catch (error) {
            send(response, 400, { ok: false, error: error?.message ?? String(error) });
          }
        }, 'fine-tune: action route');

        publish(ROUTES.run, async (request, response) => {
          if (!gate(request, response, { mutate: false })) return;
          if (!allowGet(request, response)) return;
          try {
            const url = new URL(request.url ?? ROUTES.run, 'http://127.0.0.1');
            const id = url.searchParams.get('id') ?? '';
            const since = Number(url.searchParams.get('since') ?? 0) || 0;
            const snapshot = store.snapshot(id, since);
            if (snapshot === null) {
              send(response, 404, { error: `unknown run: ${id}` });
              return;
            }
            send(response, 200, snapshot);
          } catch (error) {
            send(response, 500, { error: error?.message ?? String(error) });
          }
        }, 'fine-tune: run route');

        publish(ROUTES.stream, async (request, response) => {
          if (!gate(request, response, { mutate: false })) return;
          const url = new URL(request.url ?? ROUTES.stream, 'http://127.0.0.1');
          const id = url.searchParams.get('id') ?? '';
          const job = store.job(id);
          if (job === null) {
            send(response, 404, { error: `unknown run: ${id}` });
            return;
          }
          response.writeHead(200, {
            'content-type': 'text/event-stream; charset=utf-8',
            'cache-control': 'no-store',
            connection: 'keep-alive',
            'x-accel-buffering': 'no',
          });
          let closed = false;
          const write = (payload) => {
            if (closed) return;
            try {
              response.write(`data: ${JSON.stringify(payload)}\n\n`);
            } catch {
              closed = true;
            }
          };
          // Replay from the start: a page that reconnects mid-run must not need to
          // remember what it missed.
          write({ kind: 'snapshot', record: job.record, events: job.events });
          const unsubscribe = store.subscribe(id, (event) => write({ kind: 'event', event }));
          const heartbeat = setInterval(() => {
            if (!closed) {
              try {
                response.write(': keep-alive\n\n');
              } catch {
                closed = true;
              }
            }
          }, 15_000);
          heartbeat.unref?.();
          const stop = () => {
            if (closed) return;
            closed = true;
            clearInterval(heartbeat);
            unsubscribe();
          };
          request.on('close', stop);
          request.on('error', stop);
        }, 'fine-tune: stream route');
      });
    }
  } catch (error) {
    warn(`fine-tune: could not publish the panel routes: ${error?.message ?? error}`);
  }
  return { store, paths };
}

/**
 * Stage a dataset on the remote host before a run starts.
 *
 * Two things are uploaded: the split files, and the `fnt_worker` package the
 * remote Python needs on its path. Shipping the worker with every run is
 * deliberate — otherwise a machine whose `fnt_worker` is an older copy silently
 * trains with different label masking, and the result is not comparable.
 *
 * The upload is all-or-nothing: a dataset with `train.jsonl` but no `val.jsonl`
 * would make the worker skip validation and report a number the user cannot check.
 */
async function prepareRemoteRun({ store, id, target, remoteRun, localData, sshExec, transfer }) {
  if (!validateTarget(target).ok) return { ok: false, error: validateTarget(target).error };
  try {
    store.emit(store.job(id), { kind: 'phase', phase: 'upload', message: `上传数据集到 ${target}:${remoteRun}/data` });
    const uploaded = await uploadDataset(transfer, target, localData, `${remoteRun}/data`);
    store.emit(store.job(id), {
      kind: 'log',
      level: 'info',
      message: `已上传 ${uploaded.files.length} 个文件（${(uploaded.bytes / 1024).toFixed(1)} KB）`,
    });

    // The worker package travels alongside the data.
    const workerDir = join(dirname(fileURLToPath(import.meta.url)), 'python');
    const remoteWorker = `${remoteRun}/worker`;
    await transfer.mkdirp(target, remoteWorker);
    for (const file of ['__init__.py', '__main__.py', 'calibrate.py']) {
      await transfer.put(target, join(workerDir, 'fnt_worker', file), `${remoteWorker}/fnt_worker/${file}`);
    }
    store.emit(store.job(id), { kind: 'log', level: 'info', message: '已上传 fnt_worker 包（与本地版本一致）' });
    return { ok: true, uploaded };
  } catch (error) {
    return { ok: false, error: `上传失败：${error?.message ?? error}` };
  }
}

/**
 * Pull the adapter back once the remote run exits.
 *
 * Runs detached from the request: the run is already `running` when this is
 * called, and a failed pull must not turn a successful training run into a
 * failure. It records the outcome as a run event, so the panel shows "adapter
 * retrieved" or the transfer error instead of nothing.
 */
function collectRemoteArtifacts({ store, id, target, remoteRun, localOut, sshExec, transfer, warn }) {
  const attempt = async () => {
    try {
      const result = await downloadArtifacts(transfer, target, remoteRun, localOut, { sshExec });
      if (result.files.length > 0) {
        store.emit(store.job(id), { kind: 'artifact', label: 'remote-adapter', path: join(localOut, 'adapter') });
        store.emit(store.job(id), { kind: 'log', level: 'info', message: `已取回 ${result.files.length} 个文件到 ${join(localOut, 'adapter')}` });
      } else {
        store.emit(store.job(id), { kind: 'log', level: 'warn', message: `远端没有可取的产物：${result.skipped.join('；')}` });
      }
    } catch (error) {
      store.emit(store.job(id), { kind: 'log', level: 'warn', message: `取回产物失败（适配器仍在远端 ${remoteRun}）：${error?.message ?? error}` });
      warn(`fine-tune: artifact pull failed for ${id}: ${error?.message ?? error}`);
    }
  };
  // Poll for terminal state rather than hooking the store's internals.
  const timer = setInterval(() => {
    const job = store.job(id);
    if (job === null) { clearInterval(timer); return; }
    if (job.state === 'succeeded' || job.state === 'failed' || job.state === 'cancelled' || job.state === 'interrupted') {
      clearInterval(timer);
      void attempt();
    }
  }, 2000);
  timer.unref?.();
}

/**
 * One POST endpoint, one discriminator, every mutation the panel can make.
 *
 * Keeping them behind a single route with an `op` field (instead of eight paths)
 * means the gate, the body cap and the error shape are written once.
 */
/**
 * The install currently running, or null.
 *
 * Module scope on purpose: a venv is a shared resource, and two concurrent
 * `env.install` calls would fight over it. The route reads this before starting,
 * so the second request gets a clear refusal instead of corrupting the first.
 */
let installing = null;

/** Read one dataset's manifest, or null. */
function loadDatasetManifest(datasetsRoot, id) {
  try {
    return JSON.parse(readFileSync(join(datasetsRoot, id, 'manifest.json'), 'utf8'));
  } catch {
    return null;
  }
}

export async function action({ body, settings, paths, store, backends, gpu, invalidateGpu }) {
  const op = String(body?.op ?? '');
  const root = paths.root;
  switch (op) {
    case 'gpu': {
      const gpuInfo = await gpu();
      const estimate = body?.modelId === undefined
        ? null
        : estimateMemory({
          modelId: String(body.modelId),
          mode: MODES[String(body?.mode ?? 'qlora')] === undefined ? 'qlora' : String(body?.mode ?? 'qlora'),
          maxSeqLen: clampInt(body?.maxSeqLen, 2048, 256, 8192),
          batchSize: clampInt(body?.batchSize, 1, 1, 16),
          loraR: clampInt(body?.loraR, 16, 4, 128),
          gpuGiB: gpuInfo?.gpus?.[0]?.totalGiB,
        });
      return {
        body: {
          ok: true,
          gpu: gpuInfo,
          estimate,
          recommendation: estimate === null ? null : recommendation(estimate),
          modes: Object.values(MODES),
        },
      };
    }
    case 'capability': {
      // What this machine can honour, for the panel to gate advanced knobs on.
      // The GPU report is cached and may be null (no probe succeeded); that is
      // "unknown", and the panel then disables the hardware-dependent knobs
      // rather than pretending they work.
      const gpuInfo = await gpu();
      const first = gpuInfo?.gpus?.[0] ?? null;
      const probe = await capabilityProbe(); // { bf16Supported, deps }
      return {
        body: {
          ok: true,
          capability: capabilityOf({
            gpu: first === null ? null : {
              name: first.name,
              count: gpuInfo?.gpus?.length ?? 1,
              capability: first.computeCapability ?? null,
            },
            bf16Supported: probe.bf16Supported,
            platform: process.platform,
            deps: probe.deps,
          }),
        },
      };
    }
    case 'probe': {
      const backend = String(body?.backend ?? 'local');
      if (backends[backend] === undefined) return { status: 400, body: { ok: false, error: `未知后端: ${backend}` } };
      const result = await backends[backend].probe(body?.target ?? settings.sshTarget);
      return { body: { ok: result?.ok === true, result } };
    }
    case 'env.install': {
      // The one action that downloads gigabytes, so it is explicit and its progress
      // is streamed: `install()` calls onEvent for each step. Concurrent installs
      // would fight over the same venv, so a second request while one runs is
      // refused rather than queued.
      if (installing !== null) {
        return { status: 409, body: { ok: false, error: `????????${installing.phase}` } };
      }
      if (body?.confirm !== true) {
        return { status: 409, body: { ok: false, error: '?????? 3-5 GB ???????? confirm: true' } };
      }
      installing = { phase: 'starting' };
      const events = [];
      try {
        const result = await install({
          root,
          force: body?.force === true,
          onEvent: (event) => {
            installing = event;
            events.push(event);
            warn(`[install] ${event.phase}: ${event.message}`);
          },
        });
        installing = null;
        return { body: { ok: result.ok === true, result, events: events.slice(-40) } };
      } catch (error) {
        installing = null;
        return { status: 500, body: { ok: false, error: `?????${error?.message ?? error}`, events: events.slice(-40) } };
      }
    }
    case 'env.status': {
      const ready = readReady(root);
      return {
        body: {
          ok: true,
          installing,
          ready,
          venvPython: venvPaths(root).python,
          venvExists: existsSync(venvPaths(root).python),
          effectivePythonPath: settings.pythonPath,
        },
      };
    }
    case 'source.create': {
      const validated = validateSourceInput(body?.source);
      if (validated.error !== undefined) return { status: 400, body: { ok: false, error: validated.error } };
      if (loadSource(root, validated.value.id) !== null) return { status: 409, body: { ok: false, error: `数据源 ${validated.value.id} 已存在` } };
      return { body: { ok: true, source: saveSource(root, validated.value) } };
    }
    case 'dataset.sample': {
      // Builds a dataset from the sample shipped with the plugin. A fresh install
      // has no data at all, and "find some JSONL first" is the step where a new
      // user stops ? this makes the pipeline runnable before they bring anything.
      //
      // Rebuilding is refused, not an error: the button is idempotent so a double
      // click does not create `sample-2`, `sample-3`.
      const sampleFile = join(dirname(fileURLToPath(import.meta.url)), 'sample', 'sample.jsonl');
      if (!existsSync(sampleFile)) {
        return { status: 404, body: { ok: false, error: `???????????${sampleFile}` } };
      }
      const datasetId = 'sample';
      if (existsSync(join(paths.datasets, datasetId, 'manifest.json'))) {
        return { body: { ok: true, dataset: loadDatasetManifest(paths.datasets, datasetId), existing: true } };
      }
      const built = buildDataset({
        root,
        source: {
          id: 'sample',
          kind: 'dataset',
          label: '??????60 ????????',
          file: sampleFile,
          format: 'auto',
          redaction: 'path',
          mode: 'answer',
          createdAt: Date.now(),
        },
        datasetId,
      });
      return { body: { ok: true, dataset: built.manifest } };
    }
    case 'dataset.build': {
      const source = loadSource(root, String(body?.sourceId ?? ''));
      if (source === null) return { status: 404, body: { ok: false, error: '数据源不存在' } };
      const datasetId = String(body?.datasetId ?? '').trim() || `ds-${new Date().toISOString().replace(/[^0-9]/g, '').slice(0, 14)}`;
      if (!/^[a-z0-9][a-z0-9._-]{1,60}$/.test(datasetId)) return { status: 400, body: { ok: false, error: 'datasetId 只能用 a-z0-9._-' } };
      const built = buildDataset({
        root,
        source,
        datasetId,
        dshHome: dshHomeOf(),
        options: { valRatio: body?.valRatio, evalRatio: body?.evalRatio },
      });
      return { body: { ok: true, dataset: built.manifest } };
    }
    case 'run.start': {
      if (!allowSpawn(settings)) return { status: 403, body: { ok: false, error: '未开启「允许启动训练进程」。在 ' + configFileFor(root) + ' 里写 {"allowSpawn": true} （没有这个文件就新建），然后重启 DSH。' } };
      // Several datasets may be trained on together: mined tool calls plus a
      // hand-written domain file is a normal combination, and forcing the user to
      // concatenate them by hand first would lose each source's own split.
      const rawIds = Array.isArray(body?.datasetIds)
        ? body.datasetIds
        : [body?.datasetId ?? ''];
      const datasetIds = [...new Set(rawIds.map((value) => String(value ?? '')))].filter((value) => value !== '');
      if (datasetIds.length === 0) return { status: 400, body: { ok: false, error: '没有选择数据集' } };
      for (const id of datasetIds) {
        if (!/^[a-z0-9][a-z0-9._-]{1,60}$/.test(id)) return { status: 400, body: { ok: false, error: `数据集 id 不合法: ${id}` } };
        if (!existsSync(join(paths.datasets, id, 'train.jsonl'))) {
          return { status: 404, body: { ok: false, error: `数据集不存在或没有 train.jsonl: ${id}` } };
        }
      }
      const datasetId = datasetIds.join('+');
      const datasetDirs = datasetIds.map((id) => join(paths.datasets, id));
      // One validator for every knob: the panel's form, the tool call and this
      // route all go through it, so a bad rank is a sentence, not an OOM.
      const validated = validateTrainingPlan(body);
      if (validated.ok !== true) return { status: 400, body: { ok: false, error: validated.error } };
      const plan = validated.value;
      const backend = body?.backend === 'ssh' ? 'ssh' : 'local';
      const gpuInfo = backend === 'local' ? await gpu() : null;
      const estimate = estimateMemory({ ...plan, modelId: plan.modelId, gpuGiB: gpuInfo?.gpus?.[0]?.totalGiB });
      // Refuse before spawning, not after OOM: the user's time is the scarce thing.
      if (backend === 'local' && estimate.fits === false) {
        return { status: 409, body: { ok: false, error: `显存放不下（${planSummary(plan)}）：任务没有启动。`, estimate, recommendation: recommendation(estimate) } };
      }
      const id = store.create({ kind: 'train', datasetId, backend, target: body?.target ?? null, input: { ...plan, summary: planSummary(plan) } });
      // Materialise the *effective* dataset (minus anything the user removed in
      // the preview) where the worker will read it. Doing it here — not at build
      // time — is what makes "remove a bad sample" reach the very next run.
      let effective;
      try {
        // One dataset keeps the existing filtered path; several are concatenated
        // per split, each keeping its own exclusions applied.
        if (datasetDirs.length === 1) {
          effective = writeFilteredDataset(datasetDirs[0], join(paths.runs, id, 'data'));
        } else {
          // Apply each source's exclusions before concatenating, so a sample the
          // user removed from one dataset stays out of the merged run.
          const staged = [];
          datasetDirs.forEach((dir, index) => {
            const stage = join(paths.runs, id, `stage-${index}`);
            writeFilteredDataset(dir, stage);
            staged.push(stage);
          });
          effective = { ...writeMergedDataset(staged, join(paths.runs, id, 'data')), sources: datasetIds };
        }
      } catch (error) {
        store.move(id, 'failed', { error: { code: 'data', message: `无法准备训练数据：${error?.message ?? error}` } });
        return { status: 500, body: { ok: false, error: `无法准备训练数据：${error?.message ?? error}` } };
      }
      if (effective.counts.train === 0) {
        store.move(id, 'failed', { error: { code: 'data', message: '移除之后训练集是空的' } });
        return { status: 409, body: { ok: false, error: '移除之后训练集是空的：恢复几条样本再训。' } };
      }
      // The host creates every directory this run will write to, before the worker
      // starts. On this machine Python cannot create a directory anywhere under the
      // plugin root ? `os.mkdir` returns EACCES while the same call from Node or
      // .NET succeeds, which points at a filter driver or scanner that treats the
      // Python process differently. Rather than fight that, the parent does it: a
      // worker that only ever writes *files* into directories it was handed has one
      // fewer thing to get wrong.
      //
      // Also belt and braces: mount() creates these, but a run must not depend on
      // the mount having happened after the directories were removed by hand.
      const localOut = join(paths.outputs, id);
      for (const dir of [paths.outputs, localOut]) mkdirSync(dir, { recursive: true });
      const remoteRoot = String(settings.sshRoot ?? '/tmp/dsh-fine-tune').replace(/\/+$/, '');
      const remoteRun = `${remoteRoot}/runs/${id}`;
      const spec = {
        verb: 'train',
        model: plan.modelId,
        // Local runs read the staged directory directly; a remote run reads the
        // path the upload below populated, so the two never disagree.
        data: backend === 'ssh' ? `${remoteRun}/data` : effective.dir,
        out: backend === 'ssh' ? remoteRun : localOut,
        dryRun: body?.dryRun === true,
        ...plan,
        ...(backend === 'ssh' ? { target: String(body?.target ?? settings.sshTarget ?? ''), remoteRoot } : {}),
      };

      // ---- remote staging: upload before start, pull after exit.
      //
      // Both halves run in the background so the panel is not blocked by a slow
      // link; the run goes `queued` first and only becomes `running` once the
      // upload is done. A failed upload fails the run with the transfer's own
      // words instead of a mysterious "worker exited 2".
      if (backend === 'ssh') {
        const target = String(body?.target ?? settings.sshTarget ?? '');
        const prepared = await prepareRemoteRun({
          store, id, target, remoteRun, localData: effective.dir, sshExec, transfer,
        });
        if (prepared.ok !== true) {
          store.move(id, 'failed', { error: { code: 'transfer', message: prepared.error } });
          return { status: 502, body: { ok: false, error: prepared.error, run: store.get(id) } };
        }
      }

      let handle;
      try {
        handle = backends[backend].start(id, spec, hooks(store, id));
      } catch (error) {
        store.move(id, 'failed', { error: { code: 'spawn', message: error?.message ?? String(error) } });
        return { status: 500, body: { ok: false, error: error?.message ?? String(error), run: store.get(id) } };
      }
      store.attach(id, handle);
      store.move(id, 'running');
      if (backend === 'ssh') {
        // Fire-and-forget: the run is already running, and a failed pull is
        // recorded on the run rather than losing the trained adapter silently.
        collectRemoteArtifacts({ store, id, target: String(body?.target ?? settings.sshTarget ?? ''), remoteRun, localOut: join(paths.outputs, id), sshExec, transfer, warn });
      }
      return { body: { ok: true, run: store.get(id), estimate, remote: backend === 'ssh' ? { run: remoteRun } : null } };
    }
    case 'run.cancel': {
      const id = String(body?.id ?? '');
      const job = store.job(id);
      if (job === null) return { status: 404, body: { ok: false, error: '任务不存在' } };
      if (job.handle !== null) job.handle.kill();
      store.move(id, 'cancelled');
      return { body: { ok: true, run: store.get(id) } };
    }
    case 'run.delete': {
      // Removes a finished run from the list *and* deletes its artifacts. Refused
      // while the run is active: deleting the directory under a live worker would
      // leave it writing into a path that no longer exists. Cancel first, then
      // delete ? two deliberate steps for one destructive action.
      if (body?.confirm !== true) {
        return { status: 409, body: { ok: false, error: '???????????????? confirm: true' } };
      }
      const id = String(body?.id ?? '');
      if (!/^[a-z]+-\d{14}-[a-z0-9]{6}$/.test(id)) return { status: 400, body: { ok: false, error: '?? id ???' } };
      // Disk is the source of truth, not the in-memory map: a run created before
      // this process started (or one whose record failed to parse) is still a
      // directory a user may want gone. The map only adds the "is it live" check.
      const runFile = join(paths.runs, id, 'run.json');
      if (!existsSync(runFile)) return { status: 404, body: { ok: false, error: `??????${id}` } };
      const inMemory = store.job(id);
      let state = null;
      try {
        state = JSON.parse(readFileSync(runFile, 'utf8')).state ?? null;
      } catch { /* an unreadable record is still deletable */ }
      if (inMemory !== null) state = inMemory.record.state;
      if (state === 'running' || state === 'queued') {
        return { status: 409, body: { ok: false, error: '?????????????' } };
      }
      // The run record lives in runs/<id>; the worker's output in outputs/<id>.
      // Both go, or the next mount restores a run whose data is half gone.
      for (const dir of [join(paths.runs, id), join(paths.outputs, id)]) {
        rmSync(dir, { recursive: true, force: true });
      }
      store.forget(id);
      return { body: { ok: true, id } };
    }
    case 'dataset.preview': {
      const datasetId = String(body?.datasetId ?? '');
      if (!/^[a-z0-9][a-z0-9._-]{1,60}$/.test(datasetId)) return { status: 400, body: { ok: false, error: 'datasetId 不合法' } };
      const dir = join(paths.datasets, datasetId);
      if (!existsSync(join(dir, 'manifest.json'))) return { status: 404, body: { ok: false, error: `没有数据集 ${datasetId}` } };
      try {
        const page = previewPage(dir, {
          split: String(body?.split ?? 'train'),
          limit: body?.limit,
          offset: body?.offset,
          chars: body?.chars,
        });
        return {
          body: {
            ok: true,
            ...page,
            manifest: JSON.parse(readFileSync(join(dir, 'manifest.json'), 'utf8')),
            excluded: loadExclusions(dir),
          },
        };
      } catch (error) {
        return { status: 400, body: { ok: false, error: error?.message ?? String(error) } };
      }
    }
    case 'dataset.delete': {
      // Irreversible, so the caller has to say so explicitly: the panel asks for
      // confirmation and sends `confirm: true`. Without it the route refuses,
      // which stops a stray click (or a retried request) from removing data.
      if (body?.confirm !== true) {
        return { status: 409, body: { ok: false, error: '??????????????? confirm: true' } };
      }
      const removed = deleteDataset(root, body?.datasetId);
      if (removed.ok !== true) return { status: 400, body: { ok: false, error: removed.error } };
      return { body: { ok: true, id: removed.id } };
    }
    case 'dataset.exclude': {
      // Removal / restoration of one sample by hash. Reversible on purpose: the
      // dataset on disk stays what the miner produced, and the effective set is
      // that minus the excluded hashes.
      const datasetId = String(body?.datasetId ?? '');
      if (!/^[a-z0-9][a-z0-9._-]{1,60}$/.test(datasetId)) return { status: 400, body: { ok: false, error: 'datasetId 不合法' } };
      const dir = join(paths.datasets, datasetId);
      if (!existsSync(join(dir, 'manifest.json'))) return { status: 404, body: { ok: false, error: `没有数据集 ${datasetId}` } };
      const hash = String(body?.hash ?? '');
      if (!/^[a-f0-9]{8,64}$/.test(hash)) return { status: 400, body: { ok: false, error: 'hash 不合法' } };
      const current = loadExclusions(dir);
      const excluded = body?.restore === true ? current.filter((item) => item !== hash) : [...current, hash];
      return { body: { ok: true, excluded: saveExclusions(dir, excluded) } };
    }
    case 'evaluation.pair': {
      // The "done" half of the panel's comparison flow: a page of eval cases with
      // their prompts, the expected answers, and a stable A/B shuffle. The panel
      // copies this to a judge and pastes verdicts back via `eval.record`.
      const datasetId = String(body?.datasetId ?? '');
      if (!/^[a-z0-9][a-z0-9._-]{1,60}$/.test(datasetId)) return { status: 400, body: { ok: false, error: 'datasetId 不合法' } };
      const dir = join(paths.datasets, datasetId);
      if (!existsSync(join(dir, 'manifest.json'))) return { status: 404, body: { ok: false, error: `没有数据集 ${datasetId}` } };
      const pair = evaluationPair(dir, { limit: body?.limit, offset: body?.offset });
      let fixture = { knownTools: [], judgeContext: null };
      try {
        fixture = JSON.parse(readFileSync(join(dir, 'fixture.json'), 'utf8'));
      } catch { /* older datasets have no fixture; grading then skips tool checks */ }
      return { body: { ok: true, datasetId, fixture, ...pair } };
    }
    case 'eval.record': {
      // The panel computes the outputs with whatever judge it is configured to
      // use; recording the verdict is all this server does, and it is what gates
      // the publish button.
      const id = String(body?.id ?? '');
      if (store.get(id) === null) return { status: 404, body: { ok: false, error: '任务不存在' } };
      const verdict = verdictOf({ candidate: body?.candidate, base: body?.base, tolerance: Number(body?.tolerance ?? 0.03) });
      store.progress(id, { kind: 'evaluation', verdict, base: body?.base, candidate: body?.candidate });
      return { body: { ok: true, verdict } };
    }
    case 'publish.prepare': {
      if (!allowPublish(settings)) return { status: 403, body: { ok: false, error: '未开启「允许发布模型」。在 ' + configFileFor(root) + ' 里写 {"allowPublish": true}。' } };
      const id = String(body?.id ?? '');
      const job = store.job(id);
      if (job === null || job.record === undefined) return { status: 404, body: { ok: false, error: '任务不存在' } };
      const green = job.events.some((event) => event.kind === 'evaluation' && event.verdict?.status === 'green');
      if (!green) return { status: 409, body: { ok: false, error: '还没有通过的评测：先跑一次对比评测，并且不能有回退。' } };
      const name = String(body?.name ?? `dsh-ft-${id.slice(-6)}`).toLowerCase().replace(/[^a-z0-9._-]/g, '-');
      const gguf = job.record.artifacts.find((path) => String(path).endsWith('.gguf'));
      if (gguf === undefined) return { status: 409, body: { ok: false, error: '这次训练没有产出 .gguf：先导出 GGUF。' } };
      // The panel runs `ollama create`; this endpoint answers "is it allowed and
      // where is the file" so a stale panel can never publish green it never saw.
      return { body: { ok: true, name, gguf, command: `ollama create ${name} -f ${gguf.replace(/[^/\\]+$/, 'Modelfile')}` } };
    }
    case 'gpu.invalidate': {
      invalidateGpu();
      return { body: { ok: true } };
    }
    default:
      return { status: 400, body: { ok: false, error: `未知操作: ${op}` } };
  }
}

function hooks(store, id) {
  return {
    onEvent(event) {
      if (event.kind === 'metric') store.progress(id, { kind: 'metric', step: event.step, values: event.values, loss: event.loss });
      else if (event.kind === 'artifact') store.progress(id, { kind: 'artifact', path: String(event.path ?? ''), label: event.label ?? null });
      else if (event.kind === 'log') store.progress(id, { kind: 'log', level: event.level ?? 'info', message: String(event.message ?? '') });
      else if (event.kind === 'error') store.progress(id, { kind: 'error', message: String(event.message ?? 'worker error') });
      else store.progress(id, { kind: 'worker', payload: event });
    },
    onExit({ code, signal, error }) {
      const record = store.get(id);
      if (record === null || record.state === 'cancelled' || record.state === 'interrupted') return;
      if (error !== undefined) {
        store.move(id, 'failed', { error: { code: 'spawn', message: error } });
        return;
      }
      if (code === 0) {
        store.move(id, 'succeeded');
        return;
      }
      store.move(id, 'failed', {
        exitCode: code,
        error: {
          code: signal === null ? `exit-${code}` : `signal-${signal}`,
          message: signal === null ? `worker 退出码 ${code}（看 runs/${id}/stderr.log）` : `worker 被信号 ${signal} 终止`,
        },
      });
    },
  };
}

/** One `nvidia-smi` per minute: it is a subprocess, not a syscall. */
export function probeGpu(settings, cache, save) {
  if (cache !== undefined && cache.value !== null && Date.now() - cache.time < 60_000) return Promise.resolve(cache.value);
  return new Promise((resolve) => {
    let child;
    try {
      const command = typeof settings?.nvidiaSmi === 'string' && settings.nvidiaSmi.trim() !== '' ? settings.nvidiaSmi : 'nvidia-smi';
      child = spawn(command, ['--query-gpu=name,memory.total', '--format=csv,noheader'], { windowsHide: true });
    } catch (error) {
      resolve({ available: false, reason: `无法运行 nvidia-smi：${error?.message ?? error}`, gpus: [] });
      return;
    }
    let output = '';
    child.stdout.on('data', (chunk) => { output += chunk; });
    child.stderr.on('data', () => {});
    child.on('error', (error) => resolve({ available: false, reason: `没有找到 nvidia-smi：${error.message}`, gpus: [] }));
    child.on('exit', (code) => {
      if (code !== 0) {
        resolve({ available: false, reason: 'nvidia-smi 返回非零：这台机器可能没有 NVIDIA 显卡，或者驱动没装。', gpus: [] });
        return;
      }
      const gpus = parseNvidiaSmi(output);
      const value = { available: gpus.length > 0, reason: gpus.length === 0 ? 'nvidia-smi 没有列出显卡' : null, gpus };
      save?.({ time: Date.now(), value });
      resolve(value);
    });
  });
}

function resolveRoot(settings) {
  const configured = typeof settings.dataDir === 'string' ? settings.dataDir.trim() : '';
  if (configured !== '') return resolve(configured);
  return join(dshHomeOf(), 'fine-tune');
}

function dshHomeOf() {
  const home = process.env.DSH_HOME;
  return home !== undefined && home.trim() !== '' ? home.trim() : join(homedir(), '.dsh');
}

export function publicSettings(settings) {
  return {
    dataDir: typeof settings.dataDir === 'string' ? settings.dataDir : null,
    pythonPath: typeof settings.pythonPath === 'string' ? settings.pythonPath : null,
    workerRoot: typeof settings.workerRoot === 'string' ? settings.workerRoot : null,
    sshTarget: typeof settings.sshTarget === 'string' ? settings.sshTarget : null,
    ollamaBase: typeof settings.ollamaBase === 'string' ? settings.ollamaBase : 'http://127.0.0.1:11434',
    allowSpawn: allowSpawn(settings),
    allowPublish: allowPublish(settings),
    allowRemotePanel: settings?.allowRemotePanel === true,
  };
}

export const allowSpawn = (settings) => settings?.allowSpawn === true;
export const allowPublish = (settings) => settings?.allowPublish === true;

const clampInt = (value, fallback, min, max) => {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, Math.trunc(number)));
};

/**
 * The request fence.
 *
 * 1. the socket peer must be loopback,
 * 2. the Host header must name the loopback host (a DNS-rebinding page reaches
 *    127.0.0.1 with its own Host),
 * 3. a mutating request must carry a same-origin Origin when it carries one at
 *    all — a non-browser client (curl, a script) sends none.
 */
export function loopback(request, settings) {
  if (settings?.allowRemotePanel === true) return true;
  const remote = String(request?.socket?.remoteAddress ?? '');
  const peerOk = remote === '::1' || remote === '::ffff:127.0.0.1' || /^127\.\d+\.\d+\.\d+$/.test(remote);
  if (!peerOk) return false;
  const raw = String(request?.headers?.host ?? '').toLowerCase().replace(/:\d+$/, '').replace(/^\[/, '').replace(/\]$/, '');
  return raw === '127.0.0.1' || raw === 'localhost' || raw === '::1';
}

export function sameOrigin(request) {
  const origin = request?.headers?.origin;
  if (origin === undefined || origin === '') return true;
  const host = String(request?.headers?.host ?? '');
  try {
    return new URL(origin).host === host;
  } catch {
    return false;
  }
}

function allowGet(request, response) {
  if (request?.method === 'GET' || request?.method === 'HEAD') return true;
  response.writeHead(405, { allow: 'GET, HEAD' });
  response.end();
  return false;
}

export function readJsonBody(request, limit = 64 * 1024) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    request.on('data', (chunk) => {
      size += chunk.length;
      if (size > limit) {
        reject(new Error('request body too large'));
        request.destroy?.();
        return;
      }
      chunks.push(chunk);
    });
    request.on('end', () => {
      const text = Buffer.concat(chunks).toString('utf8').trim();
      if (text === '') {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(text));
      } catch {
        reject(new Error('body is not JSON'));
      }
    });
    request.on('error', reject);
  });
}

function send(response, status, body) {
  const text = JSON.stringify(body, null, 2);
  response.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'content-length': Buffer.byteLength(text, 'utf8'),
  });
  response.end(text);
}

function warn(message) {
  console.warn(message);
}
