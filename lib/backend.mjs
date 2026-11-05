/**
 * Backend adapters: where a training run actually executes.
 *
 * Both backends speak the same contract, so the job store, the panel and the
 * tools never branch on which one ran:
 *
 *   start(jobId, spec, hooks) -> handle with { pid, kill() }
 *   - the worker writes JSON lines to stdout (`{"kind":"metric",...}`), which the
 *     adapter parses and hands to `hooks.onEvent`
 *   - stderr is teed to `runs/<id>/stderr.log` — a crashed torch run explains
 *     itself there, not in an API response
 *   - `hooks.onExit({ code, signal, error })` fires exactly once
 *
 * `local` runs `spawn` by argv, never through a shell: the model id is the one
 * user-controlled string that reaches a command line, and quoting it into
 * `cmd.exe` on Windows is how `--model && del` happens. A plugin `shellQuote`
 * is also provided because the SSH path does need a shell.
 *
 * `ssh` runs `python3 -m fnt_worker <verb> --config <json>` on the configured
 * target: the JSON is passed as one shell-quoted argument, so nothing inside it
 * can be interpreted by the remote shell.
 */
import { spawn } from 'node:child_process';
import { appendFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { StringDecoder } from 'node:string_decoder';

const MAX_LINE = 256 * 1024;

/** Split a byte stream into lines, keeping partial lines between chunks. */
export function createLineReader(onLine) {
  const decoder = new StringDecoder('utf8');
  let buffer = '';
  return {
    push(chunk) {
      buffer += decoder.write(chunk);
      let index;
      while ((index = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, index).replace(/\r$/, '');
        buffer = buffer.slice(index + 1);
        if (line.trim() !== '') onLine(line);
      }
      if (buffer.length > MAX_LINE) {
        // A worker that prints a 10 MB line is misbehaving; keep the run alive.
        onLine(buffer.slice(0, MAX_LINE));
        buffer = '';
      }
    },
    flush() {
      buffer += decoder.end();
      if (buffer.trim() !== '') onLine(buffer);
      buffer = '';
    },
  };
}

/** Parse one worker stdout line into an event, or null when it is not a JSON event. */
export function parseWorkerLine(raw) {
  const text = String(raw ?? '').trim();
  if (!text.startsWith('{')) return null;
  try {
    const value = JSON.parse(text);
    if (value === null || typeof value !== 'object' || typeof value.kind !== 'string') return null;
    return value;
  } catch {
    return null;
  }
}

/** POSIX single-quote: safe for newlines, quotes, backslashes, `$`, `!`. */
export function shellQuote(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

/**
 * The environment a worker process gets.
 *
 * `TMPDIR`/`TEMP`/`TMP` are pointed at a directory this plugin owns, not at the
 * system temp. Two reasons, both observed on Windows:
 *
 *   - the system temp had 1408 entries, and creating a file there from Python
 *     blocked for minutes (a scanner, most likely). `tempfile` treats a slow or
 *     refused probe as "unusable" and falls back ? its next choice is the current
 *     working directory, so `gettempdir()` silently became the checkout.
 *   - once every candidate is rejected, `tempfile` raises
 *     "No usable temporary directory found", which the worker reported as a
 *     missing-dependency error. That message sent the user to install torch again
 *     when torch was already there.
 *
 * A dedicated directory is small, ours, and always writable, so the probe
 * succeeds immediately.
 */
function probeEnv(settings, tmpDir) {
  const env = { ...process.env, ...(settings.env ?? {}) };
  if (typeof tmpDir === 'string' && tmpDir !== '') {
    env.TMPDIR = tmpDir;
    env.TEMP = tmpDir;
    env.TMP = tmpDir;
  }
  return env;
}

/**
 * Local backend.
 *
 * @param {object} input
 * @param {{pythonPath: string, workerRoot?: string, env?: Record<string,string>}} input.settings
 * @param {string} input.root Plugin data root.
 */
export function createLocalBackend({ settings, root, log = () => {} }) {
  const python = () => {
    const value = settings.pythonPath;
    if (typeof value !== 'string' || value.trim() === '') {
      throw new Error('未配置 pythonPath：在 cordis.patch.yml 的 fine-tune 条目里填一个装了 torch 的 Python 解释器路径。');
    }
    const path = value.trim();
    // A path that is set but wrong used to report "not configured", which sends
    // the user looking for a switch that is already on. Say what is actually
    // wrong — and name the field, because that is the only thing to edit.
    if (!existsSync(path)) {
      throw new Error(`pythonPath 指向的解释器不存在：${path}（在 cordis.patch.yml 的 fine-tune → config → pythonPath 里改）`);
    }
    return path;
  };
  const cwd = () => (typeof settings.workerRoot === 'string' && settings.workerRoot.trim() !== '' ? settings.workerRoot : root);
  // Created on demand: a plugin that never trains should not leave a directory
  // behind. The worker uses it for tempfile, torch's compile cache and dataloader
  // workers.
  let tmpDirCache = null;
  const tmpDir = () => {
    if (tmpDirCache !== null) return tmpDirCache;
    const dir = join(root, 'tmp');
    try {
      mkdirSync(dir, { recursive: true });
      tmpDirCache = dir;
    } catch {
      // Falling back to the system temp is better than refusing to start; the
      // worker reports the real problem if that one is unusable too.
      tmpDirCache = '';
    }
    return tmpDirCache;
  };
  const spawnPython = (args, extra) => spawn(python(), args, {
    cwd: cwd(),
    env: { ...probeEnv(settings, tmpDir()), PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
    windowsHide: true,
    ...extra,
  });
  return {
    id: 'local',
    async probe() {
      return new Promise((resolve) => {
        let child;
        try {
          child = spawnPython(['-c', 'import sys; print(sys.version.split()[0])']);
        } catch (error) {
          resolve({ ok: false, reason: error?.message ?? String(error) });
          return;
        }
        let output = '';
        const reader = createLineReader((line) => { output = line.trim(); });
        child.stdout.on('data', (chunk) => reader.push(chunk));
        child.stderr.on('data', () => {});
        child.on('error', (error) => resolve({ ok: false, reason: `无法启动 Python：${error.message}` }));
        child.on('exit', (code) => {
          reader.flush();
          resolve(code === 0 ? { ok: true, python: output } : { ok: false, reason: `Python 退出码 ${code}` });
        });
      });
    },
    start(jobId, spec, hooks) {
      const runDir = join(root, 'runs', jobId);
      mkdirSync(runDir, { recursive: true });
      const configPath = join(runDir, 'worker-config.json');
      writeFileSync(configPath, `${JSON.stringify(spec, null, 2)}\n`, 'utf8');
      const stderrPath = join(runDir, 'stderr.log');
      writeFileSync(stderrPath, '', 'utf8');
      let child;
      try {
        child = spawnPython(['-m', 'fnt_worker', spec.verb, '--config', configPath]);
      } catch (error) {
        hooks.onExit({ code: null, signal: null, error: error?.message ?? String(error), stderrPath });
        return { pid: null, kill() {} };
      }
      const reader = createLineReader((line) => {
        const event = parseWorkerLine(line);
        if (event === null) {
          appendFileSafe(stderrPath, `${line}\n`);
          return;
        }
        hooks.onEvent(event);
      });
      child.stdout.on('data', (chunk) => reader.push(chunk));
      child.stderr.on('data', (chunk) => appendFileSafe(stderrPath, chunk));
      let settled = false;
      const settle = (payload) => {
        if (settled) return;
        settled = true;
        reader.flush();
        hooks.onExit({ ...payload, stderrPath });
      };
      child.on('error', (error) => settle({ code: null, signal: null, error: error.message }));
      child.on('exit', (code, signal) => settle({ code, signal }));
      return {
        pid: child.pid ?? null,
        kill() {
          try {
            // On Windows `child.kill()` already maps to TaskKill for the tree.
            child.kill();
          } catch (error) {
            log(`fine-tune: could not signal ${jobId}: ${error?.message ?? error}`);
          }
        },
      };
    },
  };
}

function appendFileSafe(path, chunk) {
  try {
    appendFileSync(path, chunk, 'utf8');
  } catch { /* a full disk is reported by the exit code, not here */ }
}

/**
 * SSH backend.
 *
 * The remote side needs a Python 3 with torch and the `fnt_worker` module on its
 * path; M0-LAB.md has the one-command setup. Datasets and the worker source are
 * uploaded by the caller (which knows the dataset layout); artifacts are pulled
 * back after exit.
 *
 * @param {object} input
 * @param {(target: string, argv: string[], options: object) => Promise<{code:number}>} input.sshExec
 *   Injected by index.js (`createSshExec`) so this module stays testable without
 *   a network. It must call `options.onStdout(chunk)` / `options.onStderr(chunk)`
 *   as output arrives and reject on transport failure.
 */
export function createSshBackend({ settings, root, sshExec, log = () => {} }) {
  return {
    id: 'ssh',
    async probe(target) {
      let output = '';
      try {
        // One remote command, quoted once: version + torch + worker module.
        const command = [
          'python3', '-V',
          '&&', 'python3', '-c', shellQuote('import torch;print("torch",torch.__version__)'),
          '&&', 'python3', '-c', shellQuote('import fnt_worker;print("worker","ok")'),
        ].join(' ');
        const result = await sshExec(target, [command], {
          onStdout: (chunk) => { output += chunk; },
          onStderr: (chunk) => { output += chunk; },
          timeoutMs: 30_000,
        });
        return {
          ok: result.code === 0,
          target,
          detail: output.trim().slice(0, 600),
          reason: result.code === 0 ? null : output.trim().slice(0, 600) || `exit ${result.code}`,
        };
      } catch (error) {
        return { ok: false, target, reason: error?.message ?? String(error) };
      }
    },
    start(jobId, spec, hooks) {
      const target = spec.target;
      const remoteRoot = String(spec.remoteRoot ?? '/tmp/dsh-fine-tune').replace(/\/+$/, '');
      const runDir = join(root, 'runs', jobId);
      mkdirSync(runDir, { recursive: true });
      const stderrPath = join(runDir, 'stderr.log');
      writeFileSync(stderrPath, '', 'utf8');
      const remoteRun = `${remoteRoot}/runs/${jobId}`;
      // Only the config is written remotely here; datasets are uploaded by the
      // caller before start(), artifacts are pulled after exit.
      const remoteSpec = { ...spec, out: remoteRun };
      const config = JSON.stringify(remoteSpec, null, 2);
      // Every path is quoted: remoteRoot and jobId both reach the remote shell.
      const command = [
        'mkdir', '-p', shellQuote(remoteRun),
        '&&', 'printf', shellQuote('%s'), shellQuote(config), '>', shellQuote(`${remoteRun}/worker-config.json`),
        '&&', 'cd', shellQuote(remoteRoot),
        '&&', 'PYTHONUNBUFFERED=1', 'python3', '-m', 'fnt_worker', shellQuote(spec.verb),
        '--config', shellQuote(`${remoteRun}/worker-config.json`),
      ].join(' ');
      const controller = new AbortController();
      const reader = createLineReader((line) => {
        const event = parseWorkerLine(line);
        if (event === null) {
          appendFileSafe(stderrPath, `${line}\n`);
          return;
        }
        hooks.onEvent(event);
      });
      let settled = false;
      const settle = (payload) => {
        if (settled) return;
        settled = true;
        reader.flush();
        hooks.onExit({ ...payload, stderrPath });
      };
      sshExec(target, [command], {
        signal: controller.signal,
        onStdout: (chunk) => reader.push(chunk),
        onStderr: (chunk) => appendFileSafe(stderrPath, chunk),
      }).then((result) => settle({ code: result.code, signal: null, remoteRoot }))
        .catch((error) => {
          if (error?.name === 'AbortError') settle({ code: null, signal: 'SIGTERM', remoteRoot });
          else settle({ code: null, signal: null, error: error?.message ?? String(error), remoteRoot });
        });
      return {
        pid: null,
        kill() {
          try {
            controller.abort();
          } catch (error) {
            log(`fine-tune: could not stop remote run ${jobId}: ${error?.message ?? error}`);
          }
        },
      };
    },
  };
}
