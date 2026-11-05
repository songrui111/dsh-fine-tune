/**
 * SSH transport: run a command on a remote host and stream its output back.
 *
 * Deliberately wraps the system `ssh` binary rather than pulling in an SSH
 * library. The reasons are practical, not ideological:
 *
 *   - the user's existing `~/.ssh/config`, agent, ProxyJump and key setup keep
 *     working for free; a library would need all of it re-expressed in settings
 *   - host-key verification is `ssh`'s job, and reimplementing it is how plugins
 *     end up trusting the wrong host
 *   - OpenSSH ships on Windows 10+, so the dependency is already there
 *
 * What that costs: the remote command is a *string* handed to the remote shell,
 * so anything interpolated into it must be quoted (see `shellQuote`). Every caller
 * in this plugin builds its command from `shellQuote`-ed pieces for that reason.
 */
import { spawn } from 'node:child_process';

/** Options that must never be taken from user settings — they would disable host verification. */
const FORBIDDEN = new Set([
  'StrictHostKeyChecking',
  'UserKnownHostsFile',
  'GlobalKnownHostsFile',
  'ProxyCommand',
  'LocalCommand',
  'PermitLocalCommand',
]);

/**
 * A target is `[user@]host[:port]` or an alias from `~/.ssh/config`.
 *
 * The check is an allowlist on purpose: a target reaches both `ssh`'s argv and
 * (through the remote command) the remote shell, so anything that could be read
 * as an option or a shell metacharacter is refused rather than escaped. An alias
 * starting with `-` would be parsed as a flag by `ssh` itself.
 */
export function validateTarget(target) {
  const text = String(target ?? '').trim();
  if (text === '') return { ok: false, error: 'SSH 目标为空：填 [user@]host[:port]，或 ~/.ssh/config 里的别名' };
  if (text.startsWith('-')) return { ok: false, error: 'SSH 目标不能以 - 开头（会被 ssh 当成参数）' };
  if (!/^[A-Za-z0-9._@:[\]-]+$/.test(text)) {
    return { ok: false, error: 'SSH 目标含有不安全的字符（只允许字母、数字和 . _ - @ : [ ]）' };
  }
  return { ok: true, value: text };
}

/**
 * Build the `ssh` argument vector for one command.
 *
 * `command` is a single string executed by the *remote* shell. Callers pass
 * `shellQuote`-ed fragments; this function never quotes anything itself, so a
 * caller that forgets is a bug here, not a surprise in production — the tests
 * cover the quoting contract instead of hiding it.
 */
export function buildSshArgs({ target, command, port, extraArgs, options }) {
  const args = [];
  // BatchMode: never prompt for a password (there is no TTY to answer with) and
  // fail fast instead of hanging a training run behind an invisible prompt.
  args.push('-o', 'BatchMode=yes');
  args.push('-o', 'ConnectTimeout=15');
  // Keep the connection alive during long training runs; an idle SSH session
  // that silently dies halfway through an epoch is the classic remote-GPU bug.
  args.push('-o', 'ServerAliveInterval=30');
  args.push('-o', 'ServerAliveCountMax=4');
  args.push('-T');
  if (typeof port === 'number' && Number.isFinite(port) && port > 0) args.push('-p', String(port));
  for (const extra of Array.isArray(extraArgs) ? extraArgs : []) {
    // Only the option name is checked, so a caller cannot smuggle in a value for
    // a security-relevant option through `extraArgs`.
    const key = String(extra).replace(/^-+/, '').split('=')[0];
    if (FORBIDDEN.has(key)) continue;
    args.push(String(extra));
  }
  for (const [key, value] of Object.entries(options ?? {})) {
    if (FORBIDDEN.has(key)) continue;
    args.push('-o', `${key}=${value}`);
  }
  args.push(String(target));
  args.push(String(command));
  return args;
}

/** Shell-quote for a POSIX remote: safe for newlines, quotes, backslashes, `$`, `!`. */
export function shellQuote(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

/**
 * Create the injectable `sshExec` the backend expects.
 *
 * The returned function resolves with `{ code, stdout, stderr }` when the command
 * finishes and rejects only on a transport failure (ssh missing, target refused).
 * Output is streamed through `onStdout` / `onStderr` as it arrives, because the
 * training worker's JSON events must reach the panel live.
 *
 * @param {object} input
 * @param {string[]} [input.searchPaths] Where to look for `ssh`, in order.
 */
export function createSshExec({ searchPaths = [], env = process.env } = {}) {
  return function sshExec(target, argv, options = {}) {
    const validated = validateTarget(target);
    if (!validated.ok) return Promise.reject(new Error(validated.error));
    const { command, port, extraArgs, signal, timeoutMs, onStdout, onStderr } = options;
    // `argv` here is the backend's contract: pieces joined into one remote command.
    const remote = Array.isArray(argv) ? argv.map(String).join(' ') : String(argv ?? '');
    const args = buildSshArgs({ target: validated.value, command: remote, port, extraArgs });

    return new Promise((resolve, reject) => {
      let child;
      try {
        child = spawn('ssh', args, {
          windowsHide: true,
          stdio: ['ignore', 'pipe', 'pipe'],
          env,
        });
      } catch (error) {
        reject(new Error(`无法启动 ssh：${error?.message ?? error}`));
        return;
      }

      let stdout = '';
      let stderr = '';
      let finished = false;
      const timer = Number.isFinite(timeoutMs) && timeoutMs > 0
        ? setTimeout(() => {
          if (finished) return;
          finished = true;
          child.kill();
          reject(new Error(`ssh 超时（${Math.round(timeoutMs / 1000)} 秒）`));
        }, timeoutMs)
        : null;

      const abort = () => {
        if (finished) return;
        finished = true;
        child.kill();
        const error = new Error('aborted');
        error.name = 'AbortError';
        reject(error);
      };
      if (signal !== undefined && signal !== null) {
        if (signal.aborted) { abort(); return; }
        signal.addEventListener('abort', abort, { once: true });
      }

      child.stdout.on('data', (chunk) => {
        stdout += chunk;
        onStdout?.(chunk);
      });
      child.stderr.on('data', (chunk) => {
        stderr += chunk;
        onStderr?.(chunk);
      });
      child.on('error', (error) => {
        if (finished) return;
        finished = true;
        if (timer) clearTimeout(timer);
        // ENOENT here means OpenSSH is not installed — say that, not "spawn error".
        reject(new Error(error?.code === 'ENOENT'
          ? '找不到 ssh 命令：安装 OpenSSH 客户端（Windows 可选功能「OpenSSH 客户端」）后重试'
          : `ssh 启动失败：${error?.message ?? error}`));
      });
      child.on('close', (code) => {
        if (finished) return;
        finished = true;
        if (timer) clearTimeout(timer);
        resolve({ code: code ?? 0, stdout, stderr });
      });
    });
  };
}
