/**
 * Moving a dataset out and an adapter back, over the same `ssh` binary the
 * commands use.
 *
 * Deliberately SFTP-over-ssh rather than `rsync` or `scp`:
 *
 *   - `scp`'s protocol is deprecated in recent OpenSSH, and its legacy mode is a
 *     classic path-injection surface (`host:file` is parsed by scp itself)
 *   - `rsync` is not installed by default on Windows or on most minimal images
 *   - `sftp -b` reads commands from a batch file on *stdin*, so paths never touch
 *     a shell and a filename with a quote or a space is not a quoting problem
 *
 * The one rule this module enforces: every path is validated (absolute, no
 * newline, no `..` escape) before it is written into a batch file, because a
 * newline in a filename would become a second sftp command.
 */
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream, mkdirSync, statSync } from 'node:fs';
import { basename, join } from 'node:path';
import { validateTarget } from './ssh.mjs';

/**
 * A remote path we are willing to hand to `sftp -b`.
 *
 * Refused: relative paths (the batch file has no notion of "current directory"
 * we control), a newline (it would start a new sftp command), and a literal `..`
 * segment (an sftp `put` into `../..` writes outside the run directory).
 */
export function validateRemotePath(path) {
  const text = String(path ?? '');
  if (text === '') return { ok: false, error: '远程路径为空' };
  if (!text.startsWith('/')) return { ok: false, error: `远程路径必须是绝对路径：${text}` };
  if (/[\r\n]/.test(text)) return { ok: false, error: '远程路径不能包含换行符' };
  if (text.split('/').includes('..')) return { ok: false, error: `远程路径不能包含 ..：${text}` };
  return { ok: true, value: text };
}

/** An sftp batch line that puts `local` at `remote`, with `-p` to keep mtimes. */
export function putLine(localPath, remotePath) {
  const remote = validateRemotePath(remotePath);
  if (!remote.ok) throw new Error(remote.error);
  // sftp's own quoting: double quotes, with `"` and `\` escaped. Local paths come
  // from our own run directory, but a dataset id is user-supplied, so escape it.
  const quote = (value) => `"${String(value).replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`;
  return `put ${quote(localPath)} ${quote(remote.value)}`;
}

function sftpArgs(target, options = {}) {
  const args = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=4'];
  if (Number.isFinite(options.port) && options.port > 0) args.push('-P', String(options.port));
  args.push('-b', '-');
  args.push(String(target));
  return args;
}

/**
 * The injectable transfer interface the SSH backend consumes.
 *
 * @param {object} input
 * @param {string} [input.sshBinary] Defaults to `sftp`; tests point it at a stub.
 */
export function createSftpTransfer({ env = process.env, sshBinary = 'sftp' } = {}) {
  return {
    id: 'sftp',

    /**
     * Create a remote directory.
     *
     * `sftp -b` has no recursive mkdir, so the path is walked one level at a
     * time; `-mkdir` ignores the "already exists" failure that makes that walk
     * safe to repeat.
     */
    async mkdirp(target, remotePath, options = {}) {
      const validatedTarget = validateTarget(target);
      if (!validatedTarget.ok) throw new Error(validatedTarget.error);
      const remote = validateRemotePath(remotePath);
      if (!remote.ok) throw new Error(remote.error);
      const parts = remote.value.split('/').filter(Boolean);
      const batch = parts.map((_, index) => `-mkdir "/${parts.slice(0, index + 1).join('/')}"`);
      if (batch.length === 0) return;
      await runSftp(sshBinary, validatedTarget.value, batch, options, env);
    },

    /**
     * Upload a file. Returns the byte count and sha256 so the caller can verify
     * the remote copy if it wants to — a truncated upload over a flaky link is a
     * silent way to train on half a dataset.
     */
    async put(target, localPath, remotePath, options = {}) {
      const validatedTarget = validateTarget(target);
      if (!validatedTarget.ok) throw new Error(validatedTarget.error);
      const remote = validateRemotePath(remotePath);
      if (!remote.ok) throw new Error(remote.error);
      const size = statSync(localPath).size;
      const sha256 = await hashFile(localPath);
      await runSftp(sshBinary, validatedTarget.value, [putLine(localPath, remote.value)], options, env);
      return { bytes: size, sha256 };
    },

    /** Download a remote file to a local path. */
    async get(target, remotePath, localPath, options = {}) {
      const validatedTarget = validateTarget(target);
      if (!validatedTarget.ok) throw new Error(validatedTarget.error);
      const remote = validateRemotePath(remotePath);
      if (!remote.ok) throw new Error(remote.error);
      mkdirSync(join(localPath, '..'), { recursive: true });
      const quote = (value) => `"${String(value).replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`;
      await runSftp(sshBinary, validatedTarget.value, [`get ${quote(remote.value)} ${quote(localPath)}`], options, env);
      return { bytes: statSync(localPath).size };
    },
  };
}

/** Resolve when the sftp batch succeeds; reject with sftp's own words otherwise. */
function runSftp(binary, target, lines, options, env) {
  const args = sftpArgs(target, options);
  return new Promise((resolve, reject) => {
    let child;
    try {
      child = spawn(binary, args, { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'], env });
    } catch (error) {
      reject(new Error(`无法启动 ${binary}：${error?.message ?? error}`));
      return;
    }
    let stderr = '';
    child.stdout.on('data', () => {});
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('error', (error) => {
      reject(new Error(error?.code === 'ENOENT'
        ? '找不到 sftp 命令：OpenSSH 客户端没有安装（Windows 可选功能「OpenSSH 客户端」）'
        : `${binary} 启动失败：${error?.message ?? error}`));
    });
    child.on('close', (code) => {
      if (code === 0) resolve({ code });
      else reject(new Error(`sftp 失败（退出码 ${code}）：${stderr.trim().slice(0, 600) || '没有输出'}`));
    });
    child.stdin.end(`${lines.join('\n')}\n`);
  });
}

/** sha256 of a file, streamed so a 2 GB dataset does not sit in memory. */
export function hashFile(path) {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    const stream = createReadStream(path);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

/** Upload every file in a run's staged data directory. */
export async function uploadDataset(transfer, target, localDir, remoteDir, options = {}) {
  const { readdirSync } = await import('node:fs');
  const files = readdirSync(localDir, { withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name)
    .sort();
  if (files.length === 0) throw new Error(`本地数据目录是空的：${localDir}`);
  await transfer.mkdirp(target, remoteDir, options);
  const uploaded = [];
  for (const name of files) {
    const localPath = join(localDir, name);
    const remotePath = `${remoteDir}/${basename(name)}`;
    const result = await transfer.put(target, localPath, remotePath, options);
    uploaded.push({ name, ...result });
  }
  return { files: uploaded, bytes: uploaded.reduce((sum, file) => sum + file.bytes, 0) };
}

/**
 * Pull `adapter/` (and `merged/` when the user asked for a merge) back.
 *
 * The file list comes from a remote `ls` rather than an sftp `ls` parser: sftp's
 * long-listing format is human-oriented and has changed shape between OpenSSH
 * versions, while `ls -1` is one name per line and stable. Names are read as
 * opaque strings and validated by `validateRemotePath`'s rules (no newline, no ..)
 * before they are used.
 *
 * `sshExec` is injected so this stays testable without a network.
 */
export async function downloadArtifacts(transfer, target, remoteOut, localOut, { sshExec, depth = 0 } = {}) {
  if (depth > 2) return { files: [], skipped: ['嵌套超过两层，已忽略'] };
  const pulled = [];
  const skipped = [];
  for (const dir of ['adapter', 'merged']) {
    const remoteDir = `${remoteOut}/${dir}`;
    const listing = await listRemote(sshExec, target, remoteDir);
    if (!listing.ok) {
      skipped.push(`${dir}: ${listing.reason}`);
      continue;
    }
    mkdirSync(join(localOut, dir), { recursive: true });
    for (const name of listing.entries) {
      const localPath = join(localOut, dir, name);
      await transfer.get(target, `${remoteDir}/${name}`, localPath);
      pulled.push(`${dir}/${name}`);
    }
  }
  return { files: pulled, skipped };
}

/**
 * One name per line, or the reason it failed.
 *
 * `ls -1` on a path we built ourselves; the path is quoted because it contains the
 * user-supplied remote root.
 */
async function listRemote(sshExec, target, remoteDir) {
  const validated = validateRemotePath(remoteDir);
  if (!validated.ok) return { ok: false, reason: validated.error };
  const { shellQuote } = await import('./ssh.mjs');
  try {
    const result = await sshExec(target, [`ls -1 ${shellQuote(validated.value)} 2>/dev/null`], { timeoutMs: 30_000 });
    if (result.code !== 0) return { ok: false, reason: '远端不存在（没开启合并时这是正常的）' };
    const entries = String(result.stdout ?? '')
      .split('\n')
      .map((line) => line.trim())
      .filter((name) => name !== '' && !name.includes('/') && name !== '.' && name !== '..');
    return { ok: true, entries };
  } catch (error) {
    return { ok: false, reason: error?.message ?? String(error) };
  }
}

export { createWriteStream };
