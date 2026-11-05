/**
 * DSH session-log reader.
 *
 * A DSH session lives at `$DSH_HOME/sessions/<cwd-slug>/<session-id>/session.v4.jsonl.zstd`,
 * one JSON event per line, every line inside one zstd frame.
 *
 * The file's *first* frame is deliberately unreadable to a stock decoder. Its frame
 * descriptor byte carries the reserved bit (`0x04`) while declaring no FCS, no
 * dictionary and no checksum — zstd rejects that as "Unsupported frame parameter".
 * Clearing that one bit makes every frame — first one included — decode cleanly.
 * Measured on a 4.8 MB / 2,811-frame log (2026-11-04): 2811 ok, 0 failed.
 *
 * The decoder is deliberately defensive: one unreadable frame costs that frame's
 * events, never the session, and a log this code cannot decode reports `decodable:
 * false` with the reason instead of pretending the session was empty.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';
// zstd support arrived in Node 22.15. Imported lazily rather than at the top: a
// static `import { zstdDecompressSync }` makes the *whole module* unloadable on
// Node 20, and this module is reached from index.js, so the entire plugin failed
// to start on any Node older than 22.15 with "does not provide an export named".
// A feature that is unavailable should degrade, not take the host down with it.
const zlib = await import('node:zlib');
const zstdDecompressSync = typeof zlib.zstdDecompressSync === 'function' ? zlib.zstdDecompressSync : null;

const ZSTD_MAGIC = Buffer.from([0x28, 0xb5, 0x2f, 0xfd]);
/** The reserved bit in a zstd frame descriptor. See the header comment. */
const RESERVED_BIT = 0x04;

export function dshHome() {
  const raw = process.env.DSH_HOME;
  if (typeof raw === 'string' && raw.trim() !== '') return raw.trim();
  return join(homedir(), '.dsh');
}

/**
 * Decode a frame-aligned zstd file made of concatenated single-frame chunks.
 *
 * Returns `{ lines, frames, failed, decodable, reason }` — never throws. A file
 * with no zstd magic at all is returned as `decodable: false` with the reason,
 * because "I could not read this" and "this session is empty" are different
 * answers and the caller has to be able to tell them apart.
 */
export function decodeSessionFile(bytes) {
  const starts = [];
  for (let i = 0; i + 4 <= bytes.length; i += 1) {
    if (bytes[i] === 0x28 && bytes[i + 1] === 0xb5 && bytes[i + 2] === 0x2f && bytes[i + 3] === 0xfd) starts.push(i);
  }
  if (starts.length === 0) {
    // A plain (uncompressed) JSONL log is a legitimate shape too — some tooling
    // writes it. Accept it rather than calling it corrupt.
    const text = bytes.toString('utf8');
    if (text.trimStart().startsWith('{')) {
      return { lines: text.split('\n').filter((line) => line.trim() !== ''), frames: 0, failed: 0, decodable: true };
    }
    return { lines: [], frames: 0, failed: 0, decodable: false, reason: 'no zstd frame magic and not plain JSONL' };
  }
  starts.push(bytes.length);
  // Refuse honestly rather than throw a ReferenceError deep in a loop: on a Node
  // build without zstd, "this runtime cannot decode zstd" is the answer, and it
  // must be distinguishable from a corrupt file.
  if (zstdDecompressSync === null) {
    return {
      lines: [],
      frames: starts.length - 1,
      failed: starts.length - 1,
      decodable: false,
      reason: `this Node.js (${process.version}) has no zstd support; needs >= 22.15`,
    };
  }
  const chunks = [];
  let failed = 0;
  for (let k = 0; k + 1 < starts.length; k += 1) {
    const frame = Buffer.from(bytes.subarray(starts[k], starts[k + 1]));
    frame[4] &= ~RESERVED_BIT;
    try {
      chunks.push(zstdDecompressSync(frame, { maxOutputLength: 256 * 1024 * 1024 }));
    } catch {
      failed += 1;
    }
  }
  const text = chunks.map((chunk) => chunk.toString('utf8')).join('');
  const lines = text.split('\n').map((line) => line.trim()).filter((line) => line !== '');
  if (failed === starts.length - 1) {
    return { lines: [], frames: starts.length - 1, failed, decodable: false, reason: 'every zstd frame failed to decode' };
  }
  return { lines, frames: starts.length - 1, failed, decodable: true };
}

/** Parse one session file into `{ header, events }`; unparseable lines count as corrupt. */
export function readSessionFile(file) {
  let bytes;
  try {
    bytes = readFileSync(file);
  } catch (error) {
    return { file, decodable: false, reason: `cannot read: ${error?.message ?? error}`, header: null, events: [], corruptLines: 0 };
  }
  const decoded = decodeSessionFile(bytes);
  if (!decoded.decodable) {
    return { file, decodable: false, reason: decoded.reason ?? 'undecodable', header: null, events: [], corruptLines: 0 };
  }
  const events = [];
  let header = null;
  let corruptLines = 0;
  for (const line of decoded.lines) {
    let value;
    try {
      value = JSON.parse(line);
    } catch {
      corruptLines += 1;
      continue;
    }
    if (header === null && value?.type === 'session') {
      header = value;
      continue;
    }
    events.push(value);
  }
  return { file, decodable: true, header, events, corruptLines };
}

/**
 * Every session on disk, newest first.
 *
 * `sessions/<project-slug>/<session-dir>/session.v<N>.jsonl.zstd`. The version
 * segment is read from the filename rather than assumed, so a future v5 is listed
 * (and flagged) instead of silently ignored.
 */
export function listSessions(home = dshHome()) {
  const root = join(home, 'sessions');
  let projects;
  try {
    projects = readdirSync(root, { withFileTypes: true }).filter((entry) => entry.isDirectory());
  } catch {
    return [];
  }
  const out = [];
  for (const project of projects) {
    let dirs;
    try {
      dirs = readdirSync(join(root, project.name), { withFileTypes: true }).filter((entry) => entry.isDirectory());
    } catch {
      continue;
    }
    for (const dir of dirs) {
      const sessionDir = join(root, project.name, dir.name);
      let files;
      try {
        files = readdirSync(sessionDir).filter((name) => /^session\.v\d+\.jsonl\.zstd?$/.test(name));
      } catch {
        continue;
      }
      if (files.length === 0) continue;
      const file = join(sessionDir, files[0]);
      let mtime = 0;
      let size = 0;
      try {
        const stat = statSync(file);
        mtime = stat.mtimeMs;
        size = stat.size;
      } catch { /* a vanished file is simply skipped */ }
      const version = Number(/\.v(\d+)\./.exec(files[0])?.[1] ?? 0);
      out.push({ id: dir.name, project: project.name, file, mtime, size, version });
    }
  }
  return out.sort((a, b) => b.mtime - a.mtime);
}

/**
 * The same list without local file paths — what a panel (and its network view)
 * gets. A session id is a directory name, not a path, so the id stays.
 */
export function publicSessionList(home = dshHome(), limit = 200) {
  return listSessions(home).slice(0, limit).map(({ file, ...rest }) => rest);
}
