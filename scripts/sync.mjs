#!/usr/bin/env node
/**
 * Copy this plugin's source into the DSH profile that actually loads it.
 *
 * Why this exists: the profile installs the plugin as a *copy*
 * (`node_modules/dsh-fine-tune`), not a symlink, so editing a file here changes
 * nothing until it is copied over — and a panel change that never reaches the
 * install looks exactly like a panel change that does not work. That mistake cost
 * a full restart-and-reload cycle once; this script is the fix.
 *
 * Files DSH has open (its ESM host modules are loaded on demand) cannot be
 * overwritten on Windows. Those are reported as `busy` and skipped, not treated
 * as failures: a busy file is either identical already or is a host module that
 * only changes on a DSH restart, so the run prints the truth and exits 0 unless
 * a *client* file failed.
 *
 * Usage:
 *   node scripts/sync.mjs               # sync to the default profile (desktop)
 *   node scripts/sync.mjs --profile x   # sync to another profile name
 *   node scripts/sync.mjs --dry-run     # show what would change
 */

import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { homedir } from 'node:os';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = dirname(here);
const argv = process.argv.slice(2);
const dryRun = argv.includes('--dry-run');
const profileFlag = argv.indexOf('--profile');
const profile = profileFlag !== -1 && argv[profileFlag + 1] ? argv[profileFlag + 1] : 'desktop';

const dshHome = process.env.DSH_HOME || join(homedir(), '.dsh');
const target = join(dshHome, 'profiles', profile, 'node_modules', 'dsh-fine-tune');

/** Everything the plugin needs at runtime. Tests and docs do not need copying. */
const FILES = [
  'index.js',
  'package.json',
  'cordis.patch.yml',
  'client/client.js',
  'lib/backend.mjs',
  'lib/dataset.mjs',
  'lib/evaluate.mjs',
  'lib/hardware.mjs',
  'lib/jobs.mjs',
  'lib/pack.mjs',
  'lib/redact.mjs',
  'lib/session.mjs',
  'lib/sources.mjs',
  'lib/ssh.mjs',
  'lib/tools.mjs',
  'lib/transfer.mjs',
  'lib/training.mjs',
  'lib/textmetric.mjs',
  'python/fnt_worker/__init__.py',
  'python/fnt_worker/__main__.py',
  'python/fnt_worker/calibrate.py',
  'python/fnt_worker/textmetric.py',
];

/** A file whose bytes already match does not need writing (and avoids a busy lock). */
function identical(from, to) {
  if (!existsSync(to)) return false;
  const a = readFileSync(from);
  const b = readFileSync(to);
  return a.length === b.length && a.equals(b);
}

const report = { copied: [], same: [], busy: [], missing: [] };

if (!existsSync(target)) {
  console.error(`目标不存在：${target}`);
  console.error('先把这个插件装进 profile（在 profile 的 package.json 里加 file: 依赖），再跑这个脚本。');
  process.exit(1);
}

for (const file of FILES) {
  const from = join(root, file);
  const to = join(target, file);
  if (!existsSync(from)) {
    report.missing.push(file);
    continue;
  }
  if (identical(from, to)) {
    report.same.push(file);
    continue;
  }
  if (dryRun) {
    report.copied.push(`${file} (dry-run)`);
    continue;
  }
  try {
    mkdirSync(dirname(to), { recursive: true });
    copyFileSync(from, to);
    report.copied.push(file);
  } catch (error) {
    // EBUSY / EPERM: DSH has the module open. Nothing to do but say so.
    report.busy.push(`${file} (${error.code ?? error.message})`);
  }
}

// A file deleted from the source must also disappear from the profile. Copying
// alone leaves the old module behind, and because the entry point no longer
// imports it nothing fails ? until someone reads the profile and believes it is
// still part of the plugin. Only files in directories the manifest covers are
// considered: nothing else in the profile is this script's to delete.
const stale = [];
if (!dryRun) {
  const ownedDirs = new Set(FILES.map((file) => dirname(file)));
  for (const dir of ownedDirs) {
    const to = join(target, dir);
    if (!existsSync(to)) continue;
    for (const entry of readdirSync(to, { withFileTypes: true })) {
      if (!entry.isFile()) continue;
      const rel = dir === '.' ? entry.name : `${dir}/${entry.name}`;
      if (FILES.includes(rel)) continue;
      if (existsSync(join(root, rel))) continue;
      try {
        rmSync(join(target, rel), { force: true });
        stale.push(rel);
      } catch (error) {
        stale.push(`${rel} (delete failed: ${error.code ?? error.message})`);
      }
    }
  }
}

const line = (label, items, marker) => {
  if (items.length === 0) return;
  console.log(`${marker} ${label} (${items.length})`);
  for (const item of items) console.log(`    ${item}`);
};

console.log(`源：${root}`);
console.log(`目标：${target}`);
line('已更新', report.copied, '=');
line('已是最新', report.same, '=');
line('被 DSH 占用，跳过（重启后可同步）', report.busy, '!');
line('源文件缺失', report.missing, '!');
line('已从目标删除（源里已不存在）', stale, '=');

// The client bundle is the one file whose staleness is invisible from the UI:
// it loads in the browser, so a stale copy shows the *old* panel with no error.
const clientStale = report.copied.some((file) => file.startsWith('client/'))
  || report.busy.some((file) => file.startsWith('client/'));
if (clientStale) {
  console.log('\n面板代码有更新：重启 DSH 并刷新页面后才会生效。');
} else if (report.copied.length === 0) {
  console.log('\n没有需要同步的改动。');
}

process.exit(report.busy.some((file) => file.startsWith('client/')) ? 1 : 0);
