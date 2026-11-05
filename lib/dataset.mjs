/**
 * Dataset building: source → mined samples → packed splits → a versioned dataset
 * with a data card.
 *
 * The data card is a product feature, not paperwork. It is what makes a run
 * reproducible and what makes a dataset reviewable before it is trained on: how
 * many sessions, how many samples survived each filter, what was redacted and how
 * often, where each split came from. Every number in it is computed here, so a
 * reader never has to trust a promise in the panel.
 */
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { redactText } from './redact.mjs';
import { packSamples, median, sampleTokens, splitSamples, writeDataset } from './pack.mjs';

/**
 * Which samples a user has manually removed from a dataset.
 *
 * Exclusions are stored separately from the data (`exclusions.json`) and applied
 * at read time, so "remove this sample" is reversible without rebuilding: the
 * dataset on disk stays exactly what the miner produced, and the *effective*
 * dataset is the miner's output minus the excluded hashes.
 *
 * Every consumer goes through `loadSplit`, which applies them:
 *   - the panel's preview      (`previewPage`)
 *   - training                 (`writeFilteredDataset` → the dir the worker reads)
 *   - the evaluation pair      (`evaluationPair`)
 * so a sample removed in the UI is never trained on or graded, by construction.
 */
export function exclusionsFile(dir) {
  return join(dir, 'exclusions.json');
}

export function loadExclusions(dir) {
  try {
    const parsed = JSON.parse(readFileSync(exclusionsFile(dir), 'utf8'));
    return Array.isArray(parsed.hashes) ? parsed.hashes.map(String) : [];
  } catch {
    return [];
  }
}

export function saveExclusions(dir, hashes) {
  const unique = [...new Set(hashes.map(String).filter((hash) => /^[a-f0-9]{8,64}$/.test(hash)))].sort();
  writeFileSync(exclusionsFile(dir), `${JSON.stringify({ hashes: unique }, null, 2)}\n`, 'utf8');
  return unique;
}

/** Read one split from disk, minus the excluded hashes. */
export function loadSplit(dir, split) {
  const file = join(dir, `${split}.jsonl`);
  const excluded = new Set(loadExclusions(dir));
  const rows = [];
  try {
    for (const line of readFileSync(file, 'utf8').split('\n')) {
      if (line.trim() === '') continue;
      let row;
      try {
        row = JSON.parse(line);
      } catch {
        continue;
      }
      if (excluded.has(row.hash)) continue;
      rows.push(row);
    }
  } catch {
    return [];
  }
  return rows;
}

/** All three splits in their effective form. */
export function loadSplits(dir) {
  return { train: loadSplit(dir, 'train'), val: loadSplit(dir, 'val'), eval: loadSplit(dir, 'eval') };
}

/**
 * Materialise several datasets into one, for a run that trains on all of them.
 *
 * Concatenation per split, not a re-split: each source already decided which of
 * its rows are train/val/eval, and reshuffling across sources would move samples
 * a reviewer had already inspected. So the merged validation set is the union of
 * the sources' validation sets — which means a model can be validated on rows from
 * the same source it trained on if the sources overlap. That cannot happen here
 * because a row belongs to exactly one dataset, but it is why the result reports
 * `sources` explicitly rather than looking like a single curated file.
 *
 * Duplicate rows across sources are kept: two datasets may legitimately contain
 * the same prompt with different labels (that is a disagreement worth training
 * on), and silently dropping one would hide it. `duplicates` reports how many
 * exact repeats were seen so the number is available, not invisible.
 *
 * @param {string[]} dirs Source dataset directories, in the order given.
 * @param {string} targetDir Where the merged files are written.
 */
export function writeMergedDataset(dirs, targetDir) {
  if (!Array.isArray(dirs) || dirs.length === 0) throw new Error('至少要选一个数据集');
  mkdirSync(targetDir, { recursive: true });
  const merged = { train: [], val: [], eval: [] };
  const seen = new Map(); // split -> Set of row hashes, for the duplicate count
  let duplicates = 0;
  let excluded = 0;
  for (const dir of dirs) {
    const splits = loadSplits(dir);
    excluded += loadExclusions(dir).length;
    for (const split of ['train', 'val', 'eval']) {
      if (!seen.has(split)) seen.set(split, new Set());
      const bucket = seen.get(split);
      for (const row of splits[split]) {
        const key = `${row.hash ?? ''}|${JSON.stringify(row.messages ?? [])}`;
        if (bucket.has(key)) duplicates += 1;
        bucket.add(key);
        merged[split].push(row);
      }
    }
  }
  for (const split of ['train', 'val', 'eval']) {
    const body = merged[split].map((row) => JSON.stringify(row)).join('\n');
    writeFileSync(join(targetDir, `${split}.jsonl`), body === '' ? '' : `${body}\n`, 'utf8');
  }
  return {
    dir: targetDir,
    sources: dirs,
    excluded,
    duplicates,
    counts: { train: merged.train.length, val: merged.val.length, eval: merged.eval.length },
  };
}

/**
 * Materialise the *effective* dataset into `targetDir` for the worker.
 *
 * The worker reads plain `train.jsonl` / `val.jsonl`; the exclusions live in the
 * source dataset, not in the copy, so this is where the two are reconciled. It is
 * called immediately before training, which is why a sample removed one minute
 * earlier is absent from the run.
 *
 * @returns {{ dir: string, counts: {train:number,val:number,eval:number,excluded:number}, source: string }}
 */
export function writeFilteredDataset(dir, targetDir) {
  const splits = loadSplits(dir);
  const excluded = loadExclusions(dir);
  mkdirSync(targetDir, { recursive: true });
  for (const split of ['train', 'val', 'eval']) {
    const rows = splits[split];
    const body = rows.map((row) => JSON.stringify(row)).join('\n');
    writeFileSync(join(targetDir, `${split}.jsonl`), body === '' ? '' : `${body}\n`, 'utf8');
  }
  return {
    dir: targetDir,
    source: dir,
    excluded: excluded.length,
    counts: { train: splits.train.length, val: splits.val.length, eval: splits.eval.length },
  };
}

/**
 * A page of samples for the preview pane, prompt and label each shown as a
 * prefix plus a suffix — the interesting parts of a sample are where the request
 * states the situation and where the label starts, not 2000 chars of a table in
 * the middle.
 *
 * @returns {{ rows: Array<object>, total: number, start: number, pageSize: number }}
 */
export function previewPage(dir, { split = 'train', limit = 50, offset = 0, chars = 240 } = {}) {
  if (!['train', 'val', 'eval'].includes(split)) throw new Error(`未知切分: ${split}`);
  const all = loadSplit(dir, split);
  const pageSize = Math.min(200, Math.max(1, Math.trunc(Number(limit) || 50)));
  const start = Math.min(all.length, Math.max(0, Math.trunc(Number(offset) || 0)));
  const rows = all.slice(start, start + pageSize).map((row) => {
    const messages = Array.isArray(row.messages) ? row.messages : [];
    const prompt = String(messages[0]?.content ?? '');
    const label = String(messages[messages.length - 1]?.content ?? '');
    return {
      hash: row.hash ?? null,
      meta: row.meta ?? {},
      prompt: excerpt(prompt, chars),
      label: excerpt(label, chars),
      promptChars: prompt.length,
      labelChars: label.length,
    };
  });
  return { rows, total: all.length, start, pageSize };
}

function excerpt(text, chars) {
  const limit = Math.max(40, Math.trunc(Number(chars) || 240));
  if (text.length <= limit) return text;
  const head = Math.ceil(limit * 0.65);
  const tail = limit - head;
  return `${text.slice(0, head)}\n…〔省略 ${text.length - limit} 字符〕…\n${text.slice(text.length - tail)}`;
}

/** Read a local dataset file (JSONL of {messages}, or JSON holding such an array). */
export function readDatasetFile(file) {
  const text = String(readFileSync(file, 'utf8')).trim();
  if (text === '') return { rows: [], skipped: 0 };
  if (text.startsWith('[')) {
    const rows = JSON.parse(text);
    return { rows: Array.isArray(rows) ? rows : [], skipped: 0 };
  }
  const rows = [];
  let skipped = 0;
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (trimmed === '') continue;
    try {
      rows.push(JSON.parse(trimmed));
    } catch {
      skipped += 1;
    }
  }
  return { rows, skipped };
}

/** Map one external dataset row to the chat shape, tolerating the usual flavours. */
export function normaliseRow(row, sourceId) {
  if (row === null || typeof row !== 'object') return null;
  if (Array.isArray(row.messages)) {
    const messages = row.messages
      .filter((message) => message !== null && typeof message === 'object' && typeof message.role === 'string')
      .map((message) => ({ role: message.role, content: typeof message.content === 'string' ? message.content : JSON.stringify(message.content ?? '') }));
    if (messages.length >= 2) return { messages, meta: { sourceId, group: String(row.group ?? row.sessionId ?? sourceId), origin: 'dataset' } };
    return null;
  }
  const prompt = row.prompt ?? row.instruction ?? row.question ?? row.input ?? row.system_prompt;
  const answer = row.response ?? row.output ?? row.answer ?? row.completion;
  if (typeof prompt === 'string' && typeof answer === 'string' && prompt.trim() !== '' && answer.trim() !== '') {
    return { messages: [{ role: 'user', content: prompt }, { role: 'assistant', content: answer }], meta: { sourceId, group: String(row.group ?? row.sessionId ?? sourceId), origin: 'dataset' } };
  }
  return null;
}


/** Build a dataset directory. Returns the manifest written to disk. */
export function buildDataset({ root, source, datasetId, dshHome, options = {} }) {
  const dir = join(root, 'datasets', datasetId);
  mkdirSync(dir, { recursive: true });
  let raw = [];
  let provenance = {};
  // Only external files build datasets now. Mining session logs was removed from
  // both halves: the picker was judged unnecessary (a user who wants to train has
  // data already) and keeping a half-served API would leave a code path nothing
  // can reach and nobody tests.
  if (source.kind === 'dataset') {
    const { rows, skipped } = readDatasetFile(source.file);
    const normalised = [];
    let invalid = 0;
    for (const row of rows) {
      const mapped = normaliseRow(row, source.id);
      if (mapped === null) {
        invalid += 1;
        continue;
      }
      const redacted = (message) => redactText(message.content, source.redaction).text;
      normalised.push({
        messages: mapped.messages.map((message) => ({ role: message.role, content: message.role === 'user' ? redacted(message) : message.content })),
        meta: mapped.meta,
      });
    }
    raw = normalised;
    provenance = { file: source.file, rows: rows.length, skippedLines: skipped, invalidRows: invalid };
  } else {
    throw new Error(`unknown source kind: ${source.kind}（现在只支持 dataset）`);
  }
  const packed = packSamples(raw);
  const splits = splitSamples(packed.kept, { valRatio: options.valRatio ?? 0.08, evalRatio: options.evalRatio ?? 0.05, seed: datasetId });
  const tokenStats = splits.train.map((sample) => sampleTokens(sample));
  // No tool registry to snapshot: an imported file has no session behind it, so
  // `knownTools` is empty and evaluation cannot flag invented tools for it. That
  // was already true of every file-sourced dataset; the field stays so the
  // manifest shape does not change under existing datasets.
  const fixture = { knownTools: [], judgeContext: null };
  const manifest = {
    schema: 'dsh-fine-tune/dataset/v1',
    id: datasetId,
    sourceId: source.id,
    sourceKind: source.kind,
    sourceLabel: source.label,
    mode: source.mode,
    redaction: source.redaction,
    createdAt: Date.now(),
    counts: {
      raw: raw.length,
      kept: packed.kept.length,
      duplicates: packed.duplicates,
      invalid: packed.invalid,
      train: splits.stats.train,
      val: splits.stats.val,
      eval: splits.stats.eval,
      groups: splits.stats.groups,
    },
    tokens: { median: median(tokenStats), p90: percentile(tokenStats, 0.9), max: tokenStats.length === 0 ? 0 : Math.max(...tokenStats) },
    knownTools: fixture.knownTools,
    provenance,
  };
  writeDataset(dir, splits, manifest);
  // Snapshot the tool registry beside the data: a re-evaluation months later must
  // grade against what the model was actually taught about, not today's registry.
  writeFixtureFile(dir, fixture);
  return { dir, manifest, splits };
}

function percentile(values, p) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.floor(sorted.length * p));
  return sorted[index];
}

/** Every dataset version on disk, newest first. */
export function listDatasets(root) {
  const dir = join(root, 'datasets');
  let entries;
  try {
    entries = readdirSync(dir, { withFileTypes: true }).filter((entry) => entry.isDirectory());
  } catch {
    return [];
  }
  const out = [];
  for (const entry of entries) {
    const file = join(dir, entry.name, 'manifest.json');
    if (!existsSync(file)) continue;
    try {
      out.push(JSON.parse(readFileSync(file, 'utf8')));
    } catch { /* a half-written manifest is not a dataset */ }
  }
  return out.sort((a, b) => (b.createdAt ?? 0) - (a.createdAt ?? 0));
}



/**
 * Delete one dataset directory.
 *
 * Two guards, both because this is irreversible:
 *
 *   - the id must be a plain dataset id. It is joined onto the datasets root, so
 *     anything with a separator or `..` is refused before it can name another
 *     directory.
 *   - the directory must exist AND contain a manifest. A stray directory that
 *     happens to sit under datasets/ is not ours to remove.
 *
 * A training run copies its data into runs/<id>/data at start, so deleting a
 * dataset never breaks a run that already began.
 *
 * @returns {{ ok: boolean, id?: string, error?: string }}
 */
export function deleteDataset(root, id) {
  const name = String(id ?? '');
  if (!/^[a-z0-9][a-z0-9._-]{1,60}$/.test(name)) return { ok: false, error: '数据集 id 不合法' };
  const dir = join(root, 'datasets', name);
  if (!existsSync(join(dir, 'manifest.json'))) return { ok: false, error: `数据集不存在：${name}` };
  rmSync(dir, { recursive: true, force: true });
  return { ok: true, id: name };
}

export function writeFixtureFile(dir, fixture) {
  writeFileSync(join(dir, 'fixture.json'), `${JSON.stringify(fixture, null, 2)}\n`, 'utf8');
}

/**
 * Everything a judge needs for one side-by-side comparison, in the order the
 * panel's "copy prompt → paste answers" flow uses it. The two sides are shown
 * shuffled as A/B (with the real order kept in `key`), because a judge that knows
 * which only sees "the new one" will prefer it.
 */
export function evaluationPair(dir, { limit = 8, offset = 0, seed = 'pair' } = {}) {
  const cases = loadSplit(dir, 'eval');
  const pageSize = Math.min(20, Math.max(1, Math.trunc(Number(limit) || 8)));
  const start = Math.min(cases.length, Math.max(0, Math.trunc(Number(offset) || 0)));
  const page = cases.slice(start, start + pageSize).map((row, index) => {
    const messages = Array.isArray(row.messages) ? row.messages : [];
    const prompt = String(messages[0]?.content ?? '');
    const expected = String(messages[messages.length - 1]?.content ?? '');
    // Deterministic A/B assignment from the sample hash: stable across reloads.
    const flip = (parseInt(String(row.hash ?? '').slice(0, 2) || '0', 16) + index) % 2 === 0;
    return {
      index: start + index,
      prompt,
      expected,
      hash: row.hash ?? null,
      flip,
      targetTools: (row.meta && (row.meta.targetTools ?? (row.meta.tool ? [row.meta.tool] : []))) ?? [],
    };
  });
  return {
    cases: page,
    total: cases.length,
    start,
    pageSize,
  };
}
