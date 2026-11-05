/**
 * Deterministic packing for a dataset: validation, SMILES-style dedupe, hashing,
 * and the on-disk shape the Python worker reads.
 *
 * "SMILES dedupe" here means what the chemoinformatics name suggests: normalise
 * an item to a canonical form, then keep the first item that maps to each form.
 * For text that canonical form is the case/whitespace/punctuation-insensitive
 * token stream, so "你好！" and "你好" and "你好!!!" collapse to one example —
 * which is exactly the duplicate class a chat transcript produces in bulk.
 */
import { createHash } from 'node:crypto';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const HASH_LENGTH = 16;

/** Canonical key for duplicate detection. */
export function canonicalKey(sample) {
  const text = (sample?.messages ?? [])
    .map((message) => (typeof message?.content === 'string' ? message.content : JSON.stringify(message?.content ?? '')))
    .join('\u0000');
  return text
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .replace(/[\u3000\uff01-\uff5e]/g, (ch) => String.fromCharCode(ch.charCodeAt(0) - 0xfee0))
    .replace(/[\p{P}\p{S}]+/gu, ' ')
    .trim();
}

export function sampleHash(sample) {
  const content = JSON.stringify(sample?.messages ?? []);
  return createHash('sha256').update(content).digest('hex').slice(0, HASH_LENGTH);
}

/**
 * Validate and dedupe.
 *
 * @returns {{
 *   kept: Array<object>, duplicates: number, invalid: number,
 *   byHash: Map<string, string>, reason?: string
 * }}
 */
export function packSamples(samples) {
  const kept = [];
  const seen = new Set();
  const byHash = new Map();
  let duplicates = 0;
  let invalid = 0;
  for (const sample of samples ?? []) {
    const messages = sample?.messages;
    if (!Array.isArray(messages) || messages.length < 2) {
      invalid += 1;
      continue;
    }
    const [first] = messages;
    const last = messages[messages.length - 1];
    const firstText = first?.content;
    const lastText = last?.content;
    if (first?.role !== 'user' || last?.role !== 'assistant') {
      invalid += 1;
      continue;
    }
    if (typeof firstText !== 'string' || typeof lastText !== 'string') {
      invalid += 1;
      continue;
    }
    if (firstText.trim() === '' || lastText.trim() === '') {
      invalid += 1;
      continue;
    }
    const key = canonicalKey(sample);
    if (key === '' || seen.has(key)) {
      duplicates += 1;
      continue;
    }
    seen.add(key);
    const hash = sampleHash(sample);
    byHash.set(hash, String(sample?.meta?.sessionId ?? ''));
    kept.push({ ...sample, hash });
  }
  return { kept, duplicates, invalid, byHash };
}

/** Count bytes-ish tokens: Chinese counts 1 char ≈ 1 token, latin ≈ 4 chars/token. */
export function estimateTokens(text) {
  const value = String(text ?? '');
  let cjk = 0;
  for (let i = 0; i < value.length; i += 1) {
    const code = value.charCodeAt(i);
    if (code >= 0x4e00 && code <= 0x9fff) cjk += 1;
    else if (code >= 0x3040 && code <= 0x30ff) cjk += 1;
  }
  const latin = value.length - cjk;
  return cjk + Math.ceil(latin / 4);
}

export function sampleTokens(sample) {
  const text = (sample?.messages ?? []).map((message) => message?.content ?? '').join('\n');
  return estimateTokens(text);
}

/**
 * Write one dataset version to disk in the shape the Python worker consumes.
 *
 * Layout:
 *   <dir>/manifest.json
 *   <dir>/train.jsonl
 *   <dir>/val.jsonl
 *   <dir>/eval.jsonl
 */
export function writeDataset(dir, splits, manifest) {
  mkdirSync(dir, { recursive: true });
  const writeJsonl = (file, rows) => {
    // `hash` is written into each row: an exclusion removes a sample by hash, and
    // every reader (preview, training, evaluation) must agree on that value.
    const body = rows.map((row) => JSON.stringify({ messages: row.messages, meta: row.meta ?? {}, hash: row.hash ?? sampleHash(row) })).join('\n');
    writeFileSync(file, body === '' ? '' : `${body}\n`, 'utf8');
  };
  writeJsonl(join(dir, 'train.jsonl'), splits.train);
  writeJsonl(join(dir, 'val.jsonl'), splits.val);
  writeJsonl(join(dir, 'eval.jsonl'), splits.eval);
  writeFileSync(join(dir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
  return dir;
}

/** Median of a numeric list (0 for an empty list) — robust to a few long outliers. */
export function median(values) {
  if (!Array.isArray(values) || values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? Math.round((sorted[middle - 1] + sorted[middle]) / 2) : sorted[middle];
}

/**
 * Split samples into train/val/eval.
 *
 * Group-aware by default: everything mined from one session shares a `group`, and
 * the split never breaks a group across sides. That is what stops a model from
 * being validated on sentences it was trained on — mined samples from one
 * conversation are near-duplicates of each other.
 *
 * **External datasets have one group and need the opposite treatment.** Every row
 * of a user-supplied JSONL carries the same group id, so a pure group split puts
 * the whole file in `val` and leaves `train` empty — the dataset builds, reports
 * "3 samples", and cannot be trained on. When there are fewer groups than the
 * split needs, this falls back to per-sample assignment so a domain-knowledge file
 * is actually usable. The fallback is reported in `stats.strategy` rather than
 * being invisible: the caller should know its validation set shares a source with
 * its training set.
 */
export function splitSamples(samples, options = {}) {
  const valRatio = clamp01(options.valRatio ?? 0.08);
  const evalRatio = clamp01(options.evalRatio ?? 0.05);
  const seed = String(options.seed ?? 'dsh-fine-tune');
  const byGroup = new Map();
  for (const sample of samples) {
    const group = String(sample?.meta?.group ?? sample?.meta?.sessionId ?? 'ungrouped');
    if (!byGroup.has(group)) byGroup.set(group, []);
    byGroup.get(group).push(sample);
  }
  const groups = [...byGroup.entries()].sort((a, b) => (hash32(`${seed}:${a[0]}`) - hash32(`${seed}:${b[0]}`)) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  const total = samples.length;
  // Guarantee non-empty validation when there is anything worth validating on.
  const wantVal = Math.max(valRatio > 0 && total >= 3 ? 1 : 0, Math.round(total * valRatio));
  const wantEval = Math.max(evalRatio > 0 && total >= 12 ? 1 : 0, Math.round(total * evalRatio));

  // Group splitting needs one group per side it wants to fill, **plus one left for
  // training**. Without that `+ 1`, two groups exactly fill `eval` and `val` and
  // `train` ends up empty — the `chain` mode produced 624 samples spread over two
  // sessions and every one of them landed outside the training set. Below this
  // threshold the per-sample fallback runs instead of starving `train`.
  const sidesWanted = (wantEval > 0 ? 1 : 0) + (wantVal > 0 ? 1 : 0);
  const minimumGroups = sidesWanted + 1;

  if (groups.length < minimumGroups) {
    // Deterministic per-sample order: same seed, same split, every time.
    const ordered = [...samples].sort((a, b) => hash32(`${seed}:${a?.meta?.hash ?? a?.messages?.[0]?.content ?? ''}`)
      - hash32(`${seed}:${b?.meta?.hash ?? b?.messages?.[0]?.content ?? ''}`));
    const evalCount = Math.min(wantEval, Math.max(0, ordered.length - 1));
    const valCount = Math.min(wantVal, Math.max(0, ordered.length - evalCount - 1));
    const splits = {
      eval: ordered.slice(0, evalCount),
      val: ordered.slice(evalCount, evalCount + valCount),
      train: ordered.slice(evalCount + valCount),
    };
    return {
      ...splits,
      stats: {
        groups: groups.length,
        train: splits.train.length,
        val: splits.val.length,
        eval: splits.eval.length,
        strategy: 'sample',
      },
    };
  }

  const splits = { train: [], val: [], eval: [] };
  for (const [, bucket] of groups) {
    let target = 'train';
    if (splits.eval.length < wantEval) target = 'eval';
    else if (splits.val.length < wantVal) target = 'val';
    splits[target].push(...bucket);
  }
  return {
    ...splits,
    stats: { groups: groups.length, train: splits.train.length, val: splits.val.length, eval: splits.eval.length, strategy: 'group' },
  };
}

/** Clamp to [0, 1]; used by the split ratios. */
const clamp01 = (value) => {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.min(1, Math.max(0, number));
};

/** Deterministic grouping helpers (FNV-1a; no dependency). */
export function hash32(text) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}
