/**
 * Data sources: the unit of dataset identity.
 *
 * Everything downstream (packing, training, evaluation, publishing) is keyed by a
 * source id, and the source records what it is. There is exactly one kind now:
 *
 *   dataset ? samples come from a local JSONL/JSON file the user already has.
 *
 * Mining DSH session logs used to be the other kind (`agent`). It was removed from
 * both halves: the panel no longer offers the picker, so an `agent` source could
 * not be created, and keeping a branch nothing can reach would mean untested code
 * shipping as if it worked. Existing datasets built from sessions still train ?
 * they are plain files on disk by then.
 *
 * Keeping the record in the data (not in a UI checkbox) means a training run can
 * always state what it was trained on.
 */
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

export const SOURCE_KINDS = ['dataset'];

export function sourceDir(root, id) {
  return join(root, 'sources', id);
}

export function loadSource(root, id) {
  try {
    return JSON.parse(readFileSync(join(sourceDir(root, id), 'source.json'), 'utf8'));
  } catch {
    return null;
  }
}

export function listSources(root) {
  const dir = join(root, 'sources');
  let entries;
  try {
    entries = readdirSync(dir, { withFileTypes: true }).filter((entry) => entry.isDirectory());
  } catch {
    return [];
  }
  const out = [];
  for (const entry of entries) {
    const source = loadSource(root, entry.name);
    if (source !== null) out.push(source);
  }
  return out.sort((a, b) => (b.createdAt ?? 0) - (a.createdAt ?? 0));
}

export function saveSource(root, source) {
  const dir = sourceDir(root, source.id);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'source.json'), `${JSON.stringify(source, null, 2)}\n`, 'utf8');
  return source;
}

/** Validate the fields a source needs before anything is created. */
/**
 * Coerce a mode input into a list of mode ids.
 *
 * Accepts an array (the panel), a single id, or the legacy `'both'`. An external
 * `dataset` source has no mining step, so its label is 'answer' and the list is
 * purely informational.
 */
export function normalizeModes(value, kind = 'agent') {
  const raw = Array.isArray(value) ? value : [value ?? (kind === 'agent' ? 'action' : 'answer')];
  const ids = raw
    .map((entry) => String(entry))
    .flatMap((entry) => (entry === 'both' ? ['action', 'answer'] : [entry]))
    .filter((entry) => entry !== '');
  return ids.length > 0 ? [...new Set(ids)] : [kind === 'agent' ? 'action' : 'answer'];
}

export function validateSourceInput(input) {
  const kind = String(input?.kind ?? '');
  if (!SOURCE_KINDS.includes(kind)) return { error: `kind must be one of ${SOURCE_KINDS.join(', ')}` };
  const id = String(input?.id ?? '').trim();
  if (id === '' || !/^[a-z0-9][a-z0-9._-]{1,48}$/.test(id)) return { error: 'id must be lowercase letters/digits/._- (2–49 chars)' };
  const label = String(input?.label ?? '').trim();
  if (label === '') return { error: 'label is required' };
  if (kind === 'dataset') {
    const file = String(input?.file ?? '').trim();
    if (file === '') return { error: 'dataset sources need `file`: a local path to .jsonl or .json' };
    if (!existsSync(file)) return { error: `dataset file not found: ${file}` };
  }
  return {
    value: {
      id,
      kind,
      label,
      createdAt: Date.now(),
      file: String(input.file),
      format: String(input.format ?? 'auto'),
      redaction: String(input.redaction ?? 'path'),
      // An imported file is conversation pairs, so the label shape is 'answer'.
      // Kept on the record because the data card prints it.
      mode: 'answer',
    },
  };
}
