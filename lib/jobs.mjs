/**
 * Job store: the durable half of a training run.
 *
 * A training run outlives the page that started it, and on a laptop it often
 * outlives DSH itself. So the truth about a run is not in memory — it is
 * `runs/<jobId>/run.json` plus an append-only `events.jsonl`. The memory copy is
 * a cache for the UI. On mount every non-terminal run is reconciled against disk:
 * a run whose process is not ours is marked `interrupted` with the raw reason,
 * never silently reported as running.
 *
 * The state machine is deliberately small:
 *
 *   queued ──► running ──► succeeded
 *                 │
 *                 ├──► failed
 *                 ├──► cancelled
 *                 └──► interrupted   (host restarted / process vanished)
 *
 * `queued` and `running` are the only non-terminal states; that is what makes
 * "what should I resume?" answerable with one filter.
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';

export const TERMINAL = new Set(['succeeded', 'failed', 'cancelled', 'interrupted']);
export const ACTIVE = new Set(['queued', 'running']);

const MAX_EVENTS_IN_MEMORY = 4000;

export function createJobStore({ root, log = () => {} }) {
  const runsDir = join(root, 'runs');
  mkdirSync(runsDir, { recursive: true });
  /** @type {Map<string, {record: object, events: object[], listeners: Set<Function>, handle: any}>} */
  const jobs = new Map();

  function dirOf(id) {
    return join(runsDir, id);
  }

  function recordFile(id) {
    return join(dirOf(id), 'run.json');
  }

  function eventsFile(id) {
    return join(dirOf(id), 'events.jsonl');
  }

  function persist(job) {
    writeFileSync(recordFile(job.record.id), `${JSON.stringify(job.record, null, 2)}\n`, 'utf8');
  }

  function emit(job, event) {
    const entry = { seq: (job.events[job.events.length - 1]?.seq ?? 0) + 1, time: Date.now(), ...event };
    job.events.push(entry);
    if (job.events.length > MAX_EVENTS_IN_MEMORY) job.events.splice(0, job.events.length - MAX_EVENTS_IN_MEMORY);
    try {
      appendFileSync(eventsFile(job.record.id), `${JSON.stringify(entry)}\n`, 'utf8');
    } catch (error) {
      log(`fine-tune: could not append an event for ${job.record.id}: ${error?.message ?? error}`);
    }
    for (const listener of job.listeners) {
      try {
        listener(entry);
      } catch { /* a dead listener must not stop the run */ }
    }
    return entry;
  }

  /** Load every run on disk; reconcile the ones that claim to be active. */
  function restore() {
    let dirs = [];
    try {
      dirs = readdirSync(runsDir, { withFileTypes: true }).filter((entry) => entry.isDirectory()).map((entry) => entry.name);
    } catch {
      return { restored: 0, interrupted: 0 };
    }
    let restored = 0;
    let interrupted = 0;
    for (const id of dirs) {
      let record;
      try {
        record = JSON.parse(readFileSync(recordFile(id), 'utf8'));
      } catch {
        continue;
      }
      const events = [];
      try {
        for (const line of readFileSync(eventsFile(id), 'utf8').split('\n')) {
          if (line.trim() === '') continue;
          try {
            events.push(JSON.parse(line));
          } catch { /* skip a torn final line */ }
        }
      } catch { /* a run with no events yet */ }
      const job = { record, events: events.slice(-MAX_EVENTS_IN_MEMORY), listeners: new Set(), handle: null };
      if (ACTIVE.has(record.state)) {
        // We just started: no child process of ours owns this run any more.
        record.state = 'interrupted';
        record.finishedAt = record.finishedAt ?? Date.now();
        record.error = { code: 'host-restarted', message: 'DSH 重启时这个任务还在运行；它的进程已经不属于当前宿主。' };
        persist(job);
        emit(job, { kind: 'interrupted', code: 'host-restarted' });
        interrupted += 1;
      }
      jobs.set(record.id, job);
      restored += 1;
    }
    return { restored, interrupted };
  }

  function create({ kind, sourceId = null, datasetId = null, backend = 'local', target = null, input = {} }) {
    const id = `${kind}-${new Date().toISOString().replace(/[^0-9]/g, '').slice(0, 14)}-${randomUUID().slice(0, 6)}`;
    mkdirSync(dirOf(id), { recursive: true });
    const record = {
      id,
      kind,
      state: 'queued',
      createdAt: Date.now(),
      startedAt: null,
      finishedAt: null,
      sourceId,
      datasetId,
      backend,
      target,
      input,
      pid: null,
      exitCode: null,
      error: null,
      metrics: {},
      artifacts: [],
    };
    const job = { record, events: [], listeners: new Set(), handle: null };
    jobs.set(id, job);
    persist(job);
    emit(job, { kind: 'created', backend, sourceId, datasetId });
    return id;
  }

  function get(id) {
    return jobs.get(id)?.record ?? null;
  }

  function job(id) {
    return jobs.get(id) ?? null;
  }

  function list({ kind, limit = 50 } = {}) {
    const all = [...jobs.values()].map((entry) => entry.record);
    const filtered = kind === undefined ? all : all.filter((record) => record.kind === kind);
    return filtered.sort((a, b) => b.createdAt - a.createdAt).slice(0, limit);
  }

  function move(id, state, patch = {}) {
    const target = jobs.get(id);
    if (target === undefined) throw new Error(`unknown job: ${id}`);
    if (TERMINAL.has(target.record.state)) return target.record;
    const now = Date.now();
    target.record.state = state;
    if (state === 'running' && target.record.startedAt === null) target.record.startedAt = now;
    if (state !== 'running' && target.record.startedAt === null) target.record.startedAt = now;
    if (TERMINAL.has(state)) target.record.finishedAt = now;
    Object.assign(target.record, patch);
    persist(target);
    emit(target, { kind: 'state', state, ...patch });
    return target.record;
  }

  /** Append a progress event (a parsed worker line, a metric tick). */
  function progress(id, event) {
    const target = jobs.get(id);
    if (target === undefined) return null;
    if (event?.kind === 'metric' && event.values !== undefined) {
      Object.assign(target.record.metrics, event.values);
      persist(target);
    }
    if (event?.kind === 'artifact' && typeof event.path === 'string' && !target.record.artifacts.includes(event.path)) {
      target.record.artifacts.push(event.path);
      persist(target);
    }
    return emit(target, event);
  }

  function attach(id, handle) {
    const target = jobs.get(id);
    if (target === undefined) return;
    target.handle = handle;
    target.record.pid = handle?.pid ?? null;
    persist(target);
  }

  function subscribe(id, listener) {
    const target = jobs.get(id);
    if (target === undefined) return () => {};
    target.listeners.add(listener);
    return () => target.listeners.delete(listener);
  }

  /** One SSE-ready snapshot: the record, recent events, and a cursor. */
  function snapshot(id, since = 0) {
    const target = jobs.get(id);
    if (target === undefined) return null;
    const events = target.events.filter((event) => event.seq > since);
    return {
      record: target.record,
      events,
      cursor: target.events[target.events.length - 1]?.seq ?? since,
    };
  }

  return { root, runsDir, restore, create, get, job, list, move, progress, attach, subscribe, snapshot, emit, recordFile, eventsFile };
}
