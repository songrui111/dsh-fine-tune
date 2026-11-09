/**
 * End-to-end test against the RUNNING DSH host, over HTTP.
 *
 * Everything else in this repo tests modules directly; this exercises the same
 * path the browser takes - same route, same headers, same same-origin gate - so a
 * mismatch between the client's request shape and the host's expectations shows up
 * here rather than in the panel.
 *
 * Phases: state -> env -> data -> gpu -> train -> eval -> publish
 */

const BASE = process.argv[2] ?? 'http://127.0.0.1:9029';
const ORIGIN = BASE;

const results = [];
function record(phase, ok, detail) {
  results.push({ phase, ok });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${phase.padEnd(8)} ${detail}`);
}

/** POST to the action route with the headers a browser on this origin sends. */
async function act(body) {
  const response = await fetch(`${BASE}/fine-tune/api/action`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      origin: ORIGIN,
      host: new URL(BASE).host,
    },
    body: JSON.stringify(body),
  });
  let payload = null;
  try { payload = await response.json(); } catch { payload = null; }
  return { status: response.status, payload };
}

async function getState(extra = '') {
  const response = await fetch(`${BASE}/fine-tune/api/state${extra}`, {
    headers: { origin: ORIGIN },
  });
  try { return { status: response.status, payload: await response.json() }; }
  catch { return { status: response.status, payload: null }; }
}

// ---------------------------------------------------------------- state
const state = await getState();
const settings = state.payload?.settings ?? {};
record('state', state.status === 200,
  `status=${state.status} datasets=${(state.payload?.datasets ?? []).length} runs=${(state.payload?.runs ?? []).length}`);

// ---------------------------------------------------------------- permissions
record('perm', settings.allowSpawn === true && settings.allowPublish === true,
  `allowSpawn=${settings.allowSpawn} allowPublish=${settings.allowPublish}`);

// ---------------------------------------------------------------- env
const env = await act({ op: 'env.status' });
record('env', env.status === 200,
  `status=${env.status} venvExists=${env.payload?.venvExists} ready=${env.payload?.ready === null ? 'no' : 'yes'}`);

// ---------------------------------------------------------------- data
const sample = await act({ op: 'dataset.sample' });
const counts = sample.payload?.dataset?.counts;
record('data', sample.status === 200 && counts?.raw === 60,
  counts ? `raw=${counts.raw} train=${counts.train} val=${counts.val} eval=${counts.eval}` : `status=${sample.status}`);

// ---------------------------------------------------------------- gpu
const gpu = await act({ op: 'gpu', modelId: 'Qwen/Qwen2.5-0.5B-Instruct', mode: 'qlora', maxSeqLen: 512 });
const card = gpu.payload?.gpu?.gpus?.[0];
record('gpu', gpu.status === 200 && card !== undefined,
  `${card?.name ?? 'none'} ${card?.totalGiB ?? '?'} GiB, fits=${JSON.stringify(gpu.payload?.estimate?.fits ?? 'n/a')}`);

// ---------------------------------------------------------------- train
const started = await act({
  op: 'run.start',
  datasetIds: ['sample'],
  modelId: 'Qwen/Qwen2.5-0.5B-Instruct',
  mode: 'qlora',
  maxSeqLen: 512,
  epochs: 1,
  maxSteps: 6,
});
const runId = started.payload?.run?.id ?? null;
record('train', started.status === 200 && runId !== null,
  runId ?? `status=${started.status} ${JSON.stringify(started.payload?.error ?? '').slice(0, 90)}`);

let finished = null;
if (runId !== null) {
  for (let i = 0; i < 600; i += 1) {
    await new Promise((r) => setTimeout(r, 2000));
    const s = await getState();
    const run = (s.payload?.runs ?? []).find((r) => r.id === runId) ?? null;
    if (run !== null && ['succeeded', 'failed', 'cancelled', 'interrupted'].includes(run.state)) {
      finished = run;
      break;
    }
    if (i % 20 === 0) {
      const m = run?.metrics ?? {};
      console.log(`      … ${i * 2}s loss=${typeof m.loss === 'number' ? m.loss.toFixed(4) : '-'}`);
    }
  }
  const state2 = finished?.state ?? 'timeout';
  const metrics = finished?.metrics ?? {};
  record('train-done', state2 === 'succeeded',
    `state=${state2} loss=${typeof metrics.loss === 'number' ? metrics.loss.toFixed(4) : 'n/a'} eval_loss=${typeof metrics.eval_loss === 'number' ? metrics.eval_loss.toFixed(4) : 'n/a'} peak=${metrics.peakGiB ?? '?'}GiB`);

  // ------------------------------------------------------------- eval
  const pair = await act({ op: 'evaluation.pair', datasetId: 'sample', limit: 3 });
  const cases = pair.payload?.pair?.cases ?? pair.payload?.cases ?? [];
  record('eval', pair.status === 200 && cases.length > 0,
    `status=${pair.status} cases=${cases.length}`);

  // ------------------------------------------------------------- publish
  const publish = await act({ op: 'publish.prepare', id: runId });
  const reason = publish.payload?.error ?? publish.payload?.reason ?? '';
  // 409 is the correct answer here: publishing requires a green evaluation, and a
  // 6-step run has no verdict yet. What matters is that the route answered rather
  // than refusing on the permission gate.
  const gatePassed = publish.status !== 403;
  record('publish', gatePassed,
    `status=${publish.status} ${gatePassed ? '(权限已通过，被评测门拦下=正常)' : String(reason).slice(0, 80)}`);
}

console.log();
const passed = results.filter((r) => r.ok).length;
console.log(`${passed}/${results.length} phases passed`);
process.exit(passed === results.length ? 0 : 1);
