/**
 * Evaluation: judge a fine-tuned model the way its actual use will judge it.
 *
 * There is no ground truth for "is this answer good", so this module refuses the
 * temptation to score text quality. It measures two things that *are* decidable:
 *
 *  1. **Adherence** — per eval sample, does the output do what the sample's
 *     task asks? For an ACTION sample the task is a tool call, so the check is
 *     "does the output name a real tool (and the expected arguments when the
 *     sample has them)" — the same convention a grader would use, computed
 *     locally. This needs no model call and is the regression gate.
 *  2. **Preference** — an A/B on the same eval set against the base model, using
 *     whatever judge the user configured. Without a judge this step reports
 *     `skipped` with the reason, because a fake "win rate" is worse than none.
 *
 * A candidate is `green` only when it does not regress: adherence must not drop
 * by more than the tolerance *below the base model*, and no category may
 * collapse. Passing costs nothing; a regression blocks the publish button.
 */
import { canonicalKey } from './pack.mjs';

/** Which tools a candidate answer claims to call, parsed from JSON or a code fence. */
export function toolCallsIn(text) {
  const value = String(text ?? '');
  const calls = [];
  const consider = (payload) => {
    if (payload === null || typeof payload !== 'object') return;
    if (typeof payload.tool === 'string') calls.push({ tool: payload.tool, arguments: payload.arguments });
    if (Array.isArray(payload.tool_calls)) {
      for (const item of payload.tool_calls) {
        const name = item?.function?.name ?? item?.tool ?? item?.name;
        const args = item?.function?.arguments ?? item?.arguments;
        if (typeof name === 'string') calls.push({ tool: name, arguments: typeof args === 'string' ? safeParse(args) : args });
      }
    }
  };
  const fenced = [...value.matchAll(/```(?:json)?\s*\n([\s\S]*?)```/g)].map((match) => match[1].trim());
  for (const block of fenced) {
    try {
      consider(JSON.parse(block));
    } catch { /* not JSON; fall through to the bare parse */ }
  }
  if (calls.length === 0) {
    const start = value.indexOf('{');
    const end = value.lastIndexOf('}');
    if (start !== -1 && end > start) {
      try {
        consider(JSON.parse(value.slice(start, end + 1)));
      } catch { /* plain text answer */ }
    }
  }
  return calls;
}

function safeParse(text) {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

/** Tools a sample's prompt offers: names mentioned in the sample's own context. */
function offeredTools(promptText, knownTools) {
  if (!Array.isArray(knownTools) || knownTools.length === 0) return null;
  return new Set(knownTools.filter((name) => String(promptText ?? '').includes(name)));
}

/** Does `args` agree with `expected` on every key the expected side specifies? */
function argsMatch(expected, actual) {
  if (expected === undefined || expected === null) return true;
  if (actual === undefined || actual === null || typeof expected !== 'object') return false;
  for (const [key, want] of Object.entries(expected)) {
    const got = actual[key];
    if (typeof want === 'string') {
      if (String(got ?? '') !== want) return false;
    } else if (typeof want === 'number' || typeof want === 'boolean') {
      if (got !== want) return false;
    } else if (typeof want === 'object') {
      if (!argsMatch(want, got)) return false;
    }
  }
  return true;
}

/**
 * Adherence over one eval split.
 *
 * @param {Array<{prompt: string, expected: string}>} cases
 * @param {Array<string>} outputs One output per case (same order).
 * @param {object} [options]
 * @param {string[]} [options.knownTools] The tools the session offered, for the
 *   "invented a tool" check.
 * @returns {{ score: number, total: number, pass: number, categories: object, failures: Array }}
 */
export function scoreAdherence(cases, outputs, options = {}) {
  const known = options.knownTools;
  let pass = 0;
  const categories = { tool_name: { pass: 0, total: 0 }, arguments: { pass: 0, total: 0 }, format: { pass: 0, total: 0 } };
  const failures = [];
  cases.forEach((testCase, index) => {
    const output = outputs[index] ?? '';
    const calls = toolCallsIn(output);
    const expected = toolCallsIn(testCase.expected);
    const want = expected[0] ?? null;
    let ok = true;
    let reason = null;
    if (want !== null) {
      categories.format.total += 1;
      if (calls.length === 0) {
        ok = false;
        reason = 'no tool call in the answer';
      } else {
        categories.format.pass += 1;
      }
      if (ok) {
        categories.tool_name.total += 1;
        const hit = calls.find((call) => call.tool === want.tool);
        if (hit === undefined) {
          ok = false;
          reason = `called ${calls.map((call) => call.tool).join(', ') || 'nothing'}, expected ${want.tool}`;
        } else {
          categories.tool_name.pass += 1;
          categories.arguments.total += 1;
          if (argsMatch(want.arguments, hit.arguments)) categories.arguments.pass += 1;
          else {
            ok = false;
            reason = `arguments for ${want.tool} differ`;
          }
        }
      }
      // An invented tool is a failure even when the expected one was also named:
      // publishing a model that hallucinates tool names is the exact thing this
      // plugin exists to prevent.
      if (known !== null) {
        const offered = offeredTools(testCase.prompt, known);
        const invented = calls.filter((call) => offered !== null && !offered.has(call.tool));
        if (invented.length > 0) {
          ok = false;
          reason = reason ?? `invented tool ${invented.map((call) => call.tool).join(', ')}`;
        }
      }
    } else if (String(output).trim() === '') {
      ok = false;
      reason = 'empty answer';
    }
    if (ok) pass += 1;
    else if (failures.length < 20) failures.push({ index, reason, expected: testCase.expected.slice(0, 400), got: String(output).slice(0, 400) });
  });
  const total = cases.length;
  return {
    score: total === 0 ? 0 : Math.round((pass / total) * 1000) / 1000,
    total,
    pass,
    categories: Object.fromEntries(Object.entries(categories).map(([key, value]) => [key, value.total === 0 ? null : Math.round((value.pass / value.total) * 1000) / 1000])),
    failures,
  };
}

/**
 * The publish gate.
 *
 * @param {object} input
 * @param {{score:number, total:number, categories:object}} input.candidate
 * @param {{score:number, total:number, categories:object}} input.base
 * @param {number} [input.tolerance] Allowed absolute drop below the base score.
 */
export function verdictOf({ candidate, base, tolerance = 0.03 }) {
  const reasons = [];
  if (candidate.total === 0) return { status: 'blocked', reasons: ['评测集是空的：没有可以判定的样本。'] };
  const delta = Math.round((candidate.score - base.score) * 1000) / 1000;
  if (delta < -tolerance) reasons.push(`贴合度比底座低 ${Math.abs(delta * 100).toFixed(1)} 个百分点（容差 ${(tolerance * 100).toFixed(0)}）。`);
  for (const [key, value] of Object.entries(candidate.categories ?? {})) {
    const before = base.categories?.[key];
    if (typeof value !== 'number' || typeof before !== 'number') continue;
    if (value < before - tolerance) reasons.push(`「${key}」从 ${(before * 100).toFixed(0)}% 掉到 ${(value * 100).toFixed(0)}%。`);
  }
  return { status: reasons.length === 0 ? 'green' : 'blocked', delta, reasons };
}

/** A compact Markdown report — what the panel and the tool both render. */
export function evaluationMarkdown(result) {
  const lines = [`## 评测：${result.model}`, ''];
  lines.push(`| 指标 | 底座 | 微调后 | 变化 |`);
  lines.push('|---|---|---|---|');
  lines.push(`| 贴合度 | ${percent(result.base?.score)} | ${percent(result.candidate?.score)} | ${signed(result.verdict?.delta)} |`);
  for (const key of Object.keys(result.candidate?.categories ?? {})) {
    const before = result.base?.categories?.[key];
    const after = result.candidate?.categories?.[key];
    if (before === null && after === null) continue;
    lines.push(`| ${key} | ${percent(before)} | ${percent(after)} | ${signed(typeof before === 'number' && typeof after === 'number' ? after - before : null)} |`);
  }
  lines.push('');
  lines.push(`**结论：${result.verdict?.status === 'green' ? '可以发布' : '不建议发布'}**`);
  for (const reason of result.verdict?.reasons ?? []) lines.push(`- ${reason}`);
  if (result.preference?.status === 'skipped') lines.push(`\n偏好对比：未运行（${result.preference.reason}）`);
  if (result.preference?.status === 'ok') lines.push(`\n偏好对比：${result.preference.candidateWins} 胜 / ${result.preference.baseWins} 负 / ${result.preference.ties} 平（${result.preference.judge}）`);
  const failures = result.candidate?.failures ?? [];
  if (failures.length > 0) {
    lines.push('', '### 失败样例（最多 20 条）');
    for (const failure of failures) lines.push(`- #${failure.index} — ${failure.reason ?? '未通过'}`);
  }
  return lines.join('\n');
}

const percent = (value) => (typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—');
const signed = (value) => (typeof value !== 'number' ? '—' : `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)} 个百分点`);

export { canonicalKey };
