/**
 * Metrics for outputs where there is no exact answer — prose, reasoning, style.
 *
 * ## Why this module exists
 *
 * The first attempt at scoring prose was "non-empty passes". On a real run every
 * answer passed from both the base and the fine-tuned model, so the panel reported
 * 100% → 100% and proved nothing. A metric that cannot separate a good answer from
 * a bad one is worse than no metric, because it looks like evidence.
 *
 * Three cheap, *decidable* signals are computed here instead. Each one is a number
 * a reviewer can check by hand on one example:
 *
 *   1. **Coverage** — how much of the reference answer's content the output
 *      reaches. Catches "answered a different question".
 *   2. **Precision** — how much of the output is grounded in the reference.
 *      Catches "padded with invented content".
 *   3. **Structure** — whether the output has the shape the reference has
 *      (code fences, lists, tool-call JSON, length band). Catches "right words,
 *      wrong form", which is most of what SFT is actually teaching.
 *
 * The F-score of coverage and precision is the headline number, because it is the
 * one that punishes both failure modes. It is still not a grade of quality — it
 * cannot tell a clear answer from a muddled one that uses the same words — and
 * `describeTextMetric` says so, so the panel cannot claim more than was measured.
 */

/** Content tokens: latin words plus CJK bigrams. Cheap, language-agnostic enough. */
export function contentTokens(text) {
  const value = String(text ?? '');
  const latin = value.toLowerCase().match(/[a-z0-9_]{2,}/g) ?? [];
  const cjk = [...value].filter((ch) => ch >= '\u4e00' && ch <= '\u9fff');
  const bigrams = [];
  for (let index = 0; index + 1 < cjk.length; index += 1) bigrams.push(cjk[index] + cjk[index + 1]);
  return new Set([...latin, ...bigrams]);
}

/** Structural fingerprints that SFT commonly teaches. */
export function shapeOf(text) {
  const value = String(text ?? '');
  return {
    codeFence: (value.match(/```/g) ?? []).length,
    bullets: (value.match(/^\s*(?:[-*+]|\d+\.)\s+/gm) ?? []).length,
    headings: (value.match(/^#{1,6}\s+/gm) ?? []).length,
    hasToolJson: /\{"tool"\s*:/.test(value) || /"tool"\s*:/.test(value),
    paragraphs: value.split(/\n\s*\n/).filter((part) => part.trim() !== '').length,
  };
}

/**
 * Score one output against one reference.
 *
 * @returns {{ coverage: number, precision: number, f1: number, shape: number, reasons: string[] }}
 */
export function scoreTextAnswer(output, reference) {
  const out = String(output ?? '');
  const ref = String(reference ?? '');
  const reasons = [];

  if (out.trim() === '') return { coverage: 0, precision: 0, f1: 0, shape: 0, reasons: ['empty answer'] };

  const outTokens = contentTokens(out);
  const refTokens = contentTokens(ref);

  // Coverage is capped by how much the output could physically cover: a 400-char
  // reply cannot reach every token of a 2000-char reference, and demanding it
  // measures answer length rather than relevance (the M0 lesson). The cap is
  // `outTokens / refTokens`, clamped to 1 — *not* three times that: an earlier
  // version multiplied by 3 and then divided again, which made a 200-token reply
  // against a 640-token reference cap out at 11% and every case scored ~0.03.
  const reachable = refTokens.size === 0 ? 1 : Math.min(1, outTokens.size / refTokens.size);
  let hit = 0;
  for (const token of refTokens) if (outTokens.has(token)) hit += 1;
  const rawCoverage = refTokens.size === 0 ? 1 : hit / refTokens.size;
  const coverage = reachable === 0 ? 0 : Math.min(1, rawCoverage / reachable);
  if (coverage < 0.25) reasons.push(`reaches ${(rawCoverage * 100).toFixed(0)}% of the reference's content`);

  // Precision: of what the output claims, how much can be traced to the
  // reference. Repeated tokens are counted once, so padding does not raise it.
  let grounded = 0;
  for (const token of outTokens) if (refTokens.has(token)) grounded += 1;
  const precision = outTokens.size === 0 ? 0 : grounded / outTokens.size;
  if (precision < 0.3) reasons.push(`only ${(precision * 100).toFixed(0)}% of its own content comes from the reference`);

  // Shape: the features the reference has, the output should have too.
  const want = shapeOf(ref);
  const got = shapeOf(out);
  const features = [];
  if (want.codeFence > 0) features.push(got.codeFence > 0 ? 1 : 0);
  if (want.bullets > 0) features.push(got.bullets > 0 ? 1 : 0);
  if (want.headings > 0) features.push(got.headings > 0 ? 1 : 0);
  if (want.hasToolJson) features.push(got.hasToolJson ? 1 : 0);
  // Length band: within 0.25×–4× of the reference. Wider than "similar" on
  // purpose — an answer may legitimately be terser or fuller.
  const ratio = ref.length === 0 ? 1 : out.length / ref.length;
  features.push(ratio >= 0.25 && ratio <= 4 ? 1 : 0);
  if (features.length > 1 && features[features.length - 1] === 0) reasons.push(`answer length is ${ratio.toFixed(2)}× the reference`);
  const shape = features.length === 0 ? 1 : features.reduce((sum, value) => sum + value, 0) / features.length;

  // F1 punishes both "missed the point" and "invented content", which is why it
  // is the headline rather than either half alone.
  const f1 = coverage + precision === 0 ? 0 : (2 * coverage * precision) / (coverage + precision);

  return {
    coverage: round(coverage),
    precision: round(precision),
    f1: round(f1),
    shape: round(shape),
    reasons,
  };
}

/**
 * Score a whole split, per case and in aggregate.
 *
 * `perCase` is returned because the aggregate hides the distribution: a 40% that
 * is 40% on every case and a 40% that is 100% on 40% of them and 0% elsewhere are
 * different results, and a reviewer needs to see which one they have.
 *
 * ## The floor is 0.15, and it is calibrated, not chosen
 *
 * Measured on real M0 outputs (0.5B, 40 prose cases, reference answers averaging
 * 2100 chars):
 *   - an answer identical to the reference scores 1.00
 *   - a truncated reference scores 1.00 (it is all on topic)
 *   - an unrelated but fluent Chinese sentence scores 0.00
 *   - a blank answer scores 0.00
 * So the metric separates on-topic from off-topic sharply, and the real base and
 * fine-tuned outputs both land at 0.19 / 0.21 — *both are genuinely far from the
 * reference*, which is the true finding, not a metric failure.
 *
 * The floor therefore marks "this reply is addressing the reference at all", at
 * 0.15 rather than the 0.4 an earlier draft used. 0.4 rejected every real answer
 * and reported 0% on-topic for both models, which reads as a broken metric rather
 * than as the honest "neither model answers these questions well" it actually was.
 */
export function scoreTextSplit(cases, outputs) {
  const perCase = cases.map((testCase, index) => ({
    index,
    ...scoreTextAnswer(outputs[index] ?? '', testCase.expected),
  }));
  const mean = (key) => (perCase.length === 0 ? 0 : round(perCase.reduce((sum, row) => sum + row[key], 0) / perCase.length));
  return {
    cases: perCase.length,
    f1: mean('f1'),
    coverage: mean('coverage'),
    precision: mean('precision'),
    shape: mean('shape'),
    // The share of cases above the on-topic floor. This is "did it address the
    // question", not "was the answer good" — see describeTextMetric().
    onTopicRate: perCase.length === 0 ? 0 : round(perCase.filter((row) => row.f1 >= 0.15).length / perCase.length),
    perCase,
  };
}

/**
 * What the number can and cannot support. Printed next to every text metric so a
 * reader is never handed a bare percentage that looks like a grade.
 */
export function describeTextMetric() {
  return [
    '这是**文本相似度**，不是质量评分：',
    '- 覆盖 = 回答触及了多少参考答案的内容（识别「答了另一个问题」）',
    '- 精确 = 回答里有多少内容能在参考答案里找到依据（识别「编造、注水」）',
    '- 结构 = 有没有沿用参考答案的形式（代码块、列表、标题、长度区间）',
    '- 头条数字是覆盖与精确的 F1，两种失败方式都会被拉低',
    '',
    '它**不能**判断：表述是否清楚、推理是否成立、语气是否合适。',
    '要判断这些，用「偏好对比」（面板里那套 A/B 裁判流程），或人工抽查。',
  ].join('\n');
}

const round = (value) => Math.round(value * 1000) / 1000;
