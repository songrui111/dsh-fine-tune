/**
 * Hardware and memory budgeting.
 *
 * The single most expensive failure mode of a local trainer is OOM ten minutes
 * into a run: the user has waited, the GPU is pinned, and the log says
 * "CUDA out of memory" with nothing actionable. So the estimate is a gate, not a
 * suggestion: it runs before the process starts, reports every component by name,
 * and refuses a plan it can prove will not fit.
 *
 * The coefficients are coarse, deliberately: they answer "will this fit" to
 * within ~20%, which is the decision being made. The measured M0 numbers live in
 * M0-LAB.md and replace these defaults once someone runs the lab on real hardware.
 */

/**
 * Known model shapes, keyed by Hugging Face repo id.
 *
 * `vocab` is not decoration: the LM head output is upcast to fp32 for the loss, so
 * the largest tensor a forward pass holds is vocab × seq × 4 bytes. The first
 * version of this table omitted it and the estimator was off by 20×.
 */
export const MODELS = {
  'Qwen/Qwen2.5-0.5B-Instruct': { paramsB: 0.5, hidden: 896, layers: 24, heads: 14, kvHeads: 2, maxLen: 32768, vocab: 151936 },
  'Qwen/Qwen2.5-1.5B-Instruct': { paramsB: 1.5, hidden: 1536, layers: 28, heads: 12, kvHeads: 2, maxLen: 32768, vocab: 151936 },
  'Qwen/Qwen2.5-3B-Instruct': { paramsB: 3.1, hidden: 2048, layers: 36, heads: 16, kvHeads: 2, maxLen: 32768, vocab: 151936 },
  'Qwen/Qwen2.5-7B-Instruct': { paramsB: 7.6, hidden: 3584, layers: 28, heads: 28, kvHeads: 4, maxLen: 32768, vocab: 152064 },
  'Qwen/Qwen3-1.7B': { paramsB: 1.7, hidden: 2048, layers: 28, heads: 16, kvHeads: 8, maxLen: 32768, vocab: 151936 },
  'Qwen/Qwen3-4B-Instruct-2507': { paramsB: 4.0, hidden: 2560, layers: 36, heads: 32, kvHeads: 8, maxLen: 262144, vocab: 151936 },
  'Qwen/Qwen3-8B': { paramsB: 8.2, hidden: 4096, layers: 36, heads: 32, kvHeads: 8, maxLen: 131072, vocab: 151936 },
  'google/gemma-3-4b-it': { paramsB: 4.3, hidden: 2560, layers: 34, heads: 8, kvHeads: 4, maxLen: 131072, vocab: 262144 },
  'meta-llama/Llama-3.2-3B-Instruct': { paramsB: 3.2, hidden: 3072, layers: 28, heads: 24, kvHeads: 8, maxLen: 131072, vocab: 128256 },
};

/** Guess a shape from a repo id when it is not in the table (name-based, honest). */
export function shapeOf(modelId) {
  if (MODELS[modelId] !== undefined) return { ...MODELS[modelId], known: true };
  const match = /(\d+(?:\.\d+)?)\s*B/i.exec(String(modelId ?? ''));
  if (match !== null) {
    const paramsB = Number(match[1]);
    // Linear interpolation between the smallest and largest known shape; flagged
    // `known: false` so the caller can say the estimate is a guess. The vocab
    // default is the Qwen-class 152k — the larger side of common, on purpose:
    // under-predicting the logits is the mistake that costs an OOM.
    return { paramsB, hidden: Math.round(1536 + (paramsB - 1.5) * 313), layers: 32, heads: 16, kvHeads: 4, maxLen: 32768, vocab: 152064, known: false };
  }
  return { paramsB: 4, hidden: 2560, layers: 32, heads: 16, kvHeads: 4, maxLen: 32768, vocab: 152064, known: false };
}

const GiB = 1024 ** 3;

/**
 * Measured coefficients, written by `fnt_worker calibrate` on this machine.
 *
 * This file is Data, not code: the lab copies `calibration.json` here once and
 * every estimate afterwards is calibrated instead of guessed. An empty table is
 * the honest default — it means nobody has measured anything yet, and the
 * estimate falls back to the coarse coefficients below.
 *
 * Shape:  <modelId> → { <mode> → { a, r, seqLen, device, torch, at } }
 * `a` is bytes per (token × hidden unit) per layer, measured *net of logits*
 * (logits are their own component now).
 *
 * Measured 2026-11-04, RTX 3060 Laptop 6 GB, torch 2.5.1+cu121, transformers
 * 4.46.3, peft 0.13.2, bnb 0.44.1 — two points that agree to 0.7%:
 *   seq 512  → 3.16 GiB total
 *   seq 1024 → 5.58 GiB total
 * Those totals are reproduced by the default coefficients below to within
 * +10..17% (deliberately on the safe side). An entry here overrides them for the
 * exact model/mode it names.
 */
export const MEASURED = {};

/**
 * The activation coefficient for one (model, mode), preferring a measurement.
 *
 * `a` is bytes per (token × hidden unit) per layer; `r` is the fixed reserve.
 * A measurement for a *different* mode of the same model is still a better guess
 * than the untuned constant, so it is used but flagged.
 */
export function coefficientsFor(modelId, mode) {
  const perModel = MEASURED[modelId];
  if (perModel === undefined) return { a: null, r: null, source: 'default' };
  const exact = perModel[mode];
  if (exact !== undefined && typeof exact.a === 'number') return { ...exact, source: 'measured' };
  const other = Object.values(perModel).find((entry) => typeof entry?.a === 'number');
  if (other !== undefined) return { ...other, source: 'measured-other-mode' };
  return { a: null, r: null, source: 'default' };
}

/**
 * How a base model sits in memory. This is the one choice the whole budget hangs
 * on, so it is a mode, not a flag:
 *
 *   qlora  — 4-bit NF4/FP4 quantised base (bitsandbytes). Weights are small; the
 *            adapter and its optimizer state are all that train. What makes a
 *            6 GB card possible.
 *   lora   — fp16 base, frozen, adapter in fp32. Full-precision forward pass, no
 *            quantisation noise; the base costs twice (fp16) what it does as
 *            4-bit, so on 6 GB it only fits the small models.
 *   full   — every parameter trains (fp32 master weights + Adam moments). Listed
 *            so the panel can show *why* it is not offered: at 4B it is 60+ GiB.
 */
export const TRAIN_MODES = {
  qlora: {
    label: 'QLoRA（4bit 量化底座）',
    // 4-bit body at 0.55 bytes/param. bnb leaves the embedding matrix in fp16
    // (~10% of params for a Qwen-class model, not the 35% an untrained-model
    // average suggests), so that share is added back at 2 bytes.
    baseBytesPerParam: 0.55,
    embeddingShare: 0.1,
    embeddingBytes: 2,
    // Bytes per (token × hidden unit) per layer for forward+backward
    // intermediates, net of logits. This was 4 until 2026-11-04, when a real
    // 0.5B run measured ~210 on a 3060 — a 50× underestimate. The number is
    // large because backward keeps one autograd-graph node per op per layer and
    // gradient checkpointing recomputes rather than frees them. Treat any value
    // near 4 as wrong; calibrate.
    activationsPerLayerHidden: 210,
    trainAll: false,
  },
  lora: {
    label: 'LoRA（fp16 底座）',
    baseBytesPerParam: 2,
    embeddingShare: 0,
    embeddingBytes: 0,
    // Same shape of cost; the fp16 base removes the dequantise working set but
    // adds fp16 activations. Left at the QLoRA figure until it is measured —
    // over-predicting here costs a rejected plan, under-predicting costs an OOM.
    activationsPerLayerHidden: 210,
    trainAll: false,
  },
  full: {
    label: '全参微调',
    baseBytesPerParam: 2,
    embeddingShare: 0,
    embeddingBytes: 0,
    activationsPerLayerHidden: 260,
    trainAll: true,
  },
};

/**
 * Estimate peak GPU memory for a LoRA-family fine-tune, component by component.
 *
 * @param {object} input
 * @param {string} input.modelId
 * @param {string} [input.mode] One of TRAIN_MODES keys; default `qlora`.
 * @param {number} input.maxSeqLen
 * @param {number} input.batchSize Per-device micro-batch.
 * @param {number} [input.loraR]
 * @param {boolean} [input.trainEmbeddings] Embeddings are frozen by default: on a
 *   1–4B model they are ~40% of all parameters, so training them would multiply
 *   optimizer state for no benefit on a chat SFT task.
 * @returns {{ totalGiB, components, fits, headroomGiB, note }}
 */
export function estimateMemory({
  modelId,
  mode = 'qlora',
  maxSeqLen = 2048,
  batchSize = 1,
  gradAccum = 8,
  loraR = 16,
  trainEmbeddings = false,
  gpuGiB,
  measuredPeakGiB = null,
}) {
  const shape = shapeOf(modelId);
  const profile = TRAIN_MODES[mode] ?? TRAIN_MODES.qlora;
  const baseWeights = shape.paramsB * 1e9 * profile.baseBytesPerParam
    + shape.paramsB * 1e9 * profile.embeddingShare * profile.embeddingBytes;
  const loraParams = shape.layers * 4 * shape.hidden * 2 * loraR;
  // Adapter in fp16 with fp32 Adam moments; a full fine-tune needs a master copy
  // plus grads for every parameter instead.
  const loraState = profile.trainAll
    ? shape.paramsB * 1e9 * (4 + 4 + 8)
    : loraParams * (2 * 2 + 4);
  const embedState = trainEmbeddings && !profile.trainAll ? shape.paramsB * 0.35e9 * (4 + 4) : 0;
  // Logits: `transformers` upcasts the language-modelling head output to fp32 for
  // its loss, so a forward pass materialises vocab × seq × 4 bytes — and that
  // tensor is *kept* until backward. On a 152k-vocab model at seq 512 that is
  // 0.29 GiB on its own; the first version of this estimator left it out and
  // under-predicted a real 0.5B run by 20×. It is a separate line item now
  // because it scales with the vocabulary, not with depth.
  const logits = batchSize * maxSeqLen * shape.vocab * 4;
  // Activations: forward/backward intermediates. The coefficient is measured when
  // this machine has a calibration entry (calibrate.py reports it *net of
  // logits*), and the coarse constant otherwise — `coefficientSource` travels with
  // the estimate so the panel can say which one it used.
  const measured = coefficientsFor(modelId, mode);
  const activationBytesPerToken = measured.a !== null
    ? measured.a * shape.layers * shape.hidden
    : shape.layers * shape.hidden * profile.activationsPerLayerHidden;
  const activations = batchSize * maxSeqLen * activationBytesPerToken;
  // The fragment reserve covers CUDA context, cuBLAS workspaces and allocator
  // fragmentation; it is not a per-sequence cache (training keeps none).
  const reserve = (measured.r !== null ? measured.r : 0.9) * GiB;
  const components = {
    baseWeights: round(baseWeights / GiB),
    loraState: round(loraState / GiB),
    embedState: round(embedState / GiB),
    logits: round(logits / GiB),
    activations: round(activations / GiB),
    reserve: round(reserve / GiB),
  };
  const totalGiB = round(Object.values(components).reduce((sum, value) => sum + value, 0));
  // A measured peak, when the lab has produced one for this shape, is the honest
  // number: the estimate exists to answer "will it fit" before anything runs.
  const effectiveTotal = typeof measuredPeakGiB === 'number' && Number.isFinite(measuredPeakGiB) ? round(measuredPeakGiB) : totalGiB;
  const headroomGiB = typeof gpuGiB === 'number' && Number.isFinite(gpuGiB) ? round(gpuGiB - effectiveTotal) : null;
  const fits = headroomGiB === null ? null : headroomGiB >= 0;
  return {
    model: modelId,
    mode,
    modeLabel: profile.label,
    knownShape: shape.known === true,
    maxSeqLen,
    batchSize,
    loraR,
    totalGiB: effectiveTotal,
    estimatedGiB: totalGiB,
    measured: typeof measuredPeakGiB === 'number' && Number.isFinite(measuredPeakGiB),
    coefficientSource: measured.source,
    components,
    headroomGiB,
    fits,
    note: shape.known === true
      ? null
      : '这个模型的形状不在内置表里，数字按名称里的参数量插值估计——以 M0 实测为准。',
  };
}

/** The smallest change that makes a plan fit (or null when it already fits / cannot). */
export function recommendation(estimate) {
  if (estimate.fits !== false) return null;
  const { components } = estimate;
  const over = Math.round((components.reserve + Math.max(0, -(estimate.headroomGiB ?? 0))) * 100) / 100;
  if (components.activations > components.baseWeights * 1.5) {
    return {
      action: 'maxSeqLen',
      suggest: Math.max(512, Math.floor(estimate.maxSeqLen / 2)),
      reason: `激活内存占大头（超 ${over} GiB）：先把最大序列长度减半，样本的绝大多数用不到 2048。`,
    };
  }
  // The mode is the cheaper lever when the base itself is what does not fit:
  // the same model as fp16 and as 4-bit is a 2–3× difference.
  if (estimate.mode !== 'qlora') {
    const asQlora = estimateMemory({
      modelId: estimate.model,
      mode: 'qlora',
      maxSeqLen: estimate.maxSeqLen,
      batchSize: estimate.batchSize,
      loraR: estimate.loraR,
    });
    return {
      action: 'mode',
      suggest: 'qlora',
      reason: `底座本身放不下（超 ${over} GiB）。同样这个模型换成 QLoRA（4bit）预计 ${asQlora.totalGiB} GiB。`,
    };
  }
  return {
    action: 'modelId',
    suggest: 'Qwen/Qwen2.5-1.5B-Instruct',
    reason: `连 4bit 底座都放不下（超 ${over} GiB）：换更小的底座，或在远端 GPU 上训练。`,
  };
}

const round = (value) => Math.round(value * 100) / 100;

/**
 * Read a GPU table out of `nvidia-smi --query-gpu=...` output.
 * `name, memory.total [MiB]` header style, one row per GPU.
 *
 * The real output carries a unit suffix — `NVIDIA GeForce RTX 3060 Laptop GPU,
 * 6144 MiB` — and the first version of this parser did `Number("6144 MiB")`,
 * which is NaN, so every row was skipped and the panel said "no GPU listed" on a
 * machine with a working card. The number is now extracted from whatever
 * surrounds it and the unit is honoured.
 */
export function parseNvidiaSmi(text) {
  const rows = [];
  for (const line of String(text ?? '').split(/\r?\n/)) {
    if (line.trim() === '') continue;
    const parts = line.split(',').map((part) => part.trim());
    if (parts.length < 2) continue;
    const amount = /([0-9]+(?:\.[0-9]+)?)\s*(MiB|GiB|MB|GB)?/i.exec(parts[1]);
    if (amount === null) continue;
    const value = Number(amount[1]);
    if (!Number.isFinite(value)) continue;
    const unit = (amount[2] ?? 'MiB').toLowerCase();
    // nvidia-smi reports MiB; GiB/GB only appear if a wrapper reformats it.
    const mib = unit === 'gib' || unit === 'gb' ? value * 1024 : value;
    rows.push({ name: parts[0], totalMiB: Math.round(mib), totalGiB: Math.round((mib / 1024) * 10) / 10 });
  }
  return rows;
}
