/**
 * Training hyperparameters: the numbers a user actually adjusts, validated once.
 *
 * The panel sends whatever the form holds; the host and the tool both go through
 * this function, so a bad value is rejected with a sentence a human can act on
 * ("loraR 必须是 4/8/16/32/64 之一") instead of failing 10 minutes into a run.
 * Every default here is the one M0-LAB.md measures first.
 *
 * The parameter set follows LLaMA-Factory's, grouped the way its panel groups it:
 * 1) 训练方式与底座, 2) LoRA 结构, 3) 优化, 4) 数据与序列, 5) 显存技巧, 6) 产物.
 * Where a knob only works on hardware that is not present (DeepSpeed, multi-GPU,
 * FlashAttention on a pre-Ampere card), it is still defined here so the panel can
 * show it as disabled *with the reason* — hiding a knob teaches nobody why.
 */

export const MODES = {
  qlora: {
    id: 'qlora',
    label: 'QLoRA',
    summary: '4bit 量化底座 + LoRA 适配器',
    quality: '质量接近 LoRA；量化会带来轻微噪声',
    memory: '省显存，6 GB 可跑 1.5B–4B',
    quantTypes: ['nf4', 'fp4'],
  },
  lora: {
    id: 'lora',
    label: 'LoRA',
    summary: 'fp16 全精度底座 + LoRA 适配器',
    quality: '质量最稳：无量化噪声',
    memory: '底座占 fp16 两倍，6 GB 只够 1.5B 以下',
    quantTypes: [],
  },
};

/** Allowed LoRA ranks: powers of two that a 6 GB card can actually afford. */
const RANKS = [4, 8, 16, 32, 64];
const SEQUENCE_LENGTHS = [256, 512, 1024, 1536, 2048, 3072, 4096];

/** LR schedulers `transformers` accepts by name. */
export const LR_SCHEDULERS = [
  ['linear', 'linear'],
  ['cosine', 'cosine'],
  ['cosine_with_restarts', 'cosine_with_restarts'],
  ['polynomial', 'polynomial'],
  ['constant', 'constant'],
  ['constant_with_warmup', 'constant_with_warmup'],
  ['inverse_sqrt', 'inverse_sqrt'],
  ['reduce_lr_on_plateau', 'reduce_lr_on_plateau'],
];

/**
 * Optimisers, split by whether they exist without bitsandbytes.
 *
 * `paged_adamw_8bit` is what a 6 GB card should use: the optimiser state is the
 * second-largest allocation after the base model. `adafactor` is the fallback
 * when bitsandbytes is missing (no GPU) — nearly as frugal, pure PyTorch.
 */
export const OPTIMIZERS = [
  { id: 'paged_adamw_8bit', label: 'paged AdamW 8bit', needs: 'bitsandbytes', note: '显存最省，推荐' },
  { id: 'adamw_8bit', label: 'AdamW 8bit', needs: 'bitsandbytes', note: '同上，不分页' },
  { id: 'paged_adamw_32bit', label: 'paged AdamW 32bit', needs: 'bitsandbytes', note: '更稳，占更多显存' },
  { id: 'adafactor', label: 'Adafactor', needs: null, note: '不依赖 bitsandbytes' },
  { id: 'adamw_torch', label: 'AdamW (torch)', needs: null, note: '最标准的做法，显存占用最高' },
];

/** Which modules a LoRA adapter can attach to, per architecture family. */
export const TARGET_PRESETS = [
  { id: 'all-linear', label: '全部线性层（推荐）', modules: null },
  { id: 'attn-only', label: '仅注意力', modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj'] },
  { id: 'attn-mlp', label: '注意力 + MLP', modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'] },
];

const clampInt = (value, fallback, min, max) => {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, Math.trunc(number)));
};

const clampFloat = (value, fallback, min, max) => {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, number));
};

const bool = (value, fallback) => (value === undefined || value === null ? fallback : value !== false && value !== 'false');

/**
 * Validate one training plan.
 *
 * @param {object} input Raw form values.
 * @returns {{ ok: true, value: object } | { ok: false, error: string }}
 */
export function validateTrainingPlan(input = {}) {
  const mode = String(input.mode ?? 'qlora');
  if (MODES[mode] === undefined) return { ok: false, error: `训练方式只能是 ${Object.keys(MODES).join(' / ')}` };

  const modelId = String(input.modelId ?? 'Qwen/Qwen2.5-1.5B-Instruct');
  if (!/^[A-Za-z0-9._:/-]+$/.test(modelId)) {
    return { ok: false, error: 'modelId 含有不安全的字符（只允许字母、数字和 ._:/ -）' };
  }

  // ---------------------------------------------------------------- LoRA shape
  const loraR = clampInt(input.loraR, 16, 1, 256);
  if (!RANKS.includes(loraR)) return { ok: false, error: `loraR 必须是 ${RANKS.join('/')} 之一（收到 ${loraR}）` };
  const loraAlpha = clampInt(input.loraAlpha, loraR * 2, 1, 512);
  const loraDropout = clampFloat(input.loraDropout, 0.05, 0, 0.5);
  // `use_rslora` scales the adapter by sqrt(r) instead of r, which keeps the
  // effective update size stable when r changes — otherwise raising r silently
  // raises the learning rate on the adapter.
  const useRsLora = bool(input.useRsLora, false);
  const useDora = bool(input.useDora, false);

  // -------------------------------------------------------------- optimisation
  const maxSeqLen = clampInt(input.maxSeqLen, 2048, 128, 8192);
  const epochs = clampFloat(input.epochs, 1, 0.1, 20);
  const learningRate = clampFloat(input.learningRate, 2e-4, 1e-6, 1e-2);
  const batchSize = clampInt(input.batchSize, 1, 1, 16);
  const gradAccum = clampInt(input.gradAccum, 8, 1, 128);
  const warmupRatio = clampFloat(input.warmupRatio, 0.03, 0, 0.3);

  const lrScheduler = String(input.lrScheduler ?? 'cosine');
  if (!LR_SCHEDULERS.some(([id]) => id === lrScheduler)) {
    return { ok: false, error: `学习率调度只能是 ${LR_SCHEDULERS.map(([id]) => id).join(' / ')}` };
  }

  const optim = String(input.optim ?? 'paged_adamw_8bit');
  if (!OPTIMIZERS.some((entry) => entry.id === optim)) {
    return { ok: false, error: `优化器只能是 ${OPTIMIZERS.map((entry) => entry.id).join(' / ')}` };
  }

  const weightDecay = clampFloat(input.weightDecay, 0, 0, 1);
  const adamBeta2 = clampFloat(input.adamBeta2, 0.999, 0.9, 0.9999);
  const maxGradNorm = clampFloat(input.maxGradNorm, 1, 0, 100);

  // Early stopping is about hours of wasted GPU, so it defaults on.
  const earlyStopping = bool(input.earlyStopping, false);
  const earlyStoppingPatience = clampInt(input.earlyStoppingPatience, 3, 1, 20);

  const maxSteps = input.maxSteps === undefined || input.maxSteps === null || input.maxSteps === ''
    ? null
    : clampInt(input.maxSteps, 0, 1, 100000);

  // ------------------------------------------------------------ memory tricks
  const gradientCheckpointing = bool(input.gradientCheckpointing, true);
  // `auto` lets the worker pick from what the card supports; forcing one on a
  // card that lacks it is caught by the worker, not silently downgraded here.
  const precision = String(input.precision ?? 'auto');
  if (!['auto', 'bf16', 'fp16', 'fp32'].includes(precision)) {
    return { ok: false, error: '计算精度只能是 auto / bf16 / fp16 / fp32' };
  }
  const flashAttention = bool(input.flashAttention, false);
  const deepspeed = String(input.deepspeed ?? 'none');
  if (!['none', 'zero2', 'zero3'].includes(deepspeed)) {
    return { ok: false, error: 'DeepSpeed 只能是 none / zero2 / zero3' };
  }
  if (deepspeed !== 'none') {
    return { ok: false, error: 'DeepSpeed 需要 Linux + 多卡环境；当前面板还没有接入（见 DESIGN.md §10）' };
  }

  // ------------------------------------------------------------ data handling
  // Packing concatenates short samples into full-length sequences. It raises
  // throughput a lot on short samples and is wrong when the samples must stay
  // separate (our action labels are one tool call each).
  const packing = bool(input.packing, false);
  const trainOnPrompt = bool(input.trainOnPrompt, false);
  const maskHistory = bool(input.maskHistory, true);
  // Resuming from the previous run's adapter: the same task, more data.
  const resumeFrom = input.resumeFrom === undefined || input.resumeFrom === null || input.resumeFrom === ''
    ? null
    : String(input.resumeFrom).replace(/[^A-Za-z0-9._/-]/g, '');

  // ---------------------------------------------------------------- artifacts
  const mergeAfter = bool(input.mergeAfter, false);
  const exportTarget = String(input.exportTarget ?? 'none');
  if (!['none', 'gguf', 'ollama'].includes(exportTarget)) {
    return { ok: false, error: '导出目标只能是 none / gguf / ollama' };
  }

  // A quantised base is required for QLoRA and meaningless for LoRA.
  let quantType = String(input.quantType ?? 'nf4');
  if (mode === 'qlora' && !MODES.qlora.quantTypes.includes(quantType)) {
    return { ok: false, error: `量化类型只能是 ${MODES.qlora.quantTypes.join(' / ')}` };
  }
  if (mode !== 'qlora') quantType = null;
  const doubleQuant = mode === 'qlora' && bool(input.doubleQuant, true);

  // Cross-checks that only make sense in combination.
  if (packing && maxSeqLen < 512) {
    return { ok: false, error: '打包（packing）需要 seq ≥ 512 才有意义：太短的序列打包会把答案切散' };
  }
  if (trainOnPrompt && !packing) {
    return { ok: false, error: '「连 prompt 一起训练」通常要配合打包使用；单独开启会让模型学会复述问题' };
  }
  if (useDora && mode === 'qlora') {
    return { ok: false, error: 'DoRA 在 4bit 量化底座上需要额外内存，6 GB 卡会放不下；请改用 LoRA 或关闭 DoRA' };
  }

  return {
    ok: true,
    value: {
      mode,
      modelId,
      quantType,
      doubleQuant,
      // `all-linear` (null modules) is what LLaMA-Factory defaults to: it finds
      // every linear projection, which matters because Qwen/Llama name them
      // differently (q_proj vs query_key_value vs W_pack).
      targetModules: Array.isArray(input.targetModules) && input.targetModules.length > 0
        ? input.targetModules.map(String).filter((name) => /^[A-Za-z0-9_.]+$/.test(name)).slice(0, 32)
        : undefined,
      loraR,
      loraAlpha,
      loraDropout,
      useRsLora,
      useDora,
      maxSeqLen,
      epochs,
      learningRate,
      lrScheduler,
      warmupRatio,
      batchSize,
      gradAccum,
      optim,
      weightDecay,
      adamBeta2,
      maxGradNorm,
      earlyStopping,
      earlyStoppingPatience,
      maxSteps,
      gradientCheckpointing,
      precision,
      flashAttention,
      deepspeed,
      packing,
      trainOnPrompt,
      maskHistory,
      resumeFrom,
      mergeAfter,
      exportTarget,
      seed: clampInt(input.seed, 42, 0, 1_000_000),
    },
  };
}

/** A one-line human summary of a plan, for the run record and the toast. */
export function planSummary(plan) {
  const base = `${MODES[plan.mode]?.label ?? plan.mode} · ${plan.modelId} · seq ${plan.maxSeqLen} · r${plan.loraR}/α${plan.loraAlpha} · ${plan.epochs} 轮 · lr ${plan.learningRate} · ${plan.lrScheduler ?? 'cosine'}`;
  const parts = [plan.mode === 'qlora' ? `${base} · ${plan.quantType}${plan.doubleQuant ? '+双量化' : ''}` : base];
  if (plan.useDora) parts.push('DoRA');
  if (plan.useRsLora) parts.push('rsLoRA');
  if (plan.packing) parts.push('打包');
  if (plan.optim && plan.optim !== 'paged_adamw_8bit') parts.push(plan.optim);
  return parts.join(' · ');
}

/**
 * Which advanced knobs this machine can actually honour.
 *
 * The panel renders every knob regardless; this decides enabled/disabled and the
 * sentence shown next to a disabled one. Returned as plain data so the host and
 * the client can agree without the client doing its own GPU detection.
 *
 * @param {{ gpu: object|null, torch: string|null, deps: object }} env
 */
export function capabilityOf(env = {}) {
  const gpu = env.gpu ?? null;
  const hasGpu = gpu !== null && gpu.name !== undefined && !/none|n\/a/i.test(String(gpu.name));
  const cc = Number(gpu?.capability ?? 0) || 0;
  const deps = env.deps ?? {};
  const features = {
    bf16: { ok: Boolean(env.bf16Supported), why: '这张卡不支持 bf16，将使用 fp16' },
    fp16: { ok: hasGpu, why: '没有可用的 GPU' },
    flashAttention2: {
      ok: hasGpu && cc >= 8.0 && deps.flashAttn === true,
      why: hasGpu
        ? (cc >= 8.0 ? '未安装 flash-attn（pip install flash-attn）' : 'FlashAttention-2 需要 Ampere（算力 8.0）及以上')
        : '需要 NVIDIA GPU',
    },
    deepspeed: {
      ok: hasGpu && env.platform === 'linux' && Number(gpu?.count ?? 1) > 1,
      why: gpu && Number(gpu.count ?? 1) > 1 ? '需要 Linux + 多卡' : '只有一张卡，DeepSpeed ZeRO 没有收益',
    },
    bitsandbytes: { ok: deps.bitsandbytes === true, why: '未安装 bitsandbytes（pip install bitsandbytes）' },
    packing: { ok: true, why: '' },
    gguf: { ok: deps.llamaCpp === true, why: '导出 GGUF 需要 llama.cpp 的 convert_hf_to_gguf.py' },
  };
  return { hasGpu, gpu, features };
}

export { RANKS as LORA_RANKS, SEQUENCE_LENGTHS };
