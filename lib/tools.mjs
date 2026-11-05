/**
 * The five model-facing tools.
 *
 * Each returns one Markdown document under `markdown` — the consumer of these
 * answers is the model, and the tables are already the argument — plus a small
 * structured `data` object for anything the panel wants to reuse.
 *
 * Tool design rule for this plugin: every tool that can spend GPU time or start
 * a process checks its capability (`allowSpawn` / `allowPublish`) and says which
 * switch is off. A tool that silently no-ops is worse than a missing tool,
 * because the model will describe work that never happened.
 */
import { join } from 'node:path';
import { listDatasets, buildDataset } from './dataset.mjs';
import { listSources, loadSource, saveSource, validateSourceInput } from './sources.mjs';
import { listSessions } from './session.mjs';
import { MODES, planSummary, validateTrainingPlan } from './training.mjs';

const markdownOutput = {
  type: 'object',
  properties: {
    markdown: { type: 'string' },
  },
  required: ['markdown'],
  additionalProperties: false,
};

const rendered = (value) => [{ type: 'text', text: value.markdown }];

const pct = (value) => (typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—');
const num = (value) => (typeof value === 'number' ? value.toLocaleString('en-US') : '—');

export function createTools(helpers) {
  const {
    root, store, backends,
    dshHome = () => process.env.DSH_HOME ?? '',
    estimateMemory = () => null,
    gpu = async () => ({ available: false, reason: '没有显卡探测', gpus: [] }),
  } = helpers;
  const settings = helpers.settings ?? {};

  return [
    {
      name: 'train_prepare',
      description:
        'Build a fine-tuning dataset from DSH session logs or a local JSONL/JSON file. Creates the data source, mines samples (scenario → action for agent sessions, or prompt → answer for datasets), redacts secrets/PII, SMILES-dedupes, splits by session group, and writes a data card. ' +
        'Read-only toward the GPU and the network: nothing is trained here. Returns the dataset id, split sizes, and every filter\'s drop count so a reader can judge the data before training on it.',
      parameters: {
        type: 'object',
        properties: {
          kind: { type: 'string', enum: ['agent', 'dataset'], description: 'agent = mine DSH session logs; dataset = a local .jsonl/.json you already have.' },
          id: { type: 'string', description: 'Source id, lowercase letters/digits/._-, e.g. "tool-calls-2026-11".' },
          label: { type: 'string', description: 'Human label shown in the panel.' },
          sessions: { type: 'array', items: { type: 'string' }, description: 'For kind=agent: session ids, or ["*"] for every session on disk.' },
          file: { type: 'string', description: 'For kind=dataset: absolute path to the .jsonl/.json file.' },
          mode: { type: 'string', enum: ['action', 'answer', 'both'], description: 'What a sample teaches. agent defaults to action; dataset defaults to answer.' },
          redaction: { type: 'string', enum: ['off', 'key', 'pii', 'path', 'full'], description: 'Redaction profile, default path (keys + PII + home paths).' },
          withGpuCheck: { type: 'boolean', description: 'Also report whether a 1.5B/4B plan fits the local GPU.' },
        },
        required: ['kind', 'id', 'label'],
        additionalProperties: false,
      },
      output: { schema: markdownOutput, render: (_args, value) => rendered(value) },
      async execute(args) {
        const kind = String(args?.kind ?? '');
        const sourceInput = {
          kind,
          id: args?.id,
          label: args?.label,
          sessions: args?.sessions,
          file: args?.file,
          mode: args?.mode,
          redaction: args?.redaction,
        };
        const validated = validateSourceInput(sourceInput);
        if (validated.error !== undefined) throw new Error(`train_prepare: ${validated.error}`);
        if (loadSource(root, validated.value.id) !== null) throw new Error(`train_prepare: 数据源 ${validated.value.id} 已存在，换一个 id。`);
        const source = saveSource(root, validated.value);
        const datasetId = `${source.id}-${new Date().toISOString().slice(0, 10)}`;
        const built = buildDataset({ root, source, datasetId, dshHome: dshHome() });
        const card = built.manifest;
        const lines = [
          `# 数据卡：${card.id}`,
          '',
          `来源 \`${card.sourceId}\`（${card.sourceKind === 'agent' ? '会话日志' : '外部数据集'}）· 模式 ${card.mode} · 脱敏档位 ${card.redaction}`,
          '',
          '| 项 | 数量 |',
          '|---|---|',
          `| 原始样本 | ${num(card.counts.raw)} |`,
          `| 保留（去重去坏后） | ${num(card.counts.kept)} |`,
          `| 重复 | ${num(card.counts.duplicates)} |`,
          `| 无效 | ${num(card.counts.invalid)} |`,
          `| train / val / eval | ${card.counts.train} / ${card.counts.val} / ${card.counts.eval} |`,
          `| 分组（按会话） | ${num(card.counts.groups)} |`,
          `| token 中位数 / P90 / 最大 | ${num(card.tokens.median)} / ${num(card.tokens.p90)} / ${num(card.tokens.max)} |`,
        ];
        const drops = card.provenance?.dropped;
        if (drops !== undefined) {
          lines.push('', '**过滤明细**', '', '| 丢弃原因 | 条数 |', '|---|---|');
          lines.push(`| 幻觉工具名 | ${num(drops.hallucinated_tool)} |`);
          lines.push(`| 失败调用（不是答案） | ${num(drops.failed_call)} |`);
          lines.push(`| 没有可判定回应 | ${num(drops.no_context)} |`);
          lines.push(`| 空标签 | ${num(drops.empty_label)} |`);
          lines.push(`| 触顶截断 | ${num(drops.capped)} |`);
        }
        const redactions = card.provenance?.redactions;
        if (redactions !== undefined && Object.keys(redactions).length > 0) {
          const summary = Object.entries(redactions).map(([key, value]) => `${key} × ${value}`).join('，');
          lines.push('', `**已脱敏**：${summary}（只报规则与次数，不回显原文）`);
        }
        if (Array.isArray(card.provenance?.undecodable) && card.provenance.undecodable.length > 0) {
          lines.push('', `**读不出来的会话**（${card.provenance.undecodable.length}）：${card.provenance.undecodable.map((item) => item.sessionId).join(', ')}`);
        }
        if (card.counts.kept === 0) {
          lines.push('', '> 这个数据集是空的。常见原因：会话里没有成功的工具调用（试 `mode: "answer"`）、或者选中的会话读完是空的。');
        } else if (card.counts.kept < 20) {
          lines.push('', '> 样本少于 20 条：能跑通流程，但训出来的差异大概率落在噪声里——先多选几个会话。');
        }
        if (args?.withGpuCheck === true) {
          const gpuInfo = await gpu();
          const plan = estimateMemory({ modelId: 'Qwen/Qwen2.5-1.5B-Instruct', maxSeqLen: Math.min(4096, Math.max(1024, card.tokens.p90 * 2)), gpuGiB: gpuInfo?.gpus?.[0]?.totalGiB });
          lines.push('', `**显存预估**（1.5B，seq ${plan.maxSeqLen}）：约 ${plan.totalGiB} GiB${plan.headroomGiB === null ? '' : `，余量 ${plan.headroomGiB} GiB`}`);
        }
        return { markdown: lines.join('\n') };
      },
      presentCall: (args) => ({ card: 'generic', title: `准备数据：${args?.label ?? args?.id ?? ''}` }),
    },

    {
      name: 'train_run',
      description:
        'Start a QLoRA fine-tuning run on a dataset produced by train_prepare. Refuses before spawning when the estimated GPU memory does not fit (and says what to change), when the "allow spawn" switch is off, or when the dataset does not exist. ' +
        'Returns the run id; progress is streamed to the panel, and train_status reads it back. This spends real GPU time.',
      parameters: {
        type: 'object',
        properties: {
          datasetId: { type: 'string', description: 'Dataset id from train_prepare (see the data card).' },
          mode: { type: 'string', enum: ['qlora', 'lora'], description: 'qlora = 4bit quantised base (fits 6 GB); lora = fp16 base (higher quality, ~2× base memory). Defaults to qlora.' },
          modelId: { type: 'string', description: 'Base model repo id, e.g. "Qwen/Qwen2.5-1.5B-Instruct". Defaults to the 1.5B.' },
          maxSeqLen: { type: 'number', description: 'Sequence length (128–8192, default 2048). The 6 GB default card needs 1024–2048.' },
          epochs: { type: 'number', description: 'Epochs (0.1–20, default 1).' },
          loraR: { type: 'number', enum: [4, 8, 16, 32, 64], description: 'LoRA rank, default 16.' },
          learningRate: { type: 'number', description: 'Learning rate (1e-6–1e-2, default 2e-4).' },
          dryRun: { type: 'boolean', description: 'Build the plan and estimate memory without training.' },
        },
        required: ['datasetId'],
        additionalProperties: false,
      },
      output: { schema: markdownOutput, render: (_args, value) => rendered(value) },
      async execute(args) {
        if (settings.allowSpawn !== true) throw new Error('train_run: 「允许启动训练进程」没有打开（插件配置 → fine-tune）。打开之后我再启动训练。');
        const datasetId = String(args?.datasetId ?? '');
        const validated = validateTrainingPlan({ ...args, modelId: args?.modelId ?? 'Qwen/Qwen2.5-1.5B-Instruct' });
        if (validated.ok !== true) throw new Error(`train_run: ${validated.error}`);
        const plan = validated.value;
        const modelId = plan.modelId;
        const dataset = listDatasets(root).find((entry) => entry.id === datasetId);
        if (dataset === undefined) throw new Error(`train_run: 没有数据集 ${datasetId}（先 train_prepare）。`);
        const gpuInfo = await gpu();
        const estimate = estimateMemory({ ...plan, gpuGiB: gpuInfo?.gpus?.[0]?.totalGiB });
        const lines = [`# 训练计划：${dataset.id} → ${modelId}`, '', '| 项 | 值 |', '|---|---|'];
        lines.push(`| 方式 | ${plan.mode.toUpperCase()}${plan.mode === 'qlora' ? `（${plan.quantType}${plan.doubleQuant ? '+双量化' : ''}）` : '（fp16 底座）'} |`);
        lines.push(`| 底座 | \`${modelId}\` |`);
        lines.push(`| train / val / eval | ${dataset.counts.train} / ${dataset.counts.val} / ${dataset.counts.eval} |`);
        lines.push(`| 序列长度 | ${plan.maxSeqLen} |`);
        lines.push(`| LoRA | r${plan.loraR} / α${plan.loraAlpha} / dropout ${plan.loraDropout} |`);
        lines.push(`| 轮数 / 学习率 | ${plan.epochs} / ${plan.learningRate} |`);
        lines.push(`| 显存预估 | ${estimate.totalGiB} GiB${estimate.headroomGiB === null ? '' : `（余 ${estimate.headroomGiB} GiB）`} |`);
        if (estimate.fits === false) {
          const fix = recommendation(estimate);
          throw new Error(`train_run: 显存放不下，没有启动。${fix === null ? '' : fix.reason}`);
        }
        if (args?.dryRun === true) {
          lines.push('', '> 试运行：没有启动任何进程。去掉 `dryRun` 才会真的训练。');
          return { markdown: lines.join('\n') };
        }
        const id = store.create({ kind: 'train', datasetId, backend: 'local', input: { ...plan, summary: planSummary(plan) } });
        const handle = backends.local.start(id, {
          verb: 'train', model: modelId, data: join(root, 'datasets', datasetId), out: join(root, 'outputs', id),
          ...plan,
        }, {
          onEvent: (event) => store.progress(id, event.kind === 'metric' ? { kind: 'metric', step: event.step, values: event.values } : { kind: 'worker', payload: event }),
          onExit: ({ code, signal, error }) => {
            const record = store.get(id);
            if (record === null || record.state === 'cancelled') return;
            if (error !== undefined) store.move(id, 'failed', { error: { code: 'spawn', message: error } });
            else if (code === 0) store.move(id, 'succeeded');
            else store.move(id, 'failed', { error: { code: signal === null ? `exit-${code}` : `signal-${signal}`, message: 'worker 退出异常，看 runs 目录里的 stderr.log' } });
          },
        });
        store.attach(id, handle);
        store.move(id, 'running');
        lines.push('', `已在后台启动：\`${id}\`。进度在侧栏「模型微调」面板里实时显示；跑完后用 train_status 读结果。`);
        return { markdown: lines.join('\n') };
      },
      presentCall: (args) => ({ card: 'generic', title: `启动训练：${args?.datasetId ?? ''}` }),
    },

    {
      name: 'train_status',
      description:
        'Read back fine-tuning runs: state, last loss, trainable parameters, artifacts, and any evaluation verdict. Omit `id` for the most recent run of each kind. Read-only.',
      parameters: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'Run id. Omit for the latest run.' },
          limit: { type: 'number', description: 'How many recent runs to list (1–20, default 5).' },
        },
        additionalProperties: false,
      },
      output: { schema: markdownOutput, render: (_args, value) => rendered(value) },
      async execute(args) {
        const id = typeof args?.id === 'string' && args.id.trim() !== '' ? args.id.trim() : null;
        if (id !== null) {
          const job = store.job(id);
          if (job === null) throw new Error(`train_status: 没有任务 ${id}`);
          return { markdown: runMarkdown(job) };
        }
        const limit = clamp(args?.limit, 5, 1, 20);
        const runs = store.list({ limit });
        if (runs.length === 0) return { markdown: '还没有任何训练任务。' };
        const lines = ['# 最近的训练任务', '', '| 任务 | 状态 | 数据集 | 底座 | loss | 开始 |', '|---|---|---|---|---|---|'];
        for (const run of runs) {
          lines.push(`| \`${run.id}\` | ${run.state} | ${run.datasetId ?? '—'} | ${run.input?.modelId ?? '—'} | ${typeof run.metrics?.loss === 'number' ? run.metrics.loss.toFixed(4) : '—'} | ${new Date(run.startedAt ?? run.createdAt).toLocaleString('zh-CN')} |`);
        }
        return { markdown: lines.join('\n') };
      },
      presentCall: () => ({ card: 'generic', title: '训练状态' }),
    },

    {
      name: 'train_check',
      description:
        'Diagnose whether fine-tuning can run on this machine: finds the GPU and its memory, estimates peak memory for a given model and sequence length, and names the smallest change that would make it fit. Read-only — it does not start anything.',
      parameters: {
        type: 'object',
        properties: {
          modelId: { type: 'string', description: 'Model to estimate (default "Qwen/Qwen2.5-1.5B-Instruct").' },
          maxSeqLen: { type: 'number', description: 'Sequence length to estimate at (default 2048).' },
          backend: { type: 'string', enum: ['local', 'ssh'], description: 'Which backend to check (default local).' },
        },
        additionalProperties: false,
      },
      output: { schema: markdownOutput, render: (_args, value) => rendered(value) },
      async execute(args) {
        const modelId = String(args?.modelId ?? 'Qwen/Qwen2.5-1.5B-Instruct');
        const mode = args?.mode === 'lora' ? 'lora' : 'qlora';
        const maxSeqLen = clamp(args?.maxSeqLen, 2048, 128, 8192);
        const backend = args?.backend === 'ssh' ? 'ssh' : 'local';
        const lines = ['# 训练环境体检', ''];
        if (backend === 'local') {
          const gpuInfo = await gpu();
          if (gpuInfo.available !== true) {
            lines.push(`**显卡**：❌ ${gpuInfo.reason ?? '不可用'}`);
            lines.push('', '没有 NVIDIA 显卡时，本地训练不可行：要么用 SSH 后端在远端 GPU 上练，要么只用本地数据/评测两步。');
          } else {
            lines.push('**显卡**：');
            for (const item of gpuInfo.gpus) lines.push(`- ${item.name} — ${item.totalGiB} GiB`);
          }
          const estimate = estimateMemory({ modelId, mode, maxSeqLen, gpuGiB: gpuInfo?.gpus?.[0]?.totalGiB });
          lines.push('', `**显存预估**（${mode.toUpperCase()}，${modelId}，seq ${maxSeqLen}）`, '', '| 组成 | GiB |', '|---|---|');
          for (const [key, value] of Object.entries(estimate.components)) lines.push(`| ${key} | ${value} |`);
          lines.push(`| **合计** | **${estimate.totalGiB}** |`);
          lines.push('', '| 方式 | 同一模型预计 |', '|---|---|');
          for (const other of Object.keys(MODES)) {
            const alt = estimateMemory({ modelId, mode: other, maxSeqLen, gpuGiB: gpuInfo?.gpus?.[0]?.totalGiB });
            lines.push(`| ${other.toUpperCase()} | ${alt.totalGiB} GiB${alt.fits === false ? '（放不下）' : ''} |`);
          }
          if (estimate.headroomGiB !== null) lines.push('', estimate.fits ? `✅ 余量 ${estimate.headroomGiB} GiB` : `❌ 超出 ${Math.abs(estimate.headroomGiB)} GiB。${recommendationText(estimate)}`);
        } else {
          const target = settings.sshTarget;
          if (typeof target !== 'string' || target === '') {
            lines.push('SSH 后端没有配置目标（`sshTarget`）。配置后这里会报远端 Python + torch 的版本。');
          } else {
            const probed = await backends.ssh.probe(target);
            lines.push(`**远端 ${target}**：${probed.ok ? '✅' : '❌'} ${probed.python ?? probed.reason ?? ''}`);
          }
        }
        const probe = await backends.local.probe();
        lines.push('', `**本地 Python**：${probe.ok ? `✅ ${probe.python}` : `❌ ${probe.reason}`}`);
        lines.push('', `**开关**：允许启动训练进程 = ${settings.allowSpawn === true ? '开' : '关'}；允许发布 = ${settings.allowPublish === true ? '开' : '关'}`);
        return { markdown: lines.join('\n') };
      },
      presentCall: () => ({ card: 'generic', title: '训练环境体检' }),
    },

    {
      name: 'train_eval',
      description:
        'Explain how a run will be evaluated and what its last verdict was. Evaluation itself needs a running model and a judge, which the panel drives; this tool reports the recorded verdict (adherence delta and the publish gate) plus instructions for finishing the comparison. Read-only.',
      parameters: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'Run id (default: the latest run).' },
        },
        additionalProperties: false,
      },
      output: { schema: markdownOutput, render: (_args, value) => rendered(value) },
      async execute(args) {
        const id = typeof args?.id === 'string' && args.id.trim() !== '' ? args.id.trim() : null;
        const job = id === null ? store.job(store.list({ limit: 1 })[0]?.id) : store.job(id);
        if (job === null) return { markdown: '还没有可以评测的训练任务。' };
        return { markdown: evaluationMarkdown(job) };
      },
      presentCall: () => ({ card: 'generic', title: '评测结论' }),
    },
  ];
}

function runMarkdown(job) {
  const { record } = job;
  const lines = [`# ${record.id}`, ''];
  lines.push(`- 状态：**${record.state}**${record.error === null ? '' : `（${record.error.message}）`}`);
  lines.push(`- 数据集：${record.datasetId ?? '—'} · 底座：\`${record.input?.modelId ?? '—'}\` · 后端：${record.backend}`);
  if (record.startedAt !== null) lines.push(`- 开始：${new Date(record.startedAt).toLocaleString('zh-CN')}${record.finishedAt === null ? '' : ` · 结束：${new Date(record.finishedAt).toLocaleString('zh-CN')}`}`);
  const metrics = record.metrics ?? {};
  if (typeof metrics.loss === 'number') lines.push(`- 最后一步 loss：${metrics.loss.toFixed(4)}${typeof metrics.step === 'number' ? `（step ${metrics.step}）` : ''}`);
  if (typeof metrics.trainableParams === 'number') lines.push(`- 可训练参数：${metrics.trainableParams.toLocaleString('en-US')}`);
  if (typeof metrics.peakGiB === 'number') lines.push(`- 显存峰值：${metrics.peakGiB} GiB`);
  if (Array.isArray(record.artifacts) && record.artifacts.length > 0) {
    lines.push('', '**产物**');
    for (const artifact of record.artifacts) lines.push(`- \`${artifact}\``);
  }
  const evaluation = [...job.events].reverse().find((event) => event.kind === 'evaluation');
  if (evaluation !== undefined) {
    lines.push('', `**评测**：${evaluation.verdict?.status === 'green' ? '✅ 可以发布' : '⛔ 不建议发布'}（贴合度 ${pct(evaluation.candidate?.score)}，底座 ${pct(evaluation.base?.score)}）`);
    for (const reason of evaluation.verdict?.reasons ?? []) lines.push(`- ${reason}`);
  }
  return lines.join('\n');
}

function evaluationMarkdown(job) {
  const evaluation = [...job.events].reverse().find((event) => event.kind === 'evaluation');
  const lines = [`# 评测：${job.record.id}`, ''];
  if (evaluation === undefined) {
    lines.push('还没有评测记录。评测由面板驱动（要加载两个模型做对比），结论会写回这里：', '');
    lines.push('1. 在面板里选这次训练，点「对比评测」；');
    lines.push('2. 面板把 eval 集的输出分别喂给「底座」和「微调后」，算出贴合度（工具名 / 参数 / 格式）；');
    lines.push('3. 贴合度掉落超过 3 个百分点就**不能发布**——这是硬门槛。');
    return lines.join('\n');
  }
  lines.push(`| 指标 | 底座 | 微调后 |`, '|---|---|---|');
  lines.push(`| 贴合度 | ${pct(evaluation.base?.score)} | ${pct(evaluation.candidate?.score)} |`);
  lines.push('', `**结论：${evaluation.verdict?.status === 'green' ? '可以发布' : '不建议发布'}**`);
  for (const reason of evaluation.verdict?.reasons ?? []) lines.push(`- ${reason}`);
  return lines.join('\n');
}

function recommendationText(estimate) {
  if (estimate.fits !== false) return '能放下。';
  return estimate.components.activations > estimate.components.baseWeights
    ? `先降序列长度（例如 ${Math.max(512, Math.floor(estimate.maxSeqLen / 2))}），或换更小的底座，或者在远端 GPU 上训练。`
    : '换更小的底座（如 Qwen2.5-1.5B），或在远端 GPU 上训练。';
}

function clamp(value, fallback, min, max) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, Math.trunc(number)));
}
