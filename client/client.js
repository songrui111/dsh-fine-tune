// dsh-fine-tune, browser half.
//
// A plain hand-written bundle, same shape as the ones DSH ships: no build step,
// no import/export keywords, no JSX. The page loads it as a script, the module
// loader hands it a `require`, and the only module it asks for is React.
//
// The panel is organised around the four questions a fine-tuning run actually
// asks, in order — never a settings form:
//
//   1. Are the data right?      (数据源 → 数据卡 → 样本预览)
//   2. Will it fit?             (显卡 + 显存预估 + 一条可点击的建议)
//   3. Is it training?          (实时 loss、指标、取消)
//   4. Is it better?            (A/B 评测 → 结论 → 发布按钮的开合)
//
// Two rules this half follows strictly:
//   - It never claims to have trained or produced anything by itself. Training
//     starts through the host's action route; evaluation is recorded through it.
//   - Every blocked state says which switch or number is blocking, because a
//     disabled button with no explanation is the most common complaint about
//     panels like this one.
//
// Layout facts this file depends on (verified against shipped client bundles):
//   - `sidebar.panellist` entries carry `id`, `order`, `label` and render their
//     own icon; there is no icon field.
//   - `main` is keyed by that same id.
//   - both seats are registered from the innermost callback so they appear together.
window.__ModuleLoader__.load({
  id: 'dsh-fine-tune',
  factory: (require) => {
    var module = { exports: {} }
    var exports = module.exports
    Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' })

    const React = require('react')
    const h = React.createElement

    /** Panel identity: the rail entry and the main seat must agree. */
    const PANEL_ID = 'fine-tune'
    const API = '/fine-tune/api'

    const C = {
      dim: 'rgba(127,127,127,.85)',
      line: 'rgba(127,127,127,.22)',
      soft: 'rgba(127,127,127,.10)',
      /** Opaque surface for floating layers, and the text on it.
       *
       *  Straight from DSH's theme rather than derived: its page surface
       *  (`--dsw-alias-bg-layer-1` -> `--dsw-static-neutral-bluish-00`) is #fff in
       *  the light theme, and `--dsw-alias-label-primary` is #0f1115. The panel does
       *  not paint a background of its own, so a popup in this colour is
       *  indistinguishable from the page it floats over.
       */
      solid: '#fff',
      solidText: '#0f1115',
      /** Ordinary body text. `C.text` was referenced by two styles long before it
       *  existed, so those colours silently fell back to the inherited one. */
      text: 'CanvasText',
      ok: '#2f9e75',
      warn: '#c98a2b',
      accent: '#4f8ff0',
      bad: '#cf5656',
    }

    const S = {
      root: { display: 'flex', flexDirection: 'column', gap: 12, padding: '14px 14px 28px', fontSize: 13, lineHeight: 1.5, overflowY: 'auto', height: '100%', minHeight: 0 },
      head: { display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 8 },
      title: { fontSize: 15, fontWeight: 600 },
      sub: { fontSize: 11, color: C.dim },
      section: { border: `1px solid ${C.line}`, borderRadius: 10, padding: '10px 12px' },
      sectionHead: { display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 8, gap: 8 },
      sectionTitle: { fontSize: 12.5, fontWeight: 600 },
      card: { border: `1px solid ${C.line}`, borderRadius: 10, padding: '8px 10px' },
      grid2: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 },
      big: { fontSize: 18, fontWeight: 650, lineHeight: 1.2, fontVariantNumeric: 'tabular-nums' },
      label: { fontSize: 11, color: C.dim },
      row: { display: 'flex', alignItems: 'center', gap: 8, padding: '5px 0', borderBottom: `1px solid ${C.line}` },
      rowLast: { display: 'flex', alignItems: 'center', gap: 8, padding: '5px 0' },
      clickable: { cursor: 'pointer' },
      name: { flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
      num: { fontVariantNumeric: 'tabular-nums', color: C.dim, fontSize: 11.5, whiteSpace: 'nowrap' },
      pill: { fontSize: 10, padding: '1px 7px', borderRadius: 999, border: '1px solid', whiteSpace: 'nowrap' },
      input: { width: '100%', boxSizing: 'border-box', padding: '6px 9px', borderRadius: 8, border: `1px solid ${C.line}`, background: 'transparent', color: 'inherit', fontSize: 12, outline: 'none' },
      rowInput: { display: 'flex', gap: 6, alignItems: 'center' },
      btn: { padding: '6px 12px', borderRadius: 8, border: `1px solid ${C.line}`, background: 'transparent', color: 'inherit', cursor: 'pointer', fontSize: 12, whiteSpace: 'nowrap' },
      btnPrimary: { padding: '7px 14px', borderRadius: 8, border: 'none', background: C.accent, color: '#fff', cursor: 'pointer', fontSize: 12.5, fontWeight: 600, whiteSpace: 'nowrap' },
      btnDanger: { padding: '5px 10px', borderRadius: 8, border: `1px solid ${C.bad}`, background: 'transparent', color: C.bad, cursor: 'pointer', fontSize: 11.5 },
      btnDisabled: { opacity: 0.45, cursor: 'not-allowed' },
      note: { fontSize: 11.5, color: C.dim },
      err: { fontSize: 12, color: C.bad, whiteSpace: 'pre-wrap', wordBreak: 'break-word' },
      warn: { fontSize: 11.5, color: C.warn },
      // Group headers inside the parameter block, matching LLaMA-Factory's split.
      groupTitle: { fontSize: 11, fontWeight: 600, color: C.dim, letterSpacing: 0.4, marginTop: 6, paddingTop: 6, borderTop: `1px solid ${C.line}` },
      checkRow: { display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'center' },
      checkLabel: { display: 'flex', alignItems: 'center', gap: 6, fontSize: 12.5, font: 'inherit', color: C.text },
      // A collapsed dropdown that expands into checkboxes: one compact row, the
      // same footprint as the plain selects beside it.
      // The closed control, and the wrapper its popup is positioned against.
      // `flexShrink: 0` + a minimum width keep the label on one line: it sits in a
      // flex row next to the model picker, and without these the three characters
      // of 「数据集」 wrap as soon as the row gets tight.
      picker: { position: 'relative', flexShrink: 0, minWidth: 148 },
      pickerButton: { width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6, padding: '6px 9px', fontSize: 12.5, font: 'inherit', color: C.text, border: `1px solid ${C.line}`, borderRadius: 8, background: 'transparent', cursor: 'pointer', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' },
      // The popup floats over whatever is below it (absolute, full z-index) instead
      // of pushing the layout around ? expanding a dropdown should never reflow the
      // page under the user's cursor.
      // `background` must be an opaque colour: this panel draws on the host's surface,
      // and a token that does not exist (the earlier `C.bg`) leaves the popup
      // transparent, so the rows underneath show through the list.
      // Border and shadow are DSH's own popup treatment (`--dsw-elevation-panel`
      // with `--dsw-alias-border-l3`), so this list separates from the page the way
      // the host's own menus do.
      pickerPopup: { position: 'absolute', top: '100%', left: 0, marginTop: 4, minWidth: '100%', zIndex: 30, background: C.solid, color: C.solidText, font: 'inherit', fontSize: 12.5, border: '0.5px solid #0000001f', borderRadius: 8, boxShadow: '0 3px 8px 0 #00000014, 0 0 16px 0 #0000000f', maxHeight: 260, overflowY: 'auto' },
      checkList: { display: 'flex', flexDirection: 'column', gap: 6, padding: '9px 10px' },
      inputDisabled: { fontSize: 12, padding: '4px 6px', borderRadius: 5, border: `1px solid ${C.line}`, background: 'transparent', color: C.dim, opacity: 0.6 },
      okText: { fontSize: 12.5, color: C.ok, fontWeight: 600 },
      warnText: { fontSize: 12.5, color: C.warn, fontWeight: 600 },
      code: { display: 'block', fontSize: 11, padding: '6px 8px', borderRadius: 8, background: C.soft, wordBreak: 'break-all', fontFamily: 'ui-monospace,Menlo,Consolas,monospace', whiteSpace: 'pre-wrap' },
      chart: { width: '100%', height: 96, display: 'block' },
      tabs: { display: 'flex', gap: 4, borderBottom: `1px solid ${C.line}`, paddingBottom: 6 },
      tab: { padding: '3px 10px', borderRadius: 8, border: 'none', background: 'transparent', color: C.dim, cursor: 'pointer', fontSize: 12 },
      tabOn: { padding: '3px 10px', borderRadius: 8, border: 'none', background: C.soft, color: 'inherit', cursor: 'pointer', fontSize: 12, fontWeight: 600 },
      kv: { display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '2px 10px', fontSize: 11.5 },
      k: { color: C.dim },
      v: { wordBreak: 'break-word' },
      banner: { fontSize: 12, padding: '8px 10px', borderRadius: 10, border: '1px solid', whiteSpace: 'pre-wrap', display: 'flex', gap: 8, alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' },
      toast: { position: 'sticky', bottom: 0, fontSize: 11.5, padding: '6px 10px', borderRadius: 8, background: C.soft, color: C.dim },
      list: { maxHeight: 190, overflowY: 'auto' },
      grid3: { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 },
      modeRow: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 },
      modeOn: { textAlign: 'left', padding: '7px 10px', borderRadius: 10, border: `1px solid ${C.accent}`, background: C.soft, color: 'inherit', cursor: 'pointer', fontSize: 12 },
      modeOff: { textAlign: 'left', padding: '7px 10px', borderRadius: 10, border: `1px solid ${C.line}`, background: 'transparent', color: 'inherit', cursor: 'pointer', fontSize: 12, opacity: 0.75 },
      sampleCard: { border: `1px solid ${C.line}`, borderRadius: 10, padding: '8px 10px', display: 'flex', flexDirection: 'column', gap: 4 },
      excerpt: { fontSize: 11, whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 110, overflowY: 'auto', background: C.soft, borderRadius: 8, padding: '6px 8px', fontFamily: 'ui-monospace,Menlo,Consolas,monospace' },
      pairCard: { border: `1px solid ${C.line}`, borderRadius: 10, padding: '8px 10px', display: 'flex', flexDirection: 'column', gap: 6 },
      area: { width: '100%', boxSizing: 'border-box', minHeight: 64, padding: '6px 8px', borderRadius: 8, border: `1px solid ${C.line}`, background: 'transparent', color: 'inherit', fontSize: 11.5, fontFamily: 'ui-monospace,Menlo,Consolas,monospace', resize: 'vertical' },
      steps: { display: 'flex', gap: 4, flexWrap: 'wrap' },
      stepOn: { fontSize: 11, padding: '2px 8px', borderRadius: 999, background: C.accent, color: '#fff' },
      stepOff: { fontSize: 11, padding: '2px 8px', borderRadius: 999, border: `1px solid ${C.line}`, color: C.dim },
    }

    const STATE_LABEL = {
      queued: { text: '排队', color: C.dim },
      running: { text: '训练中', color: C.accent },
      succeeded: { text: '完成', color: C.ok },
      failed: { text: '失败', color: C.bad },
      cancelled: { text: '已取消', color: C.warn },
      interrupted: { text: '已中断', color: C.warn },
    }

    const MODELS = [
      ['Qwen/Qwen2.5-0.5B-Instruct', 'Qwen2.5-0.5B-Instruct'],
      ['Qwen/Qwen2.5-1.5B-Instruct', 'Qwen2.5-1.5B-Instruct'],
      ['Qwen/Qwen3-1.7B', 'Qwen3-1.7B'],
      ['Qwen/Qwen2.5-3B-Instruct', 'Qwen2.5-3B-Instruct'],
      ['meta-llama/Llama-3.2-3B-Instruct', 'Llama-3.2-3B'],
      ['Qwen/Qwen3-4B-Instruct-2507', 'Qwen3-4B-Instruct'],
      ['Qwen/Qwen2.5-7B-Instruct', 'Qwen2.5-7B-Instruct'],
    ]
    const SEQ_LENGTHS = [256, 512, 1024, 1536, 2048, 3072, 4096]
    const RANKS = [4, 8, 16, 32, 64]
    const LEARN_RATES = [
      [1e-4, '1e-4（保守）'],
      [2e-4, '2e-4（默认）'],
      [3e-4, '3e-4（激进）'],
      [5e-4, '5e-4（只给极小模型）'],
    ]
    /** Mirrors lib/training.mjs — the host revalidates, so these only shape the UI. */
    const LR_SCHEDULERS = [
      ['cosine', 'cosine（默认）'],
      ['linear', 'linear'],
      ['cosine_with_restarts', 'cosine_with_restarts'],
      ['polynomial', 'polynomial'],
      ['constant', 'constant'],
      ['constant_with_warmup', 'constant_with_warmup'],
      ['inverse_sqrt', 'inverse_sqrt'],
      ['reduce_lr_on_plateau', 'reduce_lr_on_plateau'],
    ]
    const OPTIMIZERS = [
      ['paged_adamw_8bit', 'paged AdamW 8bit（最省显存）'],
      ['adamw_8bit', 'AdamW 8bit'],
      ['paged_adamw_32bit', 'paged AdamW 32bit（更稳）'],
      ['adafactor', 'Adafactor（不需要 bitsandbytes）'],
      ['adamw_torch', 'AdamW torch（标准，最占显存）'],
    ]
    const TARGET_PRESETS = [
      ['all-linear', '全部线性层（推荐）'],
      ['attn-only', '仅注意力'],
      ['attn-mlp', '注意力 + MLP'],
    ]
    const PRECISIONS = [
      ['auto', 'auto（按显卡自动）'],
      ['bf16', 'bf16'],
      ['fp16', 'fp16'],
      ['fp32', 'fp32'],
    ]
    const MODE_INFO = {
      qlora: {
        label: 'QLoRA',
        tagline: '4bit 量化底座 + LoRA',
        why: '省显存：6 GB 上能跑 1.5B–4B。量化会带来轻微噪声。',
      },
      lora: {
        label: 'LoRA',
        tagline: 'fp16 底座 + LoRA',
        why: '质量最稳：无量化噪声。底座占两倍显存，6 GB 只够 1.5B 以下。',
      },
    }

    /** The plan object the host validates. Kept in one place so the estimate and
     *  the start call can never disagree about what is being trained.
     *  Field order follows LLaMA-Factory's grouping; see lib/training.mjs. */
    function defaultPlan() {
      return {
        mode: 'qlora',
        modelId: 'Qwen/Qwen2.5-1.5B-Instruct',
        quantType: 'nf4',
        doubleQuant: true,
        // LoRA shape
        loraR: 16,
        loraAlpha: 32,
        loraDropout: 0.05,
        loraTarget: 'all-linear',
        useRsLora: false,
        useDora: false,
        // optimisation
        maxSeqLen: 2048,
        epochs: 1,
        learningRate: 2e-4,
        lrScheduler: 'cosine',
        warmupRatio: 0.03,
        batchSize: 1,
        gradAccum: 8,
        optim: 'paged_adamw_8bit',
        weightDecay: 0,
        adamBeta2: 0.999,
        maxGradNorm: 1,
        earlyStopping: false,
        earlyStoppingPatience: 3,
        seed: 42,
        // memory
        gradientCheckpointing: true,
        precision: 'auto',
        flashAttention: false,
        deepspeed: 'none',
        // data
        packing: false,
        trainOnPrompt: false,
        maskHistory: true,
        // artifacts
        mergeAfter: false,
        exportTarget: 'none',
      }
    }

    /** Which modules a target preset resolves to; null = every linear layer. */
    function targetModulesOf(preset) {
      if (preset === 'attn-only') return ['q_proj', 'k_proj', 'v_proj', 'o_proj']
      if (preset === 'attn-mlp') return ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
      return null
    }

    /** Client-side mirror of the host validator: same refusals, same wording. */
    function validatePlan(plan) {
      if (!MODE_INFO[plan.mode]) return '训练方式只能是 qlora / lora'
      if (!/^[A-Za-z0-9._:/-]+$/.test(String(plan.modelId || ''))) return '模型名含有不安全的字符'
      if (RANKS.indexOf(Number(plan.loraR)) === -1) return `loraR 必须是 ${RANKS.join('/')} 之一`
      const seq = Number(plan.maxSeqLen)
      if (!Number.isFinite(seq) || seq < 128 || seq > 8192) return '序列长度要在 128–8192 之间'
      const epochs = Number(plan.epochs)
      if (!Number.isFinite(epochs) || epochs < 0.1 || epochs > 20) return '轮数要在 0.1–20 之间'
      const lr = Number(plan.learningRate)
      if (!Number.isFinite(lr) || lr < 1e-6 || lr > 1e-2) return '学习率要在 1e-6–1e-2 之间'
      if (plan.mode === 'qlora' && ['nf4', 'fp4'].indexOf(plan.quantType) === -1) return '量化类型只能是 nf4 / fp4'
      return null
    }

    /** The "copy prompt → paste verdicts" payload for one comparison page. */
    function buildJudgePrompt(pair, context) {
      const lines = [
        '你是模型评测的裁判。下面有若干组「同一个请求的两个回答 A/B」。请逐组判断哪个更好。',
        '',
        '判定标准（按顺序）：',
        '1. 是否给出了正确的工具调用（工具名对、参数符合意图）；',
        '2. 是否编造了不存在的工具；',
        '3. 都没有工具调用时，回答是否直接解决了请求。',
        '',
        '输出必须是 JSON，一行一组，格式：{"index": 编号, "winner": "A"|"B"|"tie", "reason": "一句话"}',
        '',
      ]
      if (context && context.knownTools && context.knownTools.length > 0) {
        lines.push(`当时可用的工具：${context.knownTools.join(', ')}`)
        lines.push('')
      }
      for (const item of pair.cases) {
        lines.push(`--- #${item.index} ---`)
        lines.push(`请求：${item.prompt}`)
        lines.push(`A：${item.answers.A}`)
        lines.push(`B：${item.answers.B}`)
        lines.push('')
      }
      return lines.join('\n')
    }

    /**
     * Parse the judge's reply into verdicts.
     *
     * The format is "one JSON object per line", and that is exactly what is parsed:
     * a reply truncated mid-object, or a line that claims to be a verdict but does
     * not parse, is counted in `unknown` rather than dropped (a judge that returned
     * 3 of 8 verdicts must not look like it returned 3 of 3).
     */
    function parseJudgeReply(text) {
      const verdicts = []
      const unknown = []
      const raw = String(text || '').replace(/```(?:json)?/g, '')
      for (const line of raw.split(/\r?\n/)) {
        const trimmed = line.trim().replace(/,+$/, '').trim()
        const looksLikeVerdict = trimmed.indexOf('"index"') !== -1 || trimmed.indexOf('index') === 1
        if (!looksLikeVerdict) continue
        let value = null
        // A line may be a complete object, or a fragment of one at either end
        // (`text {...}` / `{...},`). Try the whole line, then the braces in it.
        const candidates = [trimmed]
        const open = trimmed.indexOf('{')
        const close = trimmed.lastIndexOf('}')
        if (open !== -1 && close > open) candidates.push(trimmed.slice(open, close + 1))
        for (const candidate of candidates) {
          try {
            value = JSON.parse(candidate)
            break
          } catch { /* try the next shape */ }
        }
        if (value === null || typeof value !== 'object' || Array.isArray(value)) {
          unknown.push(trimmed.slice(0, 80))
          continue
        }
        const index = Number(value.index)
        const winner = String(value.winner || '').toLowerCase()
        if (!Number.isFinite(index) || ['a', 'b', 'tie'].indexOf(winner) === -1) {
          unknown.push(trimmed.slice(0, 80))
          continue
        }
        verdicts.push({ index, winner: winner === 'a' ? 'A' : winner === 'b' ? 'B' : 'tie', reason: String(value.reason || '') })
      }
      return { verdicts, unknown }
    }

    /** How many A-wins / B-wins / ties a set of verdicts produced for one side. */
    function tallyVerdicts(cases, verdicts) {
      let aWins = 0
      let bWins = 0
      let ties = 0
      const byIndex = {}
      for (const verdict of verdicts) byIndex[verdict.index] = verdict
      for (const item of cases) {
        const verdict = byIndex[item.index]
        if (!verdict) continue
        if (verdict.winner === 'tie') ties += 1
        else if (verdict.winner === 'A') aWins += 1
        else bWins += 1
      }
      return { aWins, bWins, ties, judged: aWins + bWins + ties }
    }

    const fmt = (n) => (typeof n === 'number' ? n.toLocaleString('en-US') : '—')
    const pct = (v) => (typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '—')
    const gb = (v) => (typeof v === 'number' ? `${v.toFixed(2)} GiB` : '—')
    const text = (v) => (typeof v === 'string' ? v : '')

    async function readJson(response) {
      const body = await response.text()
      let parsed
      try {
        parsed = JSON.parse(body)
      } catch {
        parsed = { error: body }
      }
      return parsed
    }

    async function getJson(url) {
      const response = await fetch(url, { headers: { accept: 'application/json' } })
      const body = await readJson(response)
      if (!response.ok) throw new Error(text(body && body.error) || `HTTP ${response.status}`)
      return body
    }

    async function postAction(body) {
      const response = await fetch(`${API}/action`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
      })
      const parsed = await readJson(response)
      if (!response.ok || parsed.ok === false) {
        const error = new Error(text(parsed && parsed.error) || `HTTP ${response.status}`)
        error.body = parsed
        throw error
      }
      return parsed
    }

    function lossPoints(events) {
      if (!Array.isArray(events)) return []
      const points = []
      for (const event of events) {
        if (event && event.kind === 'metric' && event.values && typeof event.values.loss === 'number') {
          points.push({ step: event.values.step, loss: event.values.loss })
        }
      }
      return points
    }

    /** The publish button is open only when there is a green verdict *and* a gguf. */
    function publishState(settings, runDetail) {
      if (!settings || settings.allowPublish !== true) return { ok: false, reason: '「允许发布模型」没有打开（插件配置 → fine-tune）。' }
      if (runDetail === null) return { ok: false, reason: '先在「训练」里选一个任务。' }
      const events = Array.isArray(runDetail.events) ? runDetail.events : []
      const evaluation = [...events].reverse().find((event) => event && event.kind === 'evaluation')
      if (evaluation === undefined) return { ok: false, reason: '还没有评测结论：评测通过之后才能发布。' }
      if (!evaluation.verdict || evaluation.verdict.status !== 'green') return { ok: false, reason: '评测结论是「不建议发布」：贴合度有回退。' }
      const artifacts = Array.isArray(runDetail.record && runDetail.record.artifacts) ? runDetail.record.artifacts : []
      if (!artifacts.some((path) => String(path).endsWith('.gguf'))) return { ok: false, reason: '这次训练还没有 .gguf 产物。' }
      return { ok: true, reason: null }
    }

    // ------------------------------------------------------------------ pieces

    function Pill(label, color) {
      return h('span', { style: { ...S.pill, color, borderColor: color } }, label)
    }

    function Section(title, right, children) {
      return h('div', { style: S.section },
        h('div', { style: S.sectionHead },
          h('div', { style: S.sectionTitle }, title),
          right === null || right === undefined ? null : h('div', { style: S.rowInput }, right)),
        children)
    }

    /**
     * A dropdown whose panel floats over the page instead of pushing it down.
     *
     * `<details>` cannot do this: its content is in normal flow, so opening it
     * reflows everything below — the user asked for a list that covers the content
     * rather than deforming the layout, which is what every other select on this
     * panel does. So the trigger and the panel are separate elements with the panel
     * absolutely positioned, and the open state is ours.
     *
     * The panel closes on a second click and on Escape. It does not close on an
     * outside click: with several options to tick, a stray click on the row below
     * should not throw away the list mid-selection.
     */
    /**
     * A dropdown whose panel floats over the page instead of pushing it down.
     *
     * `<details>` cannot do this: its content is in normal flow, so opening it
     * reflows everything below — the user asked for a list that covers the content
     * rather than deforming the layout.
     *
     * **This must be rendered as a component, never called as `Dropdown(...)`.**
     * It uses hooks, and calling it directly runs them inside the parent's render,
     * conditionally (the dropdown only exists on some tabs). React then sees a
     * different hook count between renders and throws "Rendered more hooks than
     * during the previous render", which blanked the whole panel. `h(Dropdown, …)`
     * gives it its own fiber and its own hook list.
     *
     * The panel closes on a second click and on Escape. It does not close on an
     * outside click: with several options to tick, a stray click on the row below
     * should not throw away the list mid-selection.
     */
    function Dropdown(props) {
      const { label, items } = props
      const [open, setOpen] = React.useState(false)
      React.useEffect(() => {
        if (!open) return undefined
        const onKey = (event) => { if (event.key === 'Escape') setOpen(false) }
        window.addEventListener('keydown', onKey)
        return () => window.removeEventListener('keydown', onKey)
      }, [open])
      return h('div', { style: S.picker },
        h('button', { type: 'button', style: S.pickerButton, onClick: () => setOpen((value) => !value) },
          h('span', { style: { overflow: 'hidden', textOverflow: 'ellipsis' } }, label),
          h('span', null, open ? '▴' : '▾')),
        open ? h('div', { style: S.pickerPopup }, items) : null)
    }

    /** The loss curve: the one number a trainer watches, drawn as a sparkline. */
    function LossChart(points) {
      if (!Array.isArray(points) || points.length === 0) return h('div', { style: S.note }, '还没有 loss 数据。')
      const losses = points.map((point) => point.loss).filter((value) => Number.isFinite(value))
      if (losses.length === 0) return h('div', { style: S.note }, 'loss 不是数值。')
      const width = 360
      const height = 96
      const pad = 6
      const min = Math.min.apply(null, losses)
      const max = Math.max.apply(null, losses)
      const span = max - min || 1
      const x = (index) => pad + (index / Math.max(1, losses.length - 1)) * (width - pad * 2)
      const y = (value) => height - pad - ((value - min) / span) * (height - pad * 2)
      const path = losses.map((value, index) => `${index === 0 ? 'M' : 'L'}${x(index).toFixed(1)},${y(value).toFixed(1)}`).join(' ')
      return h('div', { style: { marginTop: 4 } },
        h('svg', { viewBox: `0 0 ${width} ${height}`, style: S.chart, preserveAspectRatio: 'none' },
          h('path', { d: path, fill: 'none', stroke: C.accent, strokeWidth: 1.6, vectorEffect: 'non-scaling-stroke' })),
        h('div', { style: { display: 'flex', justifyContent: 'space-between', fontSize: 10.5, color: C.dim } },
          h('span', null, `min ${min.toFixed(4)}`),
          h('span', null, `max ${max.toFixed(4)}`),
          h('span', null, `${losses.length} 点`)))
    }

    function Metric(valueText, labelText, color) {
      return h('div', { style: S.card },
        h('div', { style: color ? { ...S.big, color } : S.big }, valueText),
        h('div', { style: S.label }, labelText))
    }

    // ------------------------------------------------------------------ panel

    function FineTunePanel() {
      const [state, setState] = React.useState(null)
      const [error, setError] = React.useState(null)
      const [toast, setToast] = React.useState('')
      const [tab, setTab] = React.useState('data')
      const [busy, setBusy] = React.useState('')

      const [selectedRun, setSelectedRun] = React.useState(null)
      const [runDetail, setRunDetail] = React.useState(null)

      const [sourceLabel, setSourceLabel] = React.useState('')
      // An array: several datasets can be trained on together.
      const [redaction, setRedaction] = React.useState('path')
      const [datasetFile, setDatasetFile] = React.useState('')
      const [lastCard, setLastCard] = React.useState(null)

      // A list: several datasets can be trained on together (mined tool calls plus a
      // hand-written domain file is the normal combination). Preview and the A/B
      // pair view stay single-dataset and use the first selection.
      const [datasetIds, setDatasetIds] = React.useState([])
      const datasetId = datasetIds[0] ?? ''
      const [estimate, setEstimate] = React.useState(null)
      const [recommendation, setRecommendation] = React.useState(null)
      // The full plan: mode + every hyperparameter. Server-validated on start;
      // mirrored locally so a bad rank is caught before a round-trip.
      const [plan, setPlan] = React.useState(defaultPlan())
      const [planError, setPlanError] = React.useState(null)
      // What this machine can actually honour (bf16 / FlashAttention / DeepSpeed /
      // bitsandbytes / GGUF). Comes from the host so the panel never has to guess
      // from GPU stats itself; `null` means "not asked yet".
      const [capability, setCapability] = React.useState(null)

      // ---- data preview
      const [preview, setPreview] = React.useState(null)
      const [previewSplit, setPreviewSplit] = React.useState('train')
      const [previewOffset, setPreviewOffset] = React.useState(0)
      const [previewBusy, setPreviewBusy] = React.useState(false)

      // ---- evaluation flow
      const [pair, setPair] = React.useState(null)
      const [pairBusy, setPairBusy] = React.useState(false)
      const [judgeText, setJudgeText] = React.useState('')
      const [pairStep, setPairStep] = React.useState(0) // 0=取题 1=贴回答 2=贴裁判 3=结论
      const [verdictSummary, setVerdictSummary] = React.useState(null)

      const mounted = React.useRef(true)

      const say = React.useCallback((message) => {
        setToast(message)
        window.setTimeout(() => {
          if (mounted.current) setToast('')
        }, 6000)
      }, [])

      const refresh = React.useCallback(async (withSessions) => {
        try {
          // No `?sessions=1`: the panel no longer lists session logs (mining was
        // removed from the UI), and that parameter makes the host walk every
        // transcript on disk for data nothing on screen renders.
        void withSessions
        const next = await getJson(`${API}/state`)
          if (!mounted.current) return
          setState(next)
          setError(null)
        } catch (err) {
          if (mounted.current) setError(err.message)
        }
      }, [])

      React.useEffect(() => {
        mounted.current = true
        refresh(true)
        // Ask once what this machine can do; the answer gates the advanced knobs
        // (FlashAttention / DeepSpeed / GGUF) and is also what the estimate uses.
        fetch(`${API}/action`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({ op: 'capability' }),
        })
          .then((response) => response.json())
          .then((payload) => {
            if (mounted.current && payload.ok !== false) setCapability(payload.capability ?? payload)
          })
          .catch(() => {
            // No capability probe only means "unknown": the panel then disables
            // the hardware-dependent knobs, which is the safe direction.
            if (mounted.current) setCapability(null)
          })
        const timer = window.setInterval(() => refresh(false), 15000)
        return () => {
          mounted.current = false
          window.clearInterval(timer)
        }
      }, [refresh])

      // Live run stream. The server replays from the start, so a reconnecting
      // page needs no client-side bookkeeping of what it has already seen.
      React.useEffect(() => {
        if (selectedRun === null) return undefined
        let source
        try {
          source = new EventSource(`${API}/stream?id=${encodeURIComponent(selectedRun)}`)
        } catch {
          return undefined
        }
        source.onmessage = (message) => {
          if (!mounted.current) return
          let payload
          try {
            payload = JSON.parse(message.data)
          } catch {
            return
          }
          if (payload.kind === 'snapshot') {
            setRunDetail({ record: payload.record, events: payload.events })
            return
          }
          if (payload.kind === 'event') {
            setRunDetail((previous) => {
              if (previous === null) return previous
              const event = payload.event
              const record = event.kind === 'state'
                ? { ...previous.record, state: event.state, metrics: { ...previous.record.metrics, ...(event.metrics || {}) } }
                : { ...previous.record, metrics: { ...previous.record.metrics, ...(event.kind === 'metric' && event.values ? event.values : {}) } }
              return { record, events: [...previous.events, event] }
            })
          }
        }
        source.onerror = () => { /* the browser retries; nothing to do */ }
        return () => {
          source.close()
        }
      }, [selectedRun])

      // One estimate per plan change; the host owns the coefficient, this only
      // sends what the form currently says.
      React.useEffect(() => {
        let cancelled = false
        const validation = validatePlan(plan)
        setPlanError(validation)
        postAction({
          op: 'gpu',
          mode: plan.mode,
          modelId: plan.modelId,
          maxSeqLen: plan.maxSeqLen,
          batchSize: plan.batchSize,
          loraR: plan.loraR,
        })
          .then((result) => {
            if (cancelled) return
            setEstimate(result.estimate)
            setRecommendation(result.recommendation)
          })
          .catch(() => { if (!cancelled) setEstimate(null) })
        return () => { cancelled = true }
      }, [plan.mode, plan.modelId, plan.maxSeqLen, plan.batchSize, plan.loraR])

      const patchPlan = (patch) => setPlan((previous) => ({ ...previous, ...patch }))

      const openPreview = async (id, split, offset) => {
        setPreviewBusy(true)
        try {
          const page = await postAction({ op: 'dataset.preview', datasetId: id, split, offset, limit: 20, chars: 220 })
          setPreview(page)
          setPreviewSplit(split)
          setPreviewOffset(offset)
        } catch (err) {
          say(`预览失败：${err.message}`)
        } finally {
          setPreviewBusy(false)
        }
      }

      const toggleSample = async (hash, restore) => {
        if (!preview) return
        try {
          const result = await postAction({ op: 'dataset.exclude', datasetId: preview.manifest.id, hash, restore })
          await openPreview(preview.manifest.id, previewSplit, previewOffset)
          say(restore ? '已恢复这条样本' : '已从训练集移除这条样本（可恢复）')
          return result
        } catch (err) {
          say(`操作失败：${err.message}`)
        }
      }

      const loadPair = async (limit) => {
        if (datasetId === '') {
          say('先在「训练」里选一个数据集')
          return
        }
        setPairBusy(true)
        try {
          const result = await postAction({ op: 'evaluation.pair', datasetId, limit: limit || 8, offset: 0 })
          const cases = result.cases.map((item) => ({
            ...item,
            answers: { base: '', candidate: '' },
          }))
          setPair({ ...result, cases })
          setJudgeText('')
          setPairStep(1)
          setVerdictSummary(null)
        } catch (err) {
          say(`取评测题失败：${err.message}`)
        } finally {
          setPairBusy(false)
        }
      }

      const saveAnswer = (index, side, value) => {
        setPair((previous) => {
          if (previous === null) return previous
          const cases = previous.cases.map((item) => (item.index === index ? { ...item, answers: { ...item.answers, [side]: value } } : item))
          return { ...previous, cases }
        })
      }

      /** The judge payload: cases whose letters are resolved to the real side. */
      const judgeCases = () => (pair === null ? [] : pair.cases.map((item) => {
        const base = item.answers.base || ''
        const candidate = item.answers.candidate || ''
        const answers = item.flip ? { A: candidate, B: base } : { A: base, B: candidate }
        return { index: item.index, prompt: item.prompt, expected: item.expected, answers, flip: item.flip }
      }))

      const copyJudgePrompt = async () => {
        const text = buildJudgePrompt({ cases: judgeCases() }, { knownTools: pair ? pair.fixture.knownTools : [] })
        try {
          await navigator.clipboard.writeText(text)
          say('评测题已复制：贴给任意模型，再把它的 JSON 判定贴回来')
          setPairStep(2)
        } catch {
          say('浏览器不让写剪贴板：选中下面的文本手动复制')
        }
      }

      const parseJudge = () => {
        const parsed = parseJudgeReply(judgeText)
        if (parsed.verdicts.length === 0) {
          say('没解析出任何判定：需要每行一个 {"index":…,"winner":"A"|"B"|"tie"}')
          return
        }
        const summary = tallyVerdicts(pair.cases, parsed.verdicts)
        // A "Wins" number is meaningless without knowing which side won: the
        // shuffle is per case, so resolve each winner back to base/candidate.
        const flipOf = {}
        for (const item of pair.cases) flipOf[item.index] = item.flip
        let candidateWins = 0
        let baseWins = 0
        for (const verdict of parsed.verdicts) {
          if (verdict.winner === 'tie') continue
          const flip = flipOf[verdict.index] === true
          const candidateWon = (verdict.winner === 'A') !== flip
          if (candidateWon) candidateWins += 1
          else baseWins += 1
        }
        setVerdictSummary({
          candidateWins,
          baseWins,
          ties: summary.ties,
          judged: summary.judged,
          candidateScore: summary.judged === 0 ? 0 : (candidateWins + summary.ties * 0.5) / summary.judged,
          baseScore: summary.judged === 0 ? 0 : (baseWins + summary.ties * 0.5) / summary.judged,
          unknown: parsed.unknown.length,
        })
        setPairStep(3)
        if (parsed.unknown.length > 0) say(`有 ${parsed.unknown.length} 行没看懂，已跳过`)
      }

      const recordEvaluation = async () => {
        if (!selectedRun || verdictSummary === null) return
        // The host's gate compares two adherence trees. The human tally is not an
        // adherence score, so the candidate side carries the *win rate* and the
        // reason says so — recording a fabricated 0.5/0.5 would be worse.
        try {
          await act('记录评测', {
            op: 'eval.record',
            id: selectedRun,
            base: { score: verdictSummary.baseScore, total: verdictSummary.judged, categories: {} },
            candidate: { score: verdictSummary.candidateScore, total: verdictSummary.judged, categories: {} },
            tolerance: 0.03,
          })
          say('评测结论已记录到任务里（来源：人工/裁判判定）')
        } catch (err) {
          say(`记录失败：${err.message}`)
        }
      }

      /**
       * Install state, polled from `env.status`.
       *
       * The panel cannot know whether the machine has a working Python stack until
       * it asks, and the answer decides whether training is even offerable ? so it
       * is fetched on mount rather than discovered by pressing ???? and reading
       * the failure.
       */
      const [envState, setEnvState] = React.useState(null)
      const [installLog, setInstallLog] = React.useState([])
      React.useEffect(() => {
        let stop = false
        const pull = async () => {
          try {
            const result = await postAction({ op: 'env.status' })
            if (!stop && result && result.ok === true) setEnvState(result)
          } catch {
            // The route is missing on an older host; the install card simply does
            // not appear.
          }
        }
        pull()
        return () => { stop = true }
      }, [])
      const startInstall = async (force) => {
        setInstallLog([{ phase: 'start', message: force ? '?????' : '?????' }])
        try {
          const result = await postAction({ op: 'env.install', confirm: true, force })
          const events = (result && result.events) || []
          setInstallLog(events)
          if (result && result.ok === true) say('??????????')
          else say(`?????${(result && result.result && result.result.reason) || '????'}`)
          const again = await postAction({ op: 'env.status' })
          if (again && again.ok === true) setEnvState(again)
        } catch (err) {
          setInstallLog((previous) => previous.concat([{ phase: 'error', message: err.message, level: 'bad' }]))
          say(`?????${err.message}`)
        }
      }

      const settings = (state && state.settings) || {}
      const runs = (state && state.runs) || []
      const datasets = (state && state.datasets) || []
      const sources = (state && state.sources) || []
      const gpu = (state && state.gpu) || null

      const act = React.useCallback(async (label, body) => {
        setBusy(label)
        try {
          const result = await postAction(body)
          await refresh(false)
          return result
        } finally {
          setBusy('')
        }
      }, [refresh])

      /**
       * Delete a dataset, after the user confirms.
       *
       * `window.confirm` rather than an inline "are you sure" row: the delete is
       * irreversible, the button sits next to a select-this-row click target, and a
       * native dialog is the one confirmation that cannot be dismissed by the same
       * stray click that opened it.
       */
      const removeDataset = async (datasetId, label) => {
        if (typeof window.confirm === 'function'
          && !window.confirm(`删除数据集 ${datasetId}（${label}？

文件会被删掉，不能恢复。`)) {
          return
        }
        try {
          await act('删除数据集', { op: 'dataset.delete', datasetId, confirm: true })
          setDatasetIds((previous) => previous.filter((id) => id !== datasetId))
          if (preview !== null && preview.manifest && preview.manifest.id === datasetId) setPreview(null)
          say(`已删除 ${datasetId}`)
        } catch (err) {
          say(`删除失败：${err.message}`)
        }
      }

      const buildFromFile = async () => {
        try {
          const sourceId = `file-${Date.now().toString(36)}`
          await act('建数据源', {
            op: 'source.create',
            source: { kind: 'dataset', id: sourceId, label: sourceLabel || '外部数据集', file: datasetFile, mode: 'answer', redaction },
          })
          const built = await act('生成数据集', { op: 'dataset.build', sourceId })
          if (built && built.dataset) {
            setLastCard(built.dataset)
            setDatasetIds((previous) => (previous.indexOf(built.dataset.id) === -1 ? previous.concat(built.dataset.id) : previous))
          }
        } catch (err) {
          say(`导入失败：${err.message}`)
        }
      }

      const startTraining = async () => {
        const invalid = validatePlan(plan)
        if (invalid !== null) {
          setPlanError(invalid)
          say(`参数有问题：${invalid}`)
          return
        }
        try {
          const result = await act('开始训练', { op: 'run.start', datasetId, datasetIds, ...plan })
          if (result && result.run) {
            setSelectedRun(result.run.id)
            setTab('train')
            say(`已开始：${result.run.id}（${result.run.input ? result.run.input.summary : plan.mode}）`)
          }
        } catch (err) {
          if (err.body && err.body.recommendation) setRecommendation(err.body.recommendation)
          say(`没有启动：${err.message}`)
        }
      }

      const cancelRun = async (id) => {
        try {
          await act('取消任务', { op: 'run.cancel', id })
          say(`已取消 ${id}`)
        } catch (err) {
          say(`取消失败：${err.message}`)
        }
      }

      /**
       * Delete a finished run and its artifacts.
       *
       * A run in flight is refused by the host, so the button only appears on a
       * finished one ? but a stale click on a just-started run must still say why
       * rather than fail silently, which is what the message below is for.
       */
      const removeRun = async (id) => {
        if (typeof window.confirm === 'function'
          && !window.confirm(`删除训练任务 ${id}？\n\n任务记录和训练产物（adapter）都会被删掉，不能恢复。`)) {
          return
        }
        try {
          await act('删除任务', { op: 'run.delete', id, confirm: true })
          if (selectedRun === id) { setSelectedRun(null); setRunDetail(null) }
          say(`已删除 ${id}`)
        } catch (err) {
          say(`删除失败：${err.message}`)
        }
      }

      const openRun = (id) => {
        setSelectedRun(id === selectedRun ? null : id)
        if (id !== selectedRun) setRunDetail(null)
      }

      const preparePublish = async () => {
        try {
          const prepared = await act('发布准备', { op: 'publish.prepare', id: selectedRun })
          say(`可以发布。在终端执行：${prepared.command}`)
        } catch (err) {
          say(`还不能发布：${err.message}`)
        }
      }

      if (error !== null) {
        return h('div', { style: S.root },
          h('div', { style: { ...S.banner, color: C.bad, borderColor: C.bad } }, `连不上插件后端：${error}`),
          h('button', { style: S.btn, onClick: () => refresh(true) }, '重试'))
      }
      if (state === null) return h('div', { style: S.root }, h('div', { style: S.note }, '加载中…'))

      const gpuLine = gpu === null
        ? '显卡：探测中…'
        : gpu.available
          ? `显卡：${gpu.gpus.map((item) => `${item.name} ${item.totalGiB} GiB`).join(' / ')}`
          : `显卡：不可用（${text(gpu.reason) || '未知原因'}）`

      const tabs = [['data', '数据'], ['train', '训练'], ['evaluate', '评测与发布']]

      return h('div', { style: S.root },
        h('div', { style: S.head },
          h('div', { style: S.title }, '模型微调'),
          h('button', { style: busy === '' ? S.btn : { ...S.btn, ...S.btnDisabled }, disabled: busy !== '', onClick: () => refresh(true) }, '刷新')),
        h('div', { style: S.note }, gpuLine),

        // The install card. Offered only when this host has the route and no ready
        // venv: a user whose environment is already built never sees it, and one who
        // needs it is told what will be downloaded before anything starts.
        envState !== null && envState.ready === null
          ? h('div', { style: { ...S.banner, borderColor: C.accent } },
            h('span', null, '???? Python ???torch ??? 3?5 GB?????????? '
              + (envState.venvPython || '') + ' ???????????'),
            h('button', {
              style: busy === '' ? S.btnPrimary : { ...S.btnPrimary, ...S.btnDisabled },
              disabled: busy !== '',
              onClick: () => startInstall(false),
            }, installLog.length === 0 ? '????' : '????'))
          : null,
        installLog.length === 0
          ? null
          : h('div', { style: { ...S.list, maxHeight: 200, overflowY: 'auto' } },
            installLog.map((event, index) => h('div', {
              key: `${index}-${event.phase}`,
              style: event.level === 'bad' ? S.err : event.level === 'warn' ? S.warnText : S.note,
            }, `${event.phase}: ${event.message}`))),

        settings.allowSpawn !== true
          ? h('div', { style: { ...S.banner, color: C.warn, borderColor: C.warn } },
            h('span', null, '「允许启动训练进程」没打开：数据可以准备，训练不能开始。'),
            h('span', { style: S.note }, '插件配置 → fine-tune → 允许启动训练进程'))
          : null,

        h('div', { style: S.tabs }, tabs.map(([key, label]) =>
          h('button', { key, style: tab === key ? S.tabOn : S.tab, onClick: () => setTab(key) }, label))),

        // ---------------------------------------------------------- 1. data
        tab === 'data'
          ? Section('导入数据集文件',
            // The `right` slot of a Section renders on the heading row, aligned to
            // the right — that is the strip above the import field, not inside it.
            h('button', {
              style: datasetIds.length === 0
                ? { ...S.btnPrimary, ...S.btnDisabled }
                : S.btnPrimary,
              disabled: datasetIds.length === 0,
              onClick: () => setTab('train'),
              title: datasetIds.length === 0 ? '先勾选一个数据集' : '',
            }, '下一步'),
            h('div', { style: S.rowInput },
              h('input', { style: S.input, placeholder: 'C:\\path\\to\\data.jsonl（每行 {messages:[…]} 或 {prompt, response}）', value: datasetFile, onChange: (event) => setDatasetFile(event.target.value) }),
              h('button', {
                style: datasetFile === '' || busy !== '' ? { ...S.btn, ...S.btnDisabled } : S.btn,
                disabled: datasetFile === '' || busy !== '',
                onClick: buildFromFile,
              }, '导入')))
          : null,

        tab === 'data' && lastCard !== null ? DataCard(lastCard) : null,

        // The dataset list. Multi-select: several datasets can be trained on
        // together, and the training run concatenates them per split.
        tab === 'data'
          ? Section('数据集', h('div', { style: S.note }, datasetIds.length === 0
            ? '勾选一个或多个数据集作为这次训练的数据。'
            : `已选 ${datasetIds.length} 个，训练时会合并成一个。`),
            datasets.length === 0
              ? h('div', { style: S.note }, '还没有数据集。上面导入一个文件，或用插件自带的数据集。')
              : datasets.map((dataset, index, list) => {
                const picked = datasetIds.indexOf(dataset.id) !== -1
                const toggle = () => setDatasetIds((previous) => (picked
                  ? previous.filter((id) => id !== dataset.id)
                  : previous.concat(dataset.id)))
                return h('div', {
                  key: dataset.id,
                  style: { ...(index === list.length - 1 ? S.rowLast : S.row), ...S.clickable },
                  onClick: toggle,
                  title: dataset.id,
                },
                  h('input', { type: 'checkbox', checked: picked, onChange: toggle, onClick: (event) => event.stopPropagation() }),
                  h('span', { style: S.name }, dataset.id),
                  h('span', { style: S.num }, `train ${fmt(dataset.counts.train)} · val ${fmt(dataset.counts.val)} · eval ${fmt(dataset.counts.eval)}`),
                  // Last in the row, so the destructive action sits away from the
                  // counts a user reads and never in the path of the checkbox.
                  h('button', {
                    style: S.btnDanger,
                    onClick: (event) => { event.stopPropagation(); removeDataset(dataset.id, `train ${fmt(dataset.counts.train)}`) },
                    title: '删除这个数据集（不可恢复）',
                  }, '删除'))
              }))
          : null,

        // ---------------------------------------------------------- 2. train
        tab === 'train'
          ? Section('训练计划',
            h('button', {
              style: busy !== '' || datasetIds.length === 0 || settings.allowSpawn !== true || planError !== null ? { ...S.btnPrimary, ...S.btnDisabled } : S.btnPrimary,
              disabled: busy !== '' || datasetIds.length === 0 || settings.allowSpawn !== true || planError !== null,
              onClick: startTraining,
            }, '开始训练'),
            h('div', { style: { display: 'flex', flexDirection: 'column', gap: 8 } },
              h('div', { style: S.rowInput },
                // Same multi-select as the data tab, so the choice can be adjusted
                // without leaving the training plan.
                h(Dropdown, { label: datasetIds.length === 0
                    ? '数据集'
                    : `数据集：${datasetIds.length} 个（${datasetIds[0]}${datasetIds.length > 1 ? ' +…' : ''}）`, items: datasets.map((dataset) => {
                    const picked = datasetIds.indexOf(dataset.id) !== -1
                    return h('label', { key: dataset.id, style: S.checkLabel, title: `${dataset.id}　train ${fmt(dataset.counts.train)}` },
                      h('input', {
                        type: 'checkbox',
                        checked: picked,
                        onChange: () => setDatasetIds((previous) => (picked
                          ? previous.filter((id) => id !== dataset.id)
                          : previous.concat(dataset.id))),
                      }),
                      h('span', null, dataset.id))
                  }) }),
                h('select', { style: { ...S.input, minWidth: 0, flex: '1 1 0' }, value: plan.modelId, onChange: (event) => patchPlan({ modelId: event.target.value }) },
                  MODELS.map(([id, label]) => h('option', { key: id, value: id }, label)))),

              // ---- mode: the one choice that decides whether 6 GB is enough
              h('div', { style: S.modeRow }, Object.keys(MODE_INFO).map((key) => h('button', {
                key,
                style: plan.mode === key ? S.modeOn : S.modeOff,
                onClick: () => patchPlan({ mode: key }),
              },
                h('div', { style: { fontWeight: 600 } }, `${MODE_INFO[key].label}`),
                h('div', { style: { fontSize: 10.5, color: C.dim } }, MODE_INFO[key].tagline)))),
              h('div', { style: S.note }, MODE_INFO[plan.mode].why),

              // ---- hyperparameters, grouped the way LLaMA-Factory groups them.
              // Every knob is rendered; one the machine cannot honour is disabled
              // with the reason printed next to it (see capabilityOf in the host).

              // 3) optimisation
              h('div', { style: S.groupTitle }, '优化'),
              h('div', { style: S.grid3 },
                Field('序列长度', h('select', { style: S.input, value: String(plan.maxSeqLen), onChange: (event) => patchPlan({ maxSeqLen: Number(event.target.value) }) },
                  SEQ_LENGTHS.map((len) => h('option', { key: len, value: String(len) }, String(len))))),
                Field('轮数', h('input', { style: S.input, type: 'number', min: 0.1, max: 20, step: 0.5, value: plan.epochs, onChange: (event) => patchPlan({ epochs: Number(event.target.value) }) })),
                Field('学习率', h('select', { style: S.input, value: String(plan.learningRate), onChange: (event) => patchPlan({ learningRate: Number(event.target.value) }) },
                  LEARN_RATES.map(([value, label]) => h('option', { key: label, value: String(value) }, label))))),
              h('div', { style: S.grid3 },
                Field('学习率调度', h('select', { style: S.input, value: plan.lrScheduler, onChange: (event) => patchPlan({ lrScheduler: event.target.value }) },
                  LR_SCHEDULERS.map(([id, label]) => h('option', { key: id, value: id }, label)))),
                Field('预热比例', h('input', { style: S.input, type: 'number', min: 0, max: 0.3, step: 0.01, value: plan.warmupRatio, onChange: (event) => patchPlan({ warmupRatio: Number(event.target.value) }) })),
                Field('梯度累积', h('select', { style: S.input, value: String(plan.gradAccum), onChange: (event) => patchPlan({ gradAccum: Number(event.target.value) }) },
                  [1, 2, 4, 8, 16, 32].map((n) => h('option', { key: n, value: String(n) }, String(n)))))),
              h('div', { style: S.grid3 },
                Field('batch', h('select', { style: S.input, value: String(plan.batchSize), onChange: (event) => patchPlan({ batchSize: Number(event.target.value) }) },
                  [1, 2, 4, 8].map((size) => h('option', { key: size, value: String(size) }, String(size))))),
                Field('优化器', h('select', { style: S.input, value: plan.optim, onChange: (event) => patchPlan({ optim: event.target.value }) },
                  OPTIMIZERS.map(([id, label]) => h('option', { key: id, value: id }, label)))),
                Field('权重衰减', h('input', { style: S.input, type: 'number', min: 0, max: 1, step: 0.01, value: plan.weightDecay, onChange: (event) => patchPlan({ weightDecay: Number(event.target.value) }) }))),
              h('div', { style: S.grid3 },
                Field('梯度裁剪', h('input', { style: S.input, type: 'number', min: 0, max: 100, step: 0.1, value: plan.maxGradNorm, onChange: (event) => patchPlan({ maxGradNorm: Number(event.target.value) }) })),
                Field('Adam β2', h('input', { style: S.input, type: 'number', min: 0.9, max: 0.9999, step: 0.0001, value: plan.adamBeta2, onChange: (event) => patchPlan({ adamBeta2: Number(event.target.value) }) })),
                Field('随机种子', h('input', { style: S.input, type: 'number', min: 0, max: 1000000, step: 1, value: plan.seed, onChange: (event) => patchPlan({ seed: Number(event.target.value) }) }))),

              // 2) LoRA structure
              h('div', { style: S.groupTitle }, 'LoRA 结构'),
              h('div', { style: S.grid3 },
                Field('LoRA r', h('select', { style: S.input, value: String(plan.loraR), onChange: (event) => patchPlan({ loraR: Number(event.target.value), loraAlpha: Number(event.target.value) * 2 }) },
                  RANKS.map((rank) => h('option', { key: rank, value: String(rank) }, String(rank))))),
                Field('LoRA α', h('input', { style: S.input, type: 'number', min: 1, max: 512, step: 1, value: plan.loraAlpha, onChange: (event) => patchPlan({ loraAlpha: Number(event.target.value) }) })),
                Field('dropout', h('input', { style: S.input, type: 'number', min: 0, max: 0.5, step: 0.01, value: plan.loraDropout, onChange: (event) => patchPlan({ loraDropout: Number(event.target.value) }) }))),
              h('div', { style: S.grid2 },
                Field('挂载位置', h('select', { style: S.input, value: plan.loraTarget, onChange: (event) => patchPlan({ loraTarget: event.target.value }) },
                  TARGET_PRESETS.map(([id, label]) => h('option', { key: id, value: id }, label)))),
                plan.mode === 'qlora'
                  ? Field('量化', h('select', { style: S.input, value: plan.quantType, onChange: (event) => patchPlan({ quantType: event.target.value }) },
                    h('option', { value: 'nf4' }, 'NF4'),
                    h('option', { value: 'fp4' }, 'FP4')))
                  : Field('量化', h('div', { style: S.note }, 'LoRA 用 fp16 底座，不量化'))),
              h('div', { style: S.checkRow },
                Check('rsLoRA（r 变化时保持更新量稳定）', plan.useRsLora, (v) => patchPlan({ useRsLora: v })),
                Check('DoRA（质量略好，更占显存）', plan.useDora, (v) => patchPlan({ useDora: v }))),

              // 5) memory — the knobs that decide whether a 6 GB card succeeds
              h('div', { style: S.groupTitle }, '显存与精度'),
              h('div', { style: S.grid3 },
                Field('计算精度', h('select', { style: S.input, value: plan.precision, onChange: (event) => patchPlan({ precision: event.target.value }) },
                  PRECISIONS.map(([id, label]) => h('option', { key: id, value: id }, label)))),
                Field('梯度检查点', h('select', { style: S.input, value: plan.gradientCheckpointing ? 'on' : 'off', onChange: (event) => patchPlan({ gradientCheckpointing: event.target.value === 'on' }) },
                  h('option', { value: 'on' }, '开（省显存）'),
                  h('option', { value: 'off' }, '关（更快）'))),
                Field('FlashAttention-2', h('select', {
                  style: capability !== null && capability.features.flashAttention2.ok ? S.input : S.inputDisabled,
                  value: plan.flashAttention ? 'on' : 'off',
                  disabled: capability === null || !capability.features.flashAttention2.ok,
                  onChange: (event) => patchPlan({ flashAttention: event.target.value === 'on' }),
                },
                  h('option', { value: 'off' }, '关'),
                  h('option', { value: 'on' }, '开')))),
              capability !== null && !capability.features.flashAttention2.ok
                ? h('div', { style: S.note }, `FlashAttention-2 不可用：${capability.features.flashAttention2.why}`)
                : null,
              capability !== null && !capability.features.bitsandbytes.ok && plan.mode === 'qlora'
                ? h('div', { style: S.warn }, 'bitsandbytes 未安装：QLoRA 需要它。装好后重启，或改用 LoRA。')
                : null,

              // 4) data handling
              h('div', { style: S.groupTitle }, '数据'),
              h('div', { style: S.checkRow },
                Check('样本打包（短样本拼成长序列，提升吞吐）', plan.packing, (v) => patchPlan({ packing: v })),
                Check('只对回答计算损失（推荐）', plan.maskHistory, (v) => patchPlan({ maskHistory: v }))),
              plan.packing
                ? h('div', { style: S.note }, '打包会把多条样本接到一个序列里。工具调用类样本（每条一个动作）建议关闭。')
                : null,

              // early stopping — lives with optimisation but reads better here
              h('div', { style: S.checkRow },
                Check('早停（验证损失不再下降就停）', plan.earlyStopping, (v) => patchPlan({ earlyStopping: v })),
                plan.earlyStopping
                  ? Field('耐心轮数', h('input', { style: S.input, type: 'number', min: 1, max: 20, step: 1, value: plan.earlyStoppingPatience, onChange: (event) => patchPlan({ earlyStoppingPatience: Number(event.target.value) }) }))
                  : null),

              // 6) artifacts
              h('div', { style: S.groupTitle }, '产物'),
              h('div', { style: S.grid2 },
                Field('训练后合并', h('select', { style: S.input, value: plan.mergeAfter ? 'on' : 'off', onChange: (event) => patchPlan({ mergeAfter: event.target.value === 'on' }) },
                  h('option', { value: 'off' }, '不合并（只留适配器，推荐）'),
                  h('option', { value: 'on' }, '合并成完整权重'))),
                Field('导出', h('select', { style: S.input, value: plan.exportTarget, onChange: (event) => patchPlan({ exportTarget: event.target.value }) },
                  h('option', { value: 'none' }, '不导出'),
                  h('option', { value: 'gguf' }, 'GGUF（Ollama / llama.cpp）'),
                  h('option', { value: 'ollama' }, 'Ollama')))),
              plan.exportTarget === 'gguf' && capability !== null && !capability.features.gguf.ok
                ? h('div', { style: S.note }, `GGUF 导出需要 llama.cpp：${capability.features.gguf.why}`)
                : null,

              planError !== null ? h('div', { style: S.err }, planError) : null,
              estimate === null
                ? h('div', { style: S.note }, '显存估算中…')
                : h('div', { style: S.grid2 },
                  Metric(gb(estimate.totalGiB), `${text(estimate.modeLabel) || plan.mode} 预计峰值`),
                  Metric(estimate.headroomGiB === null ? '未知' : gb(estimate.headroomGiB), estimate.measured ? '余量（有实测）' : '余量', estimate.fits === false ? C.bad : C.ok)),
              estimate !== null && estimate.fits === false && recommendation !== null
                ? h('div', { style: { ...S.banner, color: C.bad, borderColor: C.bad } },
                  h('span', null, `放不下：${text(recommendation.reason)}`),
                  recommendation.action === 'maxSeqLen'
                    ? h('button', { style: S.btn, onClick: () => patchPlan({ maxSeqLen: recommendation.suggest }) }, `改成 seq ${recommendation.suggest}`)
                    : recommendation.action === 'mode'
                      ? h('button', { style: S.btn, onClick: () => patchPlan({ mode: 'qlora' }) }, '换成 QLoRA')
                      : null)
                : null,
              estimate !== null
                ? h('div', { style: S.note }, Object.keys(estimate.components).map((key) => `${key} ${estimate.components[key]}`).join(' · ') + ' GiB')
                : null))
          : null,

        tab === 'train'
          ? Section('训练任务', runs.length === 0 ? null : h('span', { style: S.note }, `${runs.length} 条`),
            runs.length === 0
              ? h('div', { style: S.note }, '还没有训练任务。')
              : runs.map((run, index, list) => {
                const status = STATE_LABEL[run.state] || { text: run.state, color: C.dim }
                return h('div', {
                  key: run.id,
                  style: { ...(index === list.length - 1 ? S.rowLast : S.row), ...S.clickable },
                  onClick: () => openRun(run.id),
                },
                  Pill(status.text, status.color),
                  h('span', { style: S.name, title: run.id }, `${run.id}  ${text(run.input && run.input.modelId)}`),
                  h('span', { style: S.num }, run.metrics && typeof run.metrics.loss === 'number' ? `loss ${run.metrics.loss.toFixed(4)}` : '—'),
                  run.state === 'running' || run.state === 'queued'
                    ? h('button', { style: S.btnDanger, onClick: (event) => { event.stopPropagation(); cancelRun(run.id) } }, '取消')
                    : h('button', {
                      style: S.btnDanger,
                      onClick: (event) => { event.stopPropagation(); removeRun(run.id) },
                      title: '删除这个任务及其产物（不可恢复）',
                    }, '删除'))
              }))
          : null,

        tab === 'train' && runDetail !== null ? RunDetail(runDetail) : null,

        tab === 'train'
          ? Section('样本预览',
            h('button', { style: S.btn, disabled: previewBusy || datasetIds.length === 0, onClick: () => openPreview(datasetId, previewSplit, 0) }, previewBusy ? '读取中…' : '预览'),
            h('div', { style: { display: 'flex', flexDirection: 'column', gap: 8 } },
              h('div', { style: S.rowInput },
                h('select', { style: S.input, value: previewSplit, onChange: (event) => openPreview(datasetId, event.target.value, 0) },
                  h('option', { value: 'train' }, 'train'),
                  h('option', { value: 'val' }, 'val'),
                  h('option', { value: 'eval' }, 'eval')),
                h('span', { style: S.note }, datasetId === '' ? '先选数据集' : datasetId)),
              preview === null
                ? h('div', { style: S.note }, '点「预览」把样本一条条看清楚：提示里是什么情景、标签里是什么动作。看到坏样本可以点「移除」，它会立刻从训练集消失（可恢复）。')
                : h('div', { style: { display: 'flex', flexDirection: 'column', gap: 6 } },
                  h('div', { style: S.note }, `${previewSplit}：共 ${fmt(preview.total)} 条，显示 ${preview.start + 1}–${Math.min(preview.total, preview.start + preview.rows.length)}；已移除 ${(preview.excluded || []).length} 条`),
                  preview.rows.map((row) => SampleCard(row, () => toggleSample(row.hash, false))),
                  preview.total > preview.start + preview.rows.length || preview.start > 0
                    ? h('div', { style: S.rowInput },
                      h('button', { style: preview.start === 0 ? { ...S.btn, ...S.btnDisabled } : S.btn, disabled: preview.start === 0, onClick: () => openPreview(preview.manifest.id, previewSplit, Math.max(0, preview.start - 20)) }, '上一页'),
                      h('button', { style: preview.total <= preview.start + 20 ? { ...S.btn, ...S.btnDisabled } : S.btn, disabled: preview.total <= preview.start + 20, onClick: () => openPreview(preview.manifest.id, previewSplit, preview.start + 20) }, '下一页'),
                      previewSplit === 'train'
                        ? h('button', { style: S.btn, onClick: () => { preview.rows.forEach((row) => toggleSample(row.hash, false)) } }, '移除本页全部')
                        : null)
                    : null,
                  (preview.excluded || []).length > 0
                    ? h('details', null,
                      h('summary', { style: S.note }, `已移除的样本（${preview.excluded.length}）`),
                      (preview.excluded || []).map((hash) => h('div', { key: hash, style: S.rowInput },
                        h('span', { style: { ...S.code, flex: 1 } }, hash),
                        h('button', { style: S.btn, onClick: () => toggleSample(hash, true) }, '恢复'))))
                    : null)))
          : null,

        // ---------------------------------------------------------- 3. evaluate
        tab === 'evaluate'
          ? Section('对比评测',
            h('button', { style: S.btn, disabled: pairBusy || datasetIds.length === 0, onClick: () => loadPair(6) }, pairBusy ? '取题中…' : '取 6 道评测题'),
            h('div', { style: { display: 'flex', flexDirection: 'column', gap: 8 } },
              h('div', { style: S.note }, selectedRun === null ? '（可选）先在「训练」里选一个任务，结论会记到它上面。' : `当前任务：${selectedRun}`),
              h('div', { style: S.note },
                '评测不打主观分，只量两件可判定的事：',
                h('br'), '1. 贴合度——在 eval 集上，输出是否调用了对的工具、参数是否一致、有没有编造工具名；',
                h('br'), '2. 偏好——同一批样本上新旧模型谁更好，由你配置的裁判判定。',
                h('br'), '贴合度比底座低超过 3 个百分点，发布按钮就会关上。'),
              pair === null
                ? null
                : h('div', { style: { display: 'flex', flexDirection: 'column', gap: 8 } },
                  h('div', { style: S.steps },
                    ['1 取题', '2 填回答', '3 贴裁判', '4 结论'].map((label, index) =>
                      h('span', { key: label, style: index === pairStep ? S.stepOn : S.stepOff }, label))),

                  pairStep === 1
                    ? h('div', { style: { display: 'flex', flexDirection: 'column', gap: 6 } },
                      h('div', { style: S.note }, `共 ${pair.total} 道题，这页 ${pair.cases.length} 道。左右两列各填一份回答（谁是新模型不重要，面板会在裁判提示里随机标成 A/B）。`),
                      pair.cases.map((item) => h('div', { key: item.index, style: S.pairCard },
                        h('div', { style: S.label }, `#${item.index}`),
                        h('div', { style: { ...S.code, maxHeight: 90, overflow: 'auto' } }, item.prompt),
                        h('div', { style: S.grid2 },
                          h('textarea', { style: S.area, placeholder: '新模型的回答（微调后）', value: item.answers.candidate, onChange: (event) => saveAnswer(item.index, 'candidate', event.target.value) }),
                          h('textarea', { style: S.area, placeholder: '旧模型的回答（底座）', value: item.answers.base, onChange: (event) => saveAnswer(item.index, 'base', event.target.value) })))),
                      h('div', { style: S.rowInput },
                        h('button', { style: S.btnPrimary, onClick: copyJudgePrompt }, '复制裁判提示词'),
                        h('span', { style: S.note }, pair.fixture.knownTools.length > 0 ? `会带上「当时可用工具」列表（${pair.fixture.knownTools.length} 个）` : '没有工具注册表记录：裁判只能靠回答内容判断')))
                    : null,

                  pairStep === 2
                    ? h('div', { style: { display: 'flex', flexDirection: 'column', gap: 6 } },
                      h('div', { style: S.note }, '把提示词贴给任意模型（或用你自己的判断），把它的 JSON 判定整段贴回下面：'),
                      h('textarea', { style: { ...S.area, minHeight: 120 }, placeholder: '{"index": 0, "winner": "A", "reason": "…"}', value: judgeText, onChange: (event) => setJudgeText(event.target.value) }),
                      h('div', { style: S.rowInput },
                        h('button', { style: S.btnPrimary, disabled: judgeText.trim() === '', onClick: parseJudge }, '解析判定'),
                        h('button', { style: S.btn, onClick: () => setPairStep(1) }, '回到回答')))
                    : null,

                  pairStep === 3 && verdictSummary !== null
                    ? h('div', { style: { display: 'flex', flexDirection: 'column', gap: 6 } },
                      h('div', { style: S.grid3 },
                        Metric(String(verdictSummary.candidateWins), '微调后赢', C.ok),
                        Metric(String(verdictSummary.baseWins), '底座赢', C.warn),
                        Metric(String(verdictSummary.ties), '平')),
                      h('div', { style: S.note }, `共判定 ${verdictSummary.judged} 组${verdictSummary.unknown ? `，${verdictSummary.unknown} 行没看懂` : ''}。`),
                      h('div', { style: S.rowInput },
                        h('button', { style: S.btnPrimary, disabled: selectedRun === null, onClick: recordEvaluation }, '把结论记到任务里'),
                        h('button', { style: S.btn, onClick: () => setPairStep(1) }, '改回答')))
                    : null)))
          : null,

        tab === 'evaluate' && runDetail !== null ? Verdict(runDetail) : null,

        tab === 'evaluate'
          ? (() => {
            const publish = publishState(settings, runDetail)
            return Section('发布到 Ollama',
              h('button', {
                style: busy !== '' || !publish.ok ? { ...S.btnPrimary, ...S.btnDisabled } : S.btnPrimary,
                disabled: busy !== '' || !publish.ok,
                onClick: preparePublish,
              }, '发布'),
              h('div', { style: S.note },
                publish.ok
                  ? '评测通过。点「发布」会得到一条 ollama create 命令，面板不替你在后台写任何东西。'
                  : publish.reason))
          })()
          : null,

        toast === '' ? null : h('div', { style: S.toast }, toast))
    }

    /** A labelled form field: label above, control below, both in one column. */
    function Field(label, control) {
      return h('div', { style: { display: 'flex', flexDirection: 'column', gap: 3 } },
        h('div', { style: S.label }, label),
        control)
    }

    /** A labelled checkbox; `onChange` gets the new boolean. */
    function Check(label, checked, onChange) {
      return h('label', { style: S.checkLabel },
        h('input', { type: 'checkbox', checked, onChange: (event) => onChange(event.target.checked) }),
        h('span', null, label))
    }

    /** One sample: the situation it teaches, the action it labels, and a remove button. */
    function SampleCard(row, onRemove) {
      const meta = row.meta || {}
      return h('div', { key: row.hash || String(meta.sessionId), style: S.sampleCard },
        h('div', { style: { display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 } },
          h('span', { style: { ...S.pill, color: C.accent, borderColor: C.accent } }, text(meta.origin) || 'sample'),
          h('span', { style: S.num }, `${fmt(row.promptChars)} → ${fmt(row.labelChars)} 字符`),
          h('span', { style: { flex: 1 } }),
          h('button', { style: S.btnDanger, onClick: onRemove }, '移除')),
        h('div', { style: S.label }, '提示（情景）'),
        h('div', { style: S.excerpt }, row.prompt),
        h('div', { style: S.label }, '标签（应做的动作）'),
        h('div', { style: S.excerpt }, row.label))
    }

    /** The data card: every number a reader needs to judge the data before training. */
    function DataCard(card) {
      const counts = card.counts || {}
      const tokens = card.tokens || {}
      const drops = card.provenance && card.provenance.dropped
      const redactions = card.provenance && card.provenance.redactions
      const redactionTotal = redactions ? Object.keys(redactions).reduce((sum, key) => sum + (redactions[key] || 0), 0) : 0
      const rows = [
        ['原始样本', fmt(counts.raw)],
        ['保留（去重去坏后）', fmt(counts.kept)],
        ['重复', fmt(counts.duplicates)],
        ['无效', fmt(counts.invalid)],
        ['train / val / eval', `${fmt(counts.train)} / ${fmt(counts.val)} / ${fmt(counts.eval)}`],
        ['分组（按会话）', fmt(counts.groups)],
        ['token 中位数 / P90 / 最大', `${fmt(tokens.median)} / ${fmt(tokens.p90)} / ${fmt(tokens.max)}`],
        ['脱敏命中', fmt(redactionTotal)],
      ]
      return Section(`数据卡：${card.id}`,
        h('span', { style: S.note }, card.sourceKind === 'agent' ? '会话日志' : '外部文件'),
        h('div', { style: { display: 'flex', flexDirection: 'column', gap: 6 } },
          h('div', { style: S.kv }, rows.map(([key, value]) => [
            h('span', { key: `${key}-k`, style: S.k }, key),
            h('span', { key: `${key}-v`, style: S.v }, value),
          ])),
          drops
            ? h('div', { style: S.note }, `丢弃：幻觉工具名 ${fmt(drops.hallucinated_tool)} · 失败调用 ${fmt(drops.failed_call)} · 没有可判定回应 ${fmt(drops.no_context)} · 超长标签 ${fmt(drops.oversized_label)}`)
            : null,
          counts.kept === 0
            ? h('div', { style: S.err }, '这个数据集是空的：会话里可能没有成功的工具调用，试试切到「学回答」。')
            : counts.kept < 20
              ? h('div', { style: S.note }, '样本少于 20 条：流程能跑通，但训出来的差别大概率落在噪声里。')
              : null))
    }

    function RunDetail(detail) {
      const record = detail.record
      const metrics = record.metrics || {}
      const status = STATE_LABEL[record.state] || { text: record.state, color: C.dim }
      const logs = detail.events.filter((event) => event.kind === 'log' || event.kind === 'error').slice(-40)
      return Section(`实时：${record.id}`, Pill(status.text, status.color),
        h('div', null,
          LossChart(lossPoints(detail.events)),
          h('div', { style: S.kv },
            h('span', { style: S.k }, '步骤'), h('span', { style: S.v }, fmt(metrics.step)),
            h('span', { style: S.k }, '可训练参数'), h('span', { style: S.v }, fmt(metrics.trainableParams)),
            h('span', { style: S.k }, '显存峰值'), h('span', { style: S.v }, typeof metrics.peakGiB === 'number' ? gb(metrics.peakGiB) : '—'),
            h('span', { style: S.k }, '产物'), h('span', { style: S.v }, (record.artifacts || []).join('  ') || '—'),
            record.error ? h('span', { style: S.k }, '错误') : null,
            record.error ? h('span', { style: S.err }, record.error.message) : null),
          logs.length === 0
            ? null
            : h('div', { style: { ...S.list, marginTop: 6 } },
              logs.map((event, index) => h('div', { key: `${event.seq}-${index}`, style: event.kind === 'error' ? S.err : S.note }, text(event.message))))))
    }

    function Verdict(detail) {
      const events = Array.isArray(detail.events) ? detail.events : []
      const evaluation = [...events].reverse().find((event) => event && event.kind === 'evaluation')
      if (evaluation === undefined) return Section('结论', null, h('div', { style: S.note }, '还没有评测记录。'))
      const green = evaluation.verdict && evaluation.verdict.status === 'green'
      const reasons = (evaluation.verdict && evaluation.verdict.reasons) || []
      return Section('结论', null,
        h('div', { style: { display: 'flex', flexDirection: 'column', gap: 6 } },
          h('div', { style: green ? S.okText : S.warnText }, green ? '✅ 可以发布' : '⛔ 不建议发布'),
          h('div', { style: S.kv },
            h('span', { style: S.k }, '贴合度：底座'), h('span', { style: S.v }, pct(evaluation.base && evaluation.base.score)),
            h('span', { style: S.k }, '贴合度：微调后'), h('span', { style: S.v }, pct(evaluation.candidate && evaluation.candidate.score))),
          reasons.map((reason) => h('div', { key: reason, style: S.note }, `· ${reason}`))))
    }

    function FineTuneIcon() {
      return h('svg', { width: 18, height: 18, viewBox: '0 0 18 18', fill: 'none', 'aria-hidden': 'true' },
        h('path', { d: 'M2.6 12.8c2.2 0 3.2-3.4 4.8-3.4s2.4 3.4 4.6 3.4 2.9-2.2 3.4-3.4', stroke: 'currentColor', strokeWidth: 1.4, strokeLinecap: 'round' }),
        h('path', { d: 'M2.6 15.2h12.8', stroke: 'currentColor', strokeWidth: 1.2, strokeLinecap: 'round', opacity: 0.55 }),
        h('circle', { cx: 13.4, cy: 5.2, r: 1.5, stroke: 'currentColor', strokeWidth: 1.3 }))
    }

    function apply(ctx) {
      try {
        if (!ctx || ctx.slots === undefined || ctx.slots === null) {
          console.error('fine-tune: this DSH has no slots service; the panel stays unmounted')
          return
        }
        ctx.slots.inject('main', () =>
          ctx.slots.inject('sidebar.panellist', () => {
            const stopMain = ctx.slots.register({ name: 'main', key: PANEL_ID }, FineTunePanel)
            const stopIcon = ctx.slots.register(
              { name: 'sidebar.panellist', id: PANEL_ID, order: 77, label: () => '模型微调' },
              FineTuneIcon)
            return () => {
              if (typeof stopIcon === 'function') stopIcon()
              if (typeof stopMain === 'function') stopMain()
            }
          }))
      } catch (error) {
        console.error('fine-tune: registering the panel failed', error)
      }
    }

    exports.apply = apply
    exports.inject = ['slots']
    /** Exposed for the offline client test: pure helpers only. */
    exports._internals = {
      lossPoints, publishState, STATE_LABEL, MODELS, SEQ_LENGTHS, RANKS, MODE_INFO,
      PANEL_ID, API, defaultPlan, validatePlan, buildJudgePrompt, parseJudgeReply, tallyVerdicts,
    }
    return module.exports
  },
})
