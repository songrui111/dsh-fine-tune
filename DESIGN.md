# dsh-fine-tune — 设计方案

> 一句话：**别人给「训练参数」，这个插件给「数据 → 训练 → 评测 → 发布」一条有质量门的流水线**，并且数据可以直接来自 DSH 自己的会话日志。

本文是完整设计文档：定位、架构、数据契约、状态机、安全模型、真实难点、里程碑与验收、已知未实现。所有「已验证」的结论都注明是怎么验证的。

---

## 1. 定位与差异化

市面上的微调工具（Axolotl / LLaMA-Factory / Unsloth 笔记本）都不关心**你的 agent 实际在做什么**。DSH 里有一个别处没有的数据源：会话历史本身就是成规模的 `(情景 → 应做动作)` 数据。本插件围绕它做三件事：

| 能力 | 现有工具 | 本插件 |
|---|---|---|
| 数据来源 | 手工准备数据集 | 从会话日志抽轨迹；或导入本地 JSONL |
| 数据质量 | 无 | 幻觉工具名剔除、失败调用剔除、脱敏、SMILES 去重、按会话分组切分、数据卡 |
| 评测 | 看 loss 曲线 | 可判定的贴合度（工具名/参数/格式）+ 固定集回放；**有回退就不能发布** |

因此它**不是**「训练面板」，而是「**训练台**」：训练只是中间一步，前后各有一道门。

### 数据源分两类（`data_source` 是主键）

| kind | 学什么 | 默认样本模式 | 评测检查 |
|---|---|---|---|
| `agent` | 在这套工具集里的**行为**：情景 → 工具调用 | `action` | 工具名 + 参数 + 不编造工具 |
| `dataset` | **风格与领域知识**：提示 → 回答 | `answer` | 格式与空答；不假装做工具名检查 |

两类在数据里区分（`source.json` → 数据集 manifest），不是面板上的一个勾选：一次训练永远能说清它练的是什么，评测也按数据类型选检查项。

---

## 2. 架构

```
浏览器半体 client.js（侧栏「模型微调」）
  │ fetch POST /fine-tune/api/action   （唯一的写入口）
  │ EventSource /fine-tune/api/stream  （SSE：回放 + 增量）
  ▼
Host 半体 index.js（Node，插件进程内）
  ├─ lib/jobs.mjs     任务状态机 + runs/<id>/run.json + events.jsonl（重启可对账）
  ├─ lib/backend.mjs  本地 spawn（argv，不过 shell）/ SSH（shell-quoted JSON 配置）
  ├─ lib/session.mjs  会话日志解码（zstd 帧 + 首帧保留位怪癖）
  ├─ lib/mine.mjs     轨迹 → 样本（判定规则；见 §4）
  ├─ lib/redact.mjs   五档脱敏（密钥/PII/路径/全量）
  ├─ lib/pack.mjs     校验、SMILES 去重、token 估算、按组切分、写盘
  ├─ lib/dataset.mjs  数据源 → 数据集版本 + 数据卡 + 工具注册表快照
  ├─ lib/evaluate.mjs 贴合度打分 + 发布门（verdictOf）
  ├─ lib/hardware.mjs 显卡探测 + 显存预估 + 一条可点击的建议
  └─ lib/tools.mjs    5 个模型工具
  ▼
训练后端 python/fnt_worker（独立进程）
  transformers + bitsandbytes(NF4) + peft(QLoRA) → train / merge / export(GGUF) / eval
  stdout：每行一个 JSON 事件；stderr：给人看的日志（tee 到 stderr.log）
```

### 两个刻意的取舍

1. **训练是独立进程，不是宿主内的函数调用。** DSH 是 Electron 单进程：把 torch 塞进去（或和它一起崩），整个 GUI 会跟着掉。所以宿主只把它当「有 pid、有退出码、有 stderr 尾巴」的 job。
2. **本地后端按 argv 启动，不过 shell。** 用户可控且会进命令行的字符串只有 `modelId`，它还额外被 `/^[A-Za-z0-9._:/-]+$/` 校验——`--model && del` 这种情况在 Windows 上是真实存在的。

---

## 3. 数据契约（已验证）

### 3.1 会话日志

位置：`$DSH_HOME/sessions/<cwd-slug>/<session-id>/session.v4.jsonl.zstd`。

字节布局（在 4.8 MB / 2811 帧的真实日志上逐字节验证过）：

```
[0..4)    zstd magic
[4]       frame descriptor = 0x04
[5..9)    4 字节小端 = 该帧解压后大小（≈388 KB/帧）
之后的每一帧都是标准 zstd 单段帧，解压后换行分隔的 JSON 事件
```

关键点：**首帧的 descriptor 置了保留位（0x04）且只声明了大小**——zstd 规范认为这与「无校验、无字典、非单段」矛盾，于是整段被拒（`Unsupported frame parameter` / `Dictionary mismatch`）。**清掉这一个保留位后，所有帧（含首帧）都能解**：2811 成功 / 0 失败。解码器就在 `lib/session.mjs`，解不出来时返回 `decodable: false` 与原因，绝不报告「这个会话是空的」。

事件类型（本机日志实测的分布）：

| 事件 | 用途 |
|---|---|
| `session` | 头部：`id`、`cwd`、`agentPreset` |
| `developer/message` | 工具注册表：`tool-addition` / `tool-removal` 块（**注意：本机日志里同一注册表被写了两次，一次全增一次全删**） |
| `user/message` | `source.kind` 区分「人」与「spliced」；人话才作标签信号 |
| `assistant/message` | `content` 块：`reasoning` / `text` / `tool-call`；**tool-call 块的 id 字段叫 `id`，不是 `toolCallId`** |
| `tool/result` | `message.toolCallId`（**在 message 上，不在 data 上**）、`message.isError` |
| `turn/end` | `reason.kind`：`completed` / `aborted` |

这三处字段位置差异都在真实日志上核过（见 §9 的「已核实的坑」），也是单测覆盖的回归点。

### 3.2 样本

```json
{"messages":[{"role":"user","content":"[系统提示]…\n[用户]…\n[工具返回]…\n[用户]…"},
             {"role":"assistant","content":"```json\n{\"tool\":\"read\",\"arguments\":{\"path\":\"README.md\"}}\n```"}],
 "meta":{"sourceId":"…","sessionId":"…","group":"…","turn":1,"step":2,"origin":"agent_retried"}}
```

- **label 永不含 `reasoning` 块**：它是模型自己的私密笔记，学它等于让模型模仿自己的草稿，而且 token 成本最大。
- `origin` 是训练价值排序：`agent_executed` / `agent_retried` / `user_accepted`。

### 3.3 数据集版本

```
<dataDir>/datasets/<datasetId>/{manifest.json, train.jsonl, val.jsonl, eval.jsonl, fixture.json}
```

`fixture.json` 存的是**生成当时**的工具注册表快照：三个月后重评，仍然按模型当时被教过的工具集打分。

---

## 4. 挖掘规则（这是整个插件最该被挑错的部分）

对每个 assistant 步骤，先判定「谁认领了它」，只有被认领的输出才成为标签：

| 观察 | 判定 |
|---|---|
| 工具结果是 `isError: true` | **不是标签**，记为「重试的前奏」，同工具的下一次成功 → `agent_retried` |
| 工具结果成功 | 是标签（`agent_executed`） |
| 无工具调用，下一条是人类消息且是纠正（「不对 / 错了 / 应该 / no / wrong…」） | **不是标签** |
| 无工具调用，下一条是 `turn/end` 或人类非纠正回复 | 是标签（`user_accepted`） |
| 文本里自称「工具不存在 / unknown tool」却没有真调用 | 不是标签 |
| 调用的工具名不在**该会话声明过的注册表**里 | **丢弃并计数**（`hallucinated_tool`） |
| 标签超过 8000 字符 | 丢弃并计数（`oversized_label`）——那是在教模型抄整篇文件 |

两条重要细节：

1. **`nextMeaningful` 会跳过后续的 assistant/tool 步骤。** 一个含多个工具调用的步骤后面跟着自己更多的 assistant 步骤；如果把这些当成「后来发生的事」，每个中间步骤都会被误判成已被接受。
2. **注册表为空 ≠ 所有工具都是幻觉。** 只有当会话真的声明过注册表（`registryKnown`）才做工具名检查，否则如实跳过——本机 15 个会话里有 9 个没有任何注册表记录（`tool-removal` 之后又对应移除）。

### 4.1 在本机真实会话上的结果（2026-11-04）

15 个会话、解出 1 个不可读之外的会话：

| 模式 | 原始 | 保留 | 重要丢弃 | train/val/eval | prompt token 中位数 |
|---|---|---|---|---|---|
| action | 121 | 121 | 幻觉工具名 2126（9 个会话有注册表，且这些会话里大量 `rw_*` 工具已被移除）、失败调用 124、超长 3 | 80/20/21 | 1865 |
| answer | 279 | 279 | 失败调用 124 | 143/47/89 | 2633 |

**诚实解读**：`hallucinated_tool` 这么高，主要是因为那两个会话的注册表被重写过，而被移除的 `rw_*` 工具之后仍在被调用——这不是模型幻觉，而是工具的可用期问题。数据集很小（~100 条量级），**训出来的差异大概率落在噪声里**，这是先跑 M0 的原因，而不是先做面板的原因。

---

## 5. 任务状态机与重启对账

```
queued ──► running ──► succeeded
              ├──► failed
              ├──► cancelled
              └──► interrupted   （宿主重启 / 进程消失）
```

- 真相在磁盘：`runs/<id>/run.json` + 只追加的 `events.jsonl`（内存只是 UI 缓存）。
- **挂载时对账**：凡是还写着 `queued/running` 的任务，进程已经不属于当前宿主的，一律标 `interrupted` 并写 `host-restarted`——绝不把「可能还在跑」当「在跑」报给用户。
- 事件带单调 `seq`；`GET /api/run?id=&since=` 与 SSE 都从该游标回放，断线重连无需客户端记账。

---

## 6. 安全模型

| 门 | 实现 | 为什么 |
|---|---|---|
| 路由围栏 | 对端必须是 loopback；`Host` 必须是 loopback 主机名；变更请求若带 `Origin` 必须同源 | 插件路由在宿主自己的鉴权之外；面板能启动 GPU 进程、能触达 Ollama |
| 启动开关 | `allowSpawn`，默认关；关着时 HTTP 403、工具抛错并点名开关 | 「静默无操作」比「没有这个工具」更糟——模型会描述没发生的事 |
| 发布开关 | `allowPublish` **且** 存在绿色评测 **且** 有 `.gguf` | 发布是不可撤销的：权重会记住训练数据 |
| 路径纪律 | `modelId` 白名单字符集；`datasetId`/`source id` 只允许 `a-z0-9._-`；后端 argv 启动不过 shell | 唯一进命令行的用户字符串 |
| 请求体 | 64 KB 上限，超限直接断 socket | 不缓冲大请求 |
| 脱敏 | 默认 `path` 档；只在数据卡里报「规则 × 次数」，从不回显命中内容 | 数据集会离开这台机器 |

**UI 与门的一致性**：面板任何按钮禁用时都会说明是哪个开关/数值在挡着（`publishState()` 返回 `{ok, reason}`，单测覆盖）。

---

## 7. 显存与模型选择（6 GB 的现实）

估算公式（`lib/hardware.mjs`），逐项列出而不是只给一个总数：

```
baseWeights  = params × 0.65e9 × 0.55   (NF4 量化主体)
              + params × 0.35e9 × 2     (未量化的 fp16 embedding)
loraState    = layers × 4 × hidden × 2 × r × 8
activations  = seqLen × layers × hidden × 4 × batch
reserve      = 0.9 GiB                   (CUDA 上下文、cuBLAS 工作区、碎片)
```

在 RTX 3060 Laptop 6 GB 上的判断：

| 底座 | seq | 合计 | 结论 |
|---|---|---|---|
| Qwen2.5-1.5B | 2048 | ≈2.6 GiB | 稳 |
| Qwen3-4B | 2048 | ≈4.3 GiB | 可以试 |
| Qwen2.5-7B | 2048 | ≈6.1 GiB | **拒绝启动**（提示换模型） |

**这套系数是估算，不是实测**——M0 的任务之一就是把它换成实测值（`estimateMemory` 已支持传入 `measuredPeakGiB`，实测后覆盖）。`train_run` 在放不下时**不启动**并给出最小改动建议，而不是等到第 10 分钟 OOM。

---

## 7. 两种训练方式：LoRA 与 QLoRA

两者都是「冻结底座 + 低秩适配器」，差别只在**底座如何进入显存**：

| | QLoRA（`mode: "qlora"`） | LoRA（`mode: "lora"`） |
|---|---|---|
| 底座 | bitsandbytes 4bit（NF4 / FP4，可选双量化）+ `prepare_model_for_kbit_training` | fp16，`use_cache=false` |
| 适配器 | 同（r 4–64，α=2r，dropout） | 同 |
| 3B @ seq2048 估算 | **3.7 GiB** | **7.4 GiB** |
| 6 GB 上 | 1.5B–4B 可行 | 只够 1.5B 及以下 |

实现要点（`python/fnt_worker/__main__.py`）：

- `load_base()` 是两条独立分支，不是给一条路径加开关：QLoRA 要求 bitsandbytes，**没装就报 deps 错误而不是悄悄降级成 LoRA**——用户是按他选的模式估算显存的。
- `merge` 在 **CPU** 上把适配器并进 fp16 底座（4bit 适配器要并到去量化的底座），避免和别的进程抢显存。
- `eval` 两侧都在 **fp16** 下跑、适配器叠在 fp16 底座上——这是 peft 定义这种比较的方式；对量化底座评测会把量化误差算进「训练效果」。

显存公式（`lib/hardware.mjs`，按模式取系数）：

```
baseWeights  = params × 1e9 × baseBytesPerParam          (qlora 0.55 / lora 2)
              + params × 1e9 × embeddingShare × 2         (qlora 0.1：bnb 不量化 embedding)
loraState    = layers × 4 × hidden × 2 × r × 8
activations  = seqLen × layers × hidden × (4 | 5) × batch
reserve      = 0.9 GiB
```

`recommendation()` 的顺序因此变成：激活占大头 → 降序列；底座占大头且当前是 LoRA → **换 QLoRA**；否则 → 换小模型。

超参统一由 `lib/training.mjs` 的 `validateTrainingPlan()` 校验（面板、工具、路由三处共用），`loraR` 只接受 4/8/16/32/64，`maxSeqLen` 128–8192，`learningRate` 1e-6–1e-2。

---

## 7.1 面板交互（四个块）

| 块 | 行为 | 数据流 |
|---|---|---|
| 训练参数 | 方式/模型/seq/轮数/lr/r/batch/量化，改动即重算显存 | `POST /action {op:"gpu", …}` |
| 样本预览 | 分页看提示与标签；「移除」→ 写入 `exclusions.json`；可恢复 | `{op:"dataset.preview"}` / `{op:"dataset.exclude"}` |
| 对比评测 | 取题 → 填两列回答 → 复制裁判提示词 → 贴回 JSON 判定 → 解析胜平负 → 记录 | `{op:"evaluation.pair"}` → 本地解析 → `{op:"eval.record"}` |
| 结论与发布 | 贴合度对比、失败样例、发布门 | `{op:"publish.prepare"}`（需绿灯 + `.gguf`） |

**剔除必须真的生效**：训练启动时把「数据集 − 已剔除」物化到 `runs/<id>/data/`，worker 读的是这份；预览、评测题、训练三处都走同一个 `loadSplit()`，不会出现「界面上删了、训练里还在」。

**评测不伪造分数**：面板做的是「裁判流程」——把两份回答随机标成 A/B（按样本哈希稳定翻转）、连同当时可用的工具列表一起交给裁判，再把 JSON 判定解析成胜平负。记录时明确标注来源是人工/裁判判定，不冒充贴合度。

---

## 7.2 M0 实测（2026-11-04，RTX 3060 Laptop 6 GB）

跑通后的结论，与设计阶段的假设有出入，以实测为准：

| 项 | 实测 |
|---|---|
| 可训练的最优配置 | QLoRA · Qwen2.5-0.5B · seq 1024（32 步 / 2.1 分钟） |
| 显存峰值 | 训练与评估均 **2.86 GiB** |
| 1.5B | seq 256 需 4.28 GiB（能跑但丢样本）；seq 512 需 6.59 GiB（放不下） |
| 4B / 7B | 均远超 6 GB |

**估算公式的两次修正**（原公式低估约 20 倍）：

1. 漏了 **logits 项**：`transformers` 把语言模型头输出升到 fp32 算损失，
   即 vocab × seq × 4 字节。152k 词表 × seq 1024 = 0.58 GiB，且要留到反向传播。
2. **激活系数**由 4 B/token/层/hidden 修正为实测 **210 B**——
   反向传播保留逐层计算节点，梯度检查点是重算而不是释放。

**评测指标天花板**：对开放式文字回答，贴合度只能是「没有跑题」的下限
（标准答案均值 2148 字符 vs 小模型回答 360–484，词重叠 0.04–0.05）。
因此 `eval` 同时输出 `adherence_scored`（工具调用样本数）与 `floor_only`（纯文字样本数），
只有前者构成真正的比较。质量判断必须走偏好对比。

---

## 8. 里程碑与验收

| 里程碑 | 内容 | 验收（可执行） | 状态 |
|---|---|---|---|
| M0 | 6 GB 上 1.5B/4B QLoRA 可行性 | `M0-LAB.md` 跑通，产出实测显存与 loss 曲线 | 待做（需装依赖） |
| M1 | 面板 + 任务状态机 + SSE + worker | `npm test` 三套全过；`python -m unittest` 全过 | **已完成** |
| M1.5 | LoRA/QLoRA 双模式 + 四处交互（参数/预览剔除/对比评测/发布门） | 33 单测 + 路由级测试 + 10 个 worker 测试全过 | **已完成** |
| M2 | 抽取/脱敏/去重/数据卡 | 真实日志跑通并产出数据卡（§4.1） | **已完成** |
| M3 | 评测门 + 发布准备 | `verdictOf` 单测；`publish.prepare` 在无绿灯时 409 | **已完成** |
| M4 | 远端 SSH 训练 + 产物回传 | 接上宿主 SSH 服务后跑一次远端训练 | 接口已留，见 §10 |

---

## 9. 已核实的坑（写下来，免得下一版重踩）

1. **会话日志首帧解不开**：descriptor 保留位 `0x04`；清掉即全部可解（2811/2811）。
2. **`tool-call` 块用 `id`**，不是 `toolCallId`；`tool/result` 的 `toolCallId` 在 `data.message` 上，不在 `data` 上。抓错位置 → 失败结果归属不到调用（第一版就这样，导致"重试"永远识别不出来）。
3. **注册表会被重写**：`developer/message` 里先全量 addition、后全量 removal；只认 `toolName` 字段、按出现顺序折叠。
4. **`nextMeaningful` 不能看后续 assistant 步骤**，否则多工具步骤全部误判为已接受。
5. **PowerShell 的 `Out-File` 会写 BOM**：worker 的配置与数据读取都用 `utf-8-sig`。
6. **`fetch` 会丢弃自定义 `Host` 头**：测 DNS rebinding 围栏必须用原始 `http.request`。
7. **大 `write_file` 标签**：真实会话里出现过 40 KB 的 README 内容作为参数；不设上限就是在教模型抄写。
8. **Windows 控制台编码**：worker 的协议行可能被控制台解码弄乱，所以宿主给子进程强制 `PYTHONIOENCODING=utf-8`，且解析器只认真 JSON。

---

## 10. 已知未实现（诚实清单）

| 项 | 现状 | 接法 |
|---|---|---|
| SSH 后端执行 | `createSshBackend` 已写好（命令构造、shell 引用、SSE 解析、取消）；宿主侧 `sshExec` 目前**直接抛「尚未接通」** | 接宿主 `ssh`/`subprocess` 服务；数据集上传/产物回传也要一起接 |
| 数据集自动上传远端 | 未做 | 用 tty 插件的 SFTP 宿主侧接口 |
| 面板触发 A/B 生成 | **不做**：真 A/B 要同时加载两个模型 | 面板做「裁判流程」：取题、填回答、贴裁判判定（见 §7.1） |
| Ollama 自动创建 | 只给出 `ollama create` 命令，不替你在后台写 | 由终端执行 |
| 显存估算系数 | 经验估值 | M0 实测后传入 `measuredPeakGiB` |
| 工具注册表缺省的会话 | 不做工具名检查，如实标注 `registryKnown=false` | — |
| 会话日志格式版本 | 只认 `session.v*.jsonl.zstd`；`v5` 会列出来但解码器未适配 | 解码失败时报 `decodable:false`，不静默 |

---

## 11. 文件结构与职责

| 文件 | 职责 | 规模 |
|---|---|---|
| `index.js` | 挂载、4 条路由、围栏、action 分发、任务钩子 | ~540 行 |
| `lib/session.mjs` | zstd 帧解码 + 会话枚举 | ~130 |
| `lib/mine.mjs` | 判定规则 + 分组切分 | ~430 |
| `lib/redact.mjs` | 五档脱敏（含 Luhn / 身份证校验位） | ~200 |
| `lib/pack.mjs` | 校验、去重、token 估算、写盘 | ~120 |
| `lib/dataset.mjs` | 数据源 → 数据集 + 数据卡 | ~230 |
| `lib/sources.mjs` | 数据源校验与持久化 | ~110 |
| `lib/evaluate.mjs` | 贴合度 + 发布门 + Markdown 报告 | ~230 |
| `lib/hardware.mjs` | 模型表、显存估算、nvidia-smi 解析 | ~140 |
| `lib/jobs.mjs` | 任务状态机 + 重启对账 | ~230 |
| `lib/backend.mjs` | 本地/SSH 适配器 | ~250 |
| `lib/tools.mjs` | 5 个模型工具 | ~340 |
| `client/client.js` | 面板（手写、无构建） | ~700 |
| `python/fnt_worker/__main__.py` | 训练/合并/导出/评测 | ~460 |

零运行时依赖、零构建步骤。测试：`test/unit.mjs`（31）、`test/host.mjs`（路由级）、`test/client.mjs`（bundle 装载）、`python/tests`（8）。
