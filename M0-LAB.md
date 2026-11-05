# M0 实验手册：6 GB 显存上能不能练出东西？

M0 回答的是整个方案里**唯一不能靠设计解决**的问题：**这台机器上，LoRA/QLoRA 训出来的模型在固定评测集上是否比底座更好，以及显存估算该信多少。**

- 如果「是」→ 用实测系数替换估算，进入完整参数面板与发布流程。
- 如果「否」→ 问题在数据量和底座选择，不在插件；先扩数据再谈。
- 如果「显存放不下」→ 用本手册的降级顺序；实在不行，远端 GPU 从可选变成必需。

**不要在没跑完这份手册之前就打开插件的「允许启动训练进程」。** 这一步约 1.5 小时（含装环境）。

---

## 0. 前置检查（5 分钟）

```powershell
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python --version
```

本机实测（2026-11-04）：`RTX 3060 Laptop GPU, 6144 MiB, 驱动 592.00`；系统 Python 3.11.9（**无 torch**）。

---

## 1. 装环境（20–40 分钟，取决于网速）

**独立 venv**，不要污染系统 Python（DSH 的 office 工具链也用系统 Python）。

```powershell
cd C:\Users\86156\Documents\deepseek-harness\default-workspace\fine-tune\python
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# CUDA 版 torch。--no-cache-dir 很关键：官方 index 的下载常卡在中途，
# 而 pip 的进度条只在终端里刷新（被重定向时看起来像卡死）。
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --progress-bar off torch --index-url https://download.pytorch.org/whl/cu121

.\.venv\Scripts\python.exe -m pip install transformers peft datasets accelerate bitsandbytes sentencepiece
```

> 官方 index 超过 10 分钟没进展时：先用清华镜像装 torch 的 **CPU 版**验证整条流程（`-i https://pypi.tuna.tsinghua.edu.cn/simple`），再补 cu121 wheel；或者直接走 §9 的远端方案。
> `HF_ENDPOINT=https://hf-mirror.com` 只在**下模型**时设，别在装包时设。

验收：

```powershell
.\.venv\Scripts\python.exe -c "import torch,transformers,peft,bitsandbytes as b; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(transformers.__version__, peft.__version__, b.__version__)"
```

必须出现 `True` 和 `NVIDIA GeForce RTX 3060 Laptop GPU`。

---

## 2. 先做显存校准（10 分钟，新增的第一步）

**为什么先校准再训练**：插件的显存公式里有两个系数——激活系数 `a`（字节/每 token×每 hidden×每层）与固定预留 `r`——都是经验值。校准用**真实模型跑一个真实优化步**，把这两个数测出来：

```powershell
@{
  model   = "Qwen/Qwen2.5-0.5B-Instruct"   # 形状探针：小到必定能跑
  mode    = "qlora"
  seqLens = @(512, 1024)
  batch   = 1
  out     = "$env:USERPROFILE\.dsh\fine-tune\m0"
} | ConvertTo-Json | Out-File -Encoding utf8 $env:TEMP\cal.json

.\.venv\Scripts\python.exe -m fnt_worker calibrate --config $env:TEMP\cal.json
```

产出 `m0\calibration.json`，里面有 `a` 与 `r`。填回 `lib/hardware.mjs` 的 `MEASURED`：

```js
export const MEASURED = {
  'Qwen/Qwen2.5-0.5B-Instruct': { qlora: { a: 16.4, r: 1.2, seqLen: 1024, device: 'RTX 3060 Laptop', torch: '2.x', at: '2026-11-04' } },
};
```

（上面的数字是格式示例，以你机器测出来的为准。）

> 这一步的产物是「本机实测」而不是「别人的经验」。没做它，后面的估算误差可能超过 30%。

---

## 3. 准备一份数据（10 分钟）

**不要用玩具数据。** 用插件从真实会话里抽出来的数据集。

1. 面板「数据」页选 5–10 个会话 → 「建数据集」（模式先选「学工具调用」）。
2. 记下数据集 id。
3. 看数据卡：token 中位数、丢弃原因、脱敏命中次数。
4. **抽 10 条人看一遍**（「样本预览」，坏样本点「移除」）。
5. 顺手确认文件里没有泄露：`Get-Content ...\train.jsonl -TotalCount 1` 不应出现 `sk-…`、邮箱、`C:\Users\<你的名字>`。

样本少于 50 条时，M0 只能验证「跑得通」，不能验证「有没有效果」。

---

## 4. 跑 M0（30–60 分钟）

### 4.1 先试运行（不过 GPU，1 分钟）

```powershell
cd C:\Users\86156\Documents\deepseek-harness\default-workspace\fine-tune\python
$data = "$env:USERPROFILE\.dsh\fine-tune\datasets\<id>"
@{
  model     = "Qwen/Qwen2.5-1.5B-Instruct"
  mode      = "qlora"
  data      = $data
  out       = "$env:USERPROFILE\.dsh\fine-tune\m0"
  maxSeqLen = 2048
  loraR     = 16
  epochs    = 1
  dryRun    = $true
} | ConvertTo-Json | Out-File -Encoding utf8 $env:TEMP\m0.json

.\.venv\Scripts\python.exe -m fnt_worker train --config $env:TEMP\m0.json
```

看到 `trainableParams` 和 `试运行：…没有开始训练。` 就说明依赖、数据、LoRA 都通了。

### 4.2 正式跑（30–40 分钟）

把 `dryRun` 改成 `false` 再跑。另开一个窗口盯显存：

```powershell
while ($true) { nvidia-smi --query-gpu=memory.used --format=csv,noheader; Start-Sleep 5 }
```

**要记下来的数字：**

| 指标 | 哪里看 | 填到哪 |
|---|---|---|
| 显存峰值 | `peakGiB` 事件 | 与 §2 的估算对比；误差 >20% 就重新校准 |
| 每步耗时 | loss 事件间隔 | 本文件「实测记录」 |
| 首步 / 末步 loss | 面板曲线或事件 | 同上 |
| 是否 OOM | 是 → 走 §5 | — |

### 4.3 再跑一次 LoRA 模式（可选，10 分钟）

把 `mode` 改成 `"lora"`。预期：**同样的 1.5B 会吃约两倍底座显存**。它跑得下来，说明 6 GB 没想象中紧，参数面板可以把 LoRA 设为可选而不只是「不建议」。

---

## 5. 如果 OOM：按这个顺序降级

1. `maxSeqLen` 2048 → 1536 → 1024
2. 保持 `mode: qlora`（LoRA 更吃显存）
3. `loraR` 16 → 8
4. 换 1.5B 或更小
5. 关掉其他吃显存的程序（Chrome 硬件加速也会占几百 MB）
6. 都没有用 → 远端 GPU（§9）

---

## 6. 评测：M0 真正的结论

```powershell
@{
  model     = "Qwen/Qwen2.5-1.5B-Instruct"
  data      = $data
  out       = "$env:USERPROFILE\.dsh\fine-tune\m0"
  evalLimit = 100
} | ConvertTo-Json | Out-File -Encoding utf8 $env:TEMP\m0-eval.json

.\.venv\Scripts\python.exe -m fnt_worker eval --config $env:TEMP\m0-eval.json
```

| 结果 | 含义 | 下一步 |
|---|---|---|
| 微调后 ≥ 底座 | **方案成立** | 回填系数，打开训练开关，进入参数面板与发布 |
| 低 0–3 个百分点 | 无回退但也没学到 | 扩数据到 300+ 条；试 4B |
| 低 >3 个百分点 | 有回退 | 抽 10 条人看标签，降 lr 到 1e-4 |

**注意**：这是**贴合度**（工具名/参数是否对），不是主观质量分。

---

## 7. 实测记录

### 2026-11-04 · RTX 3060 Laptop 6 GB · torch 2.5.1+cu121 · transformers 4.46.3 · peft 0.13.2 · bnb 0.44.1

#### 7.1 主结果：工具调用数据集（可判定指标）

| | 底座模型 | 微调后 | 变化 |
|---|---|---|---|
| **工具调用准确率** | **0.0%**（0/60） | **23.3%**（14/60） | **+23.3 个百分点** |

判定方式：样本标签是精确的工具调用，检查输出调用的工具名是否与标准答案一致。
`adherence_scored=60, floor_only=0` —— 60 条全是被真正比较的样本，没有一条凑数。

| 项 | 值 |
|---|---|
| 配置 | QLoRA · Qwen2.5-0.5B · seq 1024 · 300 步 · 1614 条训练样本 |
| 耗时 / 显存峰值 | 37.3 分钟 / **2.86 GiB** |
| loss | 1.49 → **0.64**（平滑，无 NaN） |
| eval_loss | 1.35 → 1.39（第二 epoch 略升，轻微过拟合） |

读法：底座 0% 说明它根本不会输出那个 JSON 结构；23.3% 说明**格式学会了，
工具选择只对了不到四分之一**。eval_loss 回升说明 300 步已经吃满这批数据——
**下一步该加样本，不是加步数**。

#### 7.2 对照：文字回答数据集（指标不可判定）

| 项 | 值 |
|---|---|
| 训练 | 32 步 / 137 条样本 / 2.1 分钟 |
| loss | 3.17 → 2.87 |
| 贴合度 | 100% → 100%（**两边同分，指标无区分力**） |

这组数据样本太少（137 条）、回答是开放式长文本，词重叠 0.04 vs 0.05，
任何绝对阈值都在量「回答长度」。留着是为了说明**指标选错会让人误以为成功了**。

#### 7.3 基础数据

| 项 | QLoRA · Qwen2.5-0.5B · seq 1024 |
|---|---|
| 校准 | 设备 6.0 GiB，真实形状 24 层 × 896 hidden |
| 训练显存峰值 | **2.86 GiB**（估算 6.16 GiB，偏保守 2.2×） |
| 评估显存峰值 | 修复后与训练同额；**修复前 8.98 GiB → OOM** |
| 可训练参数 | 8.80M / 323.9M（2.72%） |
| 数据可用率 | seq 512 只有 41/137 条可用；答案中位数 612 token → 需要 seq ≥ 1024 |
| 贴合度：底座 → 微调 | 100% → 100% **（同分，指标不可区分，见下）** |

### 这些数字推翻了什么

1. **6 GB 装不下 1.5B 的可用训练。** 1.5B @ seq 256 需 4.28 GiB（能跑，但样本被截断），
   @ seq 512 需 6.59 GiB（放不下）。而本机样本**答案中位数 612 token**，seq 256 会丢掉绝大多数。
   → 6 GB 上真正可行的是 0.5B 级模型。
2. **显存估算原先是错的**：漏了 logits（vocab × seq × 4 字节）与逐层激活保留，低估约 20 倍。
   修正后对实测偏差 +10~17%（偏保守，宁可误拒不可误放）。
3. **开放式文字回答没有可判定的贴合度指标。** 标准答案均值 2148 字符，小模型回答均值 360–484，
   词重叠 0.04 vs 0.05。任何绝对阈值都在量「回答长度」。→ 这类数据集必须用**偏好对比**
   （裁判或人工），不能用贴合度下结论。
4. **用工具注册表判「幻觉」会丢掉 92% 的真实数据。** 注册表记了 5 个工具，会话实际调了 14 个
   （`pwsh`/`edit`/`read`…），于是 2439 条真实调用被当成幻觉丢掉，只留下 38 条。
   改成「工具层明确回不存在的才算幻觉」后：**38 → 2737 条（72 倍）**，幻觉误判 2439 → 11。

### 跑 M0 期间修掉的真缺陷

| # | 缺陷 | 症状 |
|---|---|---|
| 1 | 估算漏 logits 项 | 低估显存 20× |
| 2 | 激活系数 4 B（实测 210 B） | 同上 |
| 3 | `DataCollatorForSeq2Seq` 把 ids 转 float | 训练第一步就崩 |
| 4 | `Frame` 类体引用局部变量 | 第一步崩 |
| 5 | **答案被整段截掉** | grad_norm=0、eval_loss=NaN |
| 6 | 验证集没有 batch_size | 训练成功后评估 OOM |
| 7 | 纯文字样本一律判「贴合」 | 两边都是 100%，无意义 |
| 8 | **用注册表判幻觉工具** | 丢掉 92% 真实调用（38 条 vs 应有的 2737） |

---

## 8. 常见问题

**Q：`No module named 'fnt_worker'`** — 必须在 `fine-tune\python` 目录下运行。

**Q：pip 看起来卡住了** — 先看 CPU 占用：持续 30% 以上多半是在解包；接近 0% 才是真卡。加 `--no-cache-dir --progress-bar off` 重试。

**Q：训练进程起来但面板没曲线** — 看 `~\.dsh\fine-tune\runs\<id>\stderr.log`，torch 的报错都在那里。

---

## 9. 远端方案（本机实在跑不动时）

SSH 后端的接口已写、执行钩子未接通（见 DESIGN.md §10）。当前可行的做法：

```powershell
# 把 data 目录和 python/ 一起 scp 到远端，远端有 torch 之后：
ssh user@gpu-host "cd /path/to/fnt && python3 -m fnt_worker train --config m0.json"
```
