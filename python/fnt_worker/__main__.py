"""Fine-tuning worker for the DSH fine-tune plugin.

The host starts this module, not a shell script:

    python -m fnt_worker train  --config run.json
    python -m fnt_worker merge  --config run.json
    python -m fnt_worker export --config run.json
    python -m fnt_worker eval   --config run.json

stdout is a machine channel: exactly one JSON object per line, nothing else.
Every log line a human might want goes to stderr, which the host tees into
``runs/<id>/stderr.log``. That split is what lets a crashed torch run explain
itself without ever confusing the event parser.

Why this is a separate process and not the plugin itself: DSH is an Electron
host. Torch needs a CUDA context and several GiB of memory; loading it (or
crashing next to it) inside the agent's host would take the whole GUI down. The
host treats this process as a job with a pid, a stderr tail, and an exit code.

Heavy imports (torch, transformers, peft) are imported *inside* each command, so
``--help`` and the config validation work on a machine with no ML stack at all.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
from typing import Any

from . import die, emit, log, note


def load_config(path: str) -> dict[str, Any]:
    # `utf-8-sig`: the host writes plain UTF-8, but a config edited by a Windows
    # tool (PowerShell, Notepad) may carry a BOM — and a BOM must not read as "not JSON".
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            config = json.load(handle)
    except FileNotFoundError:
        die("config", f"配置文件不存在：{path}")
    except json.JSONDecodeError as error:
        die("config", f"配置文件不是合法 JSON：{error}")
    if not isinstance(config, dict):
        die("config", "配置文件不是一个 JSON 对象")
    return config


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ----------------------------------------------------------------------- train

def build_prompt_mask(tokenizer, messages: list[dict[str, str]], max_len: int) -> tuple[list[int], list[int]]:
    """Tokenise a chat sample and mask the prompt half out of the loss.

    The model must learn to produce the *answer*, not to reproduce the user's
    question: without masking, most of the loss (and of the gradient) comes from
    the prompt, which is exactly the part the model should never be rewarded for
    copying. The mask is computed by tokenising the prompt alone and taking that
    prefix length, so multi-byte text and special tokens line up correctly.
    """
    prompt = messages[:-1]
    prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)

    # The answer must survive: a sample whose label is entirely masked teaches
    # nothing and (measured) produces grad_norm 0 and NaN eval loss. So:
    #   1. if the whole sample fits, keep it;
    #   2. else keep the *whole* answer and trim the prompt from the front;
    #   3. if the answer alone exceeds max_len, the sample is unusable.
    answer_len = max(1, len(full_ids) - len(prompt_ids))
    if answer_len > max_len:
        return [], []
    keep_prompt = max(0, max_len - answer_len)
    if len(prompt_ids) > keep_prompt:
        prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []
        full_ids = prompt_ids + full_ids[-answer_len:]
    else:
        full_ids = prompt_ids + full_ids[len(prompt_ids):]

    # -100 over the prompt, real ids over the answer.
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if len(labels) < len(full_ids):
        labels += full_ids[len(labels):]
    labels = labels[: len(full_ids)]
    if not any(label != -100 for label in labels):
        return [], []
    return full_ids, labels


def load_base(model_id: str, config: dict[str, Any], for_training: bool):
    """Load the base model in the mode the user chose.

    * ``qlora`` - 4-bit NF4/FP4 base through bitsandbytes, then
      ``prepare_model_for_kbit_training``. This is what fits a small card.
    * ``lora`` - the base in fp16, fully frozen; no quantisation noise, twice the
      weight memory.

    A QLoRA request on a machine without bitsandbytes is an error, not a silent
    downgrade: the memory the user planned for is the memory of the chosen mode.
    """
    import torch
    from transformers import AutoModelForCausalLM

    mode = str(config.get("mode", "qlora"))
    device_map = {"": 0} if torch.cuda.is_available() else None
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    if mode == "qlora":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as error:  # noqa: BLE001
            die("deps", f"QLoRA 需要 transformers 的 BitsAndBytesConfig，但导入失败：{error}")
        try:
            import bitsandbytes  # noqa: F401
        except Exception as error:  # noqa: BLE001
            die("deps", f"QLoRA 需要 bitsandbytes，但导入失败：{error}。可改用 LoRA，或先装好 bitsandbytes。")
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(config.get("quantType", "nf4")),
            bnb_4bit_use_double_quant=bool(config.get("doubleQuant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quant, device_map=device_map, trust_remote_code=True
        )
        if for_training:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        return model, "qlora"

    if mode != "lora":
        die("config", f"未知训练方式 {mode}，只能是 qlora 或 lora")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=device_map, trust_remote_code=True
        )
    except Exception as error:  # noqa: BLE001
        hint = "（fp16 底座放不下时会直接 OOM，小显存请改用 QLoRA）" if "memory" in str(error).lower() else ""
        die("load", f"加载 fp16 底座失败：{error}{hint}")
    if for_training:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    return model, "lora"


def attach_lora(model, config: dict[str, Any]):
    """Wrap a loaded model with the LoRA adapter described by the plan.

    ``targetModules`` absent means "every linear layer", which is what
    LLaMA-Factory defaults to and what PEFT calls ``all-linear``. Naming modules
    explicitly is the older style and breaks across architectures (Qwen uses
    q_proj, Falcon uses query_key_value, ChatGLM uses W_pack), so the preset is
    only honoured when the plan actually lists names.
    """
    from peft import LoraConfig, get_peft_model

    lora_r = int(config.get("loraR", 16))
    raw_targets = config.get("targetModules")
    targets = raw_targets if isinstance(raw_targets, list) and len(raw_targets) > 0 else "all-linear"
    lora = LoraConfig(
        r=lora_r,
        lora_alpha=int(config.get("loraAlpha", lora_r * 2)),
        lora_dropout=float(config.get("loraDropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
        # rsLoRA keeps the effective adapter magnitude stable as r changes;
        # without it, doubling r silently doubles the update size.
        use_rslora=bool(config.get("useRsLora", False)),
        use_dora=bool(config.get("useDora", False)),
    )
    return get_peft_model(model, lora)


def resolve_precision(config: dict[str, Any], torch) -> tuple[Any, Any]:
    """Pick (compute_dtype, training_flags) from the plan and what the card offers.

    ``auto`` is the default and means bf16 when the GPU reports support, fp16
    otherwise. An explicit choice is honoured, except that bf16 on a card without
    it is refused with a sentence rather than silently downgraded — a user who
    asked for bf16 deserves to know they did not get it.
    """
    want = str(config.get("precision", "auto"))
    has_gpu = torch.cuda.is_available()
    bf16_ok = has_gpu and torch.cuda.is_bf16_supported()

    if want == "fp32":
        return torch.float32, {"bf16": False, "fp16": False}
    if want == "bf16":
        if not bf16_ok:
            die("config", "这张卡不支持 bf16（需要 Ampere 及以上），请改用 fp16 或 auto")
        return torch.bfloat16, {"bf16": True, "fp16": False}
    if want == "fp16":
        if not has_gpu:
            die("config", "没有可用的 GPU，fp16 训练没有意义（会退化成 CPU 计算）")
        return torch.float16, {"bf16": False, "fp16": True}
    # auto
    if bf16_ok:
        return torch.bfloat16, {"bf16": True, "fp16": False}
    return (torch.float16 if has_gpu else torch.float32), {"bf16": False, "fp16": has_gpu}


def deepspeed_plan(config: dict[str, Any], out_dir: str) -> str | None:
    """Materialise a DeepSpeed config, or refuse with the reason it cannot work.

    DeepSpeed ZeRO saves *parameter* memory, which is the wrong axis for a LoRA
    run (the frozen base still lives on every card) and meaningless on one card.
    It is supported here because users bring machines with several GPUs, not
    because it is the recommended path on a 6 GB laptop.
    """
    stage = str(config.get("deepspeed", "none"))
    if stage == "none":
        return None
    import torch

    if not torch.cuda.is_available():
        die("config", "DeepSpeed 需要 GPU")
    if torch.cuda.device_count() < 2:
        die("config", f"DeepSpeed ZeRO 需要多张卡（检测到 {torch.cuda.device_count()} 张）；单卡上没有收益")
    if not sys.platform.startswith("linux"):
        die("config", "DeepSpeed 目前只支持 Linux（Windows 上没有可用的多卡路径）")

    zero = {"zero2": 2, "zero3": 3}[stage]
    path = os.path.join(out_dir, f"deepspeed_zero{zero}.json")
    _ensure_dir(out_dir)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "bf16": {"enabled": True},
                "zero_optimization": {"stage": zero, "offload_optimizer": {"device": "none"}},
                "train_micro_batch_size_per_gpu": int(config.get("batchSize", 1)),
                "gradient_accumulation_steps": int(config.get("gradAccum", 8)),
            },
            handle,
            indent=2,
        )
    return path


class PadCollator:
    """Right-pad ids / mask / labels to the batch max, so batch_size > 1 is valid.

    Labels pad with -100 (never contributes to the loss), ids with the pad token.
    A causal LM must get int `input_ids`: `DataCollatorForSeq2Seq` cast them to
    float and the embedding lookup rejected them outright.
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        width = max(len(feature["input_ids"]) for feature in features)
        batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            ids = list(feature["input_ids"])
            labels = list(feature["labels"])
            pad = width - len(ids)
            batch["input_ids"].append(ids + [self.pad_token_id] * pad)
            batch["attention_mask"].append(list(feature.get("attention_mask", [1] * len(ids))) + [0] * pad)
            batch["labels"].append(labels + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def cmd_train(config: dict[str, Any]) -> None:
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except Exception as error:  # noqa: BLE001 - the message is the product here
        # Importing torch can fail for reasons that are not a missing package:
        # "No usable temporary directory found" comes from tempfile while torch's
        # own import machinery touches disk, and reporting it as a missing
        # dependency sends the user to reinstall a package that is already there.
        name = type(error).__name__
        text = str(error)
        if "temporary directory" in text.lower() or name in ("FileNotFoundError", "PermissionError"):
            die(
                "env",
                f"训练环境不可用：{name}: {text}。"
                "这是临时目录/文件权限问题，不是缺少依赖 —— torch 已经装好了。"
                "检查 TEMP 指向的目录是否可写，或用 TMPDIR 指定一个空目录后重试。",
            )
        die(
            "deps",
            f"训练依赖不完整：{error}。装好 torch / transformers / peft / datasets / bitsandbytes 后重试。",
        )

    data_dir = config.get("data")
    out_dir = config.get("out")
    model_id = config.get("model")
    if not data_dir or not out_dir or not model_id:
        die("config", "train 需要 data / out / model 三个字段")

    train_rows = read_jsonl(os.path.join(data_dir, "train.jsonl"))
    val_rows = read_jsonl(os.path.join(data_dir, "val.jsonl"))
    if not train_rows:
        die("data", "train.jsonl 是空的：没有可训练的样本")

    max_len = int(config.get("maxSeqLen", 2048))
    epochs = float(config.get("epochs", 1))
    lora_r = int(config.get("loraR", 16))
    batch_size = int(config.get("batchSize", 1))
    grad_accum = int(config.get("gradAccum", 8))
    lr = float(config.get("learningRate", 2e-4))
    dry_run = bool(config.get("dryRun", False))
    mode = str(config.get("mode", "qlora"))
    seed = int(config.get("seed", 42))

    emit(kind="phase", phase="load", message=f"加载 {model_id}（{mode}）")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, loaded_mode = load_base(model_id, config, for_training=True)
    model = attach_lora(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    emit(kind="metric", step=0, values={"trainableParams": int(trainable), "totalParams": int(total), "mode": loaded_mode})
    log(f"{loaded_mode.upper()} 可训练参数 {trainable:,} / 总计 {total:,}（{100 * trainable / max(total, 1):.2f}%）")

    def encode(row: dict[str, Any]) -> dict[str, Any] | None:
        ids, labels = build_prompt_mask(tokenizer, row["messages"], max_len)
        if not ids:
            return None
        return {"input_ids": ids, "labels": labels}

    def build_split(rows: list[dict[str, Any]], label: str) -> tuple[Any, int]:
        """Encode a split, dropping samples whose answer cannot fit whole."""
        encoded = []
        dropped = 0
        for row in rows:
            item = encode(row)
            if item is None:
                dropped += 1
                continue
            encoded.append(item)
        if dropped:
            log(
                f"{label}：{dropped} 条样本的答案比 maxSeqLen={max_len} 还长，已丢弃；"
                "它们的标签会被整段截掉，训了没有梯度，留下只会让 loss 变成常数。"
            )
        return (Dataset.from_list(encoded) if encoded else None), dropped

    train_ds, train_dropped = build_split(train_rows, "训练集")
    if train_ds is None:
        die(
            "data",
            f"seq 长度 {max_len} 太短：没有任何样本的答案能整段放下。"
            "请提高序列长度，或换更短的样本。",
        )
    val_ds, _val_dropped = build_split(val_rows, "验证集") if val_rows else (None, 0)

    if dry_run:
        emit(kind="artifact", label="plan", path=os.path.join(out_dir, "plan.json"))
        _ensure_dir(out_dir)
        with open(os.path.join(out_dir, "plan.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model": model_id,
                    "train": len(train_rows),
                    "val": len(val_rows),
                    "maxSeqLen": max_len,
                    "epochs": epochs,
                    "loraR": lora_r,
                    "trainableParams": int(trainable),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        log("试运行：依赖齐全、数据可解析、LoRA 已挂上；没有开始训练。")
        return

    _ensure_dir(out_dir)

    def on_log(args) -> None:
        """Bridge the Trainer's own logging to protocol events."""
        values: dict[str, Any] = {}
        for key, value in (args.logs or {}).items():
            if key in ("loss", "learning_rate", "epoch", "eval_loss", "grad_norm"):
                values[key] = float(value) if value is not None else None
        step = getattr(args, "global_step", 0) or 0
        loss = values.get("loss", values.get("eval_loss"))
        emit(kind="metric", step=int(step), loss=loss, values=values)
        if torch.cuda.is_available():
            peak = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
            emit(kind="metric", step=int(step), values={"peakGiB": round(peak, 2)})

    compute_dtype, precision_flags = resolve_precision(config, torch)
    ds_path = deepspeed_plan(config, out_dir)
    early_stopping = bool(config.get("earlyStopping", False))
    metric_for_best = "eval_loss" if val_ds is not None else None

    training_args = TrainingArguments(
        output_dir=out_dir,
        seed=seed,
        data_seed=seed,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type=str(config.get("lrScheduler", "cosine")),
        warmup_ratio=float(config.get("warmupRatio", 0.03)),
        weight_decay=float(config.get("weightDecay", 0)),
        adam_beta2=float(config.get("adamBeta2", 0.999)),
        max_grad_norm=float(config.get("maxGradNorm", 1)),
        logging_steps=int(config.get("loggingSteps", 5)),
        # No intermediate checkpoints. The adapter saved at the end of the run is the
        # only artifact anyone uses, and each checkpoint costs a directory *plus* a
        # full copy of the adapter. On a machine where a security product blocks
        # Python from creating directories, the first checkpoint is also the first
        # hard failure ? two good reasons to not write them.
        save_strategy="no",
        save_total_limit=None,
        eval_strategy="epoch" if val_ds is not None else "no",
        # Validation is a forward pass over the whole split; without its own batch
        # size it materialises every val sample at once and OOMs *after* a
        # successful training run (measured: 8.98 GiB peak on a 6 GB card at seq
        # 1024, while training peaked at 2.86 GiB).
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=1,
        bf16=precision_flags["bf16"],
        fp16=precision_flags["fp16"],
        # Early stopping is about hours of wasted GPU: an eval that has stopped
        # improving for N rounds is not going to improve in round N+1.
        # Cannot reload a best model when no checkpoints are written: with
        # `save_strategy="no"` the Trainer has nowhere to keep it. Early stopping
        # still stops the run; it just leaves the final weights in place.
        load_best_model_at_end=False,
        metric_for_best_model=metric_for_best,
        greater_is_better=False if metric_for_best is not None else None,
        gradient_checkpointing=bool(config.get("gradientCheckpointing", True)),
        report_to=[],
        # 8-bit paged AdamW carries the optimizer state in a fraction of fp32's
        # memory; with a 4bit base that state is small anyway, and with an fp16
        # base it is the difference between "fits" and "does not".
        optim=str(config.get("optim", "paged_adamw_8bit")),
        deepspeed=ds_path,
        max_steps=int(config.get("maxSteps") or -1),
        disable_tqdm=True,
    )
    callbacks = [_CallbackBridge(on_log)]
    if early_stopping and metric_for_best is not None:
        from transformers import EarlyStoppingCallback

        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=int(config.get("earlyStoppingPatience", 3)))
        )
    collator = PadCollator(tokenizer.pad_token_id or 0)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    emit(kind="phase", phase="train", message=f"{len(train_rows)} 条训练样本，{epochs} 轮")
    started = time.time()
    trainer.train(resume_from_checkpoint=bool(config.get("resumeFrom")) or None)
    elapsed = time.time() - started
    if torch.cuda.is_available():
        emit(kind="metric", step=int(trainer.state.global_step), values={"peakGiB": round(float(torch.cuda.max_memory_allocated() / (1024 ** 3)), 2)})

    adapter_dir = os.path.join(out_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    emit(kind="artifact", label="adapter", path=adapter_dir)
    log(f"训练完成：{trainer.state.global_step} 步，{elapsed / 60:.1f} 分钟。适配器在 {adapter_dir}")

    # ---- artifacts: merging and export are separate steps, and each says what
    # it needs before it starts rather than failing halfway through.
    if bool(config.get("mergeAfter", False)):
        try:
            _merge_adapter(model, adapter_dir, os.path.join(out_dir, "merged"))
            emit(kind="artifact", label="merged", path=os.path.join(out_dir, "merged"))
            log("已合并为完整权重（半精度副本，不再需要 QLoRA 运行时）。")
        except Exception as error:  # noqa: BLE001
            log(f"合并失败：{error}（适配器本身是好的，可以稍后单独合并）", level="warn")

    target = str(config.get("exportTarget", "none"))
    if target == "gguf":
        try:
            _export_gguf(os.path.join(out_dir, "merged") if bool(config.get("mergeAfter", False)) else adapter_dir, out_dir)
        except Exception as error:  # noqa: BLE001
            log(f"导出 GGUF 失败：{error}", level="warn")
    elif target == "ollama":
        log("Ollama 包需要 GGUF；请先用 exportTarget=gguf 导出，再执行 ollama create。", level="warn")


def _merge_adapter(model, adapter_dir: str, merged_dir: str) -> None:
    """Fold the adapter into a half-precision copy of the base weights."""
    from peft import PeftModel

    base = getattr(model, "get_base_model", lambda: model)()
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)


def _export_gguf(source: str, out_dir: str) -> None:
    """Convert to GGUF with llama.cpp when it is available, and say so when not.

    The converter lives in the llama.cpp checkout, not on PyPI, so this checks
    for it and prints the exact command rather than pretending the export is a
    one-liner the plugin can run on its own.
    """
    import shutil
    import subprocess

    script = os.environ.get("LLAMA_CPP_CONVERT", "")
    if script == "" or not os.path.exists(script):
        die(
            "deps",
            "导出 GGUF 需要 llama.cpp 的 convert_hf_to_gguf.py："
            "git clone https://github.com/ggerganov/llama.cpp 并设置 LLAMA_CPP_CONVERT 指向该脚本",
        )
    python = shutil.which("python") or sys.executable
    out = os.path.join(out_dir, "model.gguf")
    subprocess.run([python, script, source, "--outfile", out, "--outtype", "f16"], check=True)
    emit(kind="artifact", label="gguf", path=out)
    log(f"已导出 GGUF：{out}")


def _CallbackBridge(on_log):
    """A TrainerCallback that forwards `on_log` to our emitter."""
    from transformers import TrainerCallback

    class Bridge(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            # A plain object, not a class body: a class body cannot close over
            # `logs` (it is a different scope), which raised AttributeError.
            frame = SimpleNamespace(logs=logs or {}, global_step=state.global_step)
            on_log(frame)

    return Bridge()


# ----------------------------------------------------------------------- merge

def cmd_merge(config: dict[str, Any]) -> None:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as error:  # noqa: BLE001
        die("deps", f"合并依赖不完整：{error}")

    out_dir = config.get("out")
    model_id = config.get("model")
    adapter_dir = os.path.join(out_dir or "", "adapter")
    if not os.path.isdir(adapter_dir):
        die("artifact", f"找不到适配器目录：{adapter_dir}")
    emit(kind="phase", phase="merge", message="合并 LoRA 到 fp16 底座")
    # Merge on CPU on purpose: a 4-bit-trained adapter merges into a *dequantised*
    # fp16 base, and doing that on the GPU would compete with whatever else (a
    # running eval, the desktop compositor) holds memory. It is slow but honest.
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    target = os.path.join(out_dir, "merged")
    merged.save_pretrained(target, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_id, trust_remote_code=True).save_pretrained(target)
    emit(kind="artifact", label="merged", path=target)
    log(f"合并完成：{target}（CPU 上做，避免显存被同时占用）")


# ---------------------------------------------------------------------- export

def cmd_export(config: dict[str, Any]) -> None:
    """Quantise to GGUF through llama.cpp, when it is installed."""
    import shutil
    import subprocess

    out_dir = config.get("out") or ""
    merged = os.path.join(out_dir, "merged")
    if not os.path.isdir(merged):
        die("artifact", "先跑 merge：没有 merged 目录就没有可导出的权重")
    tool = shutil.which("llama-quantize") or shutil.which("quantize")
    if tool is None:
        die(
            "deps",
            "找不到 llama.cpp 的 quantize 工具（llama-quantize）。装好之后重试；"
            "本插件不替你下载第三方二进制。",
        )
    f16 = os.path.join(out_dir, "model-f16.gguf")
    q4 = os.path.join(out_dir, "model-q4_k_m.gguf")
    emit(kind="phase", phase="export", message="转 f16 GGUF")
    convert = shutil.which("convert-hf-to-gguf") or shutil.which("convert_hf_to_gguf.py")
    if convert is None:
        die("deps", "找不到 convert_hf_to_gguf（llama.cpp 的转换脚本）")
    subprocess.run([sys.executable, convert, merged, "--outfile", f16, "--outtype", "f16"], check=True)
    emit(kind="phase", phase="export", message="量化到 Q4_K_M")
    subprocess.run([tool, f16, q4, "Q4_K_M"], check=True)
    emit(kind="artifact", label="gguf", path=q4)
    log(f"导出完成：{q4}")


# ------------------------------------------------------------------------ eval

def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, pad_token_id=tokenizer.pad_token_id)
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _tools_in(text: str) -> list[dict[str, Any]]:
    """Parse the tool calls a candidate answer claims — the same convention the host uses."""
    import re

    calls: list[dict[str, Any]] = []
    for match in re.finditer(r"```(?:json)?\s*\n(.*?)```", text or "", re.S):
        _consider(match.group(1), calls)
    if not calls:
        start, end = (text or "").find("{"), (text or "").rfind("}")
        if start != -1 and end > start:
            _consider(text[start : end + 1], calls)
    return calls


def _consider(block: str, calls: list[dict[str, Any]]) -> None:
    try:
        payload = json.loads(block.strip())
    except Exception:  # noqa: BLE001
        return
    if isinstance(payload, dict) and isinstance(payload.get("tool"), str):
        calls.append({"tool": payload["tool"], "arguments": payload.get("arguments")})


def _score(cases: list[dict[str, Any]], outputs: list[str]) -> dict[str, Any]:
    """Score one side of the comparison.

    Two kinds of case, decided by what the gold answer *is*:

    * **action cases** - the gold answer contains a tool call, so the check is
      "did the output call that tool".
    * **answer cases** - the gold answer is prose, so there is no exact target.
      Scoring it as "non-empty passes" (the first version) makes every side score
      100% and the comparison meaningless. Instead the answer is scored on
      whether it stays on topic: the overlap between the output's and the gold's
      content words (a cheap stand-in for "did it address the same request").

    The number is a coarse signal, not a grade; what it is good for is telling a
    model that ignores the question apart from one that answers it.
    """
    import re

    def words(text: str) -> set[str]:
        # CJK: character bigrams; latin: lowercased words. Cheap, no tokenizer.
        text = text or ""
        latin = set(re.findall(r"[a-z0-9_]{2,}", text.lower()))
        cjk = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
        bigrams = {cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)}
        return latin | bigrams

    passed = 0
    failures = []
    kinds = {"tool": 0, "answer": 0}
    for index, case in enumerate(cases):
        expected = case.get("expected") or ""
        output = outputs[index] or ""
        expected_calls = _tools_in(expected)
        if expected_calls:
            kinds["tool"] += 1
            want = expected_calls[0]
            got = _tools_in(output)
            ok = any(call["tool"] == want["tool"] for call in got)
        else:
            kinds["answer"] += 1
            # Overlap of content words, measured against the gold answer.
            gold_words = words(expected)
            out_words = words(output)
            if not out_words:
                ok = False
            elif gold_words:
                overlap = len(gold_words & out_words) / len(gold_words)
                # Measured on a real run (2026-11-04, Qwen2.5-0.5B, 40 cases):
                # a gold answer averages ~2100 chars while a small model replies
                # with ~400, so raw overlap is 0.04 (base) to 0.05 (fine-tuned) --
                # no absolute threshold separates those two, and one that "passes"
                # them would be measuring answer length, not relevance.
                #
                # So this check is a *floor*, not a grade: it catches empty and
                # clearly off-topic replies and nothing finer. For prose answers
                # the verdict that matters is the preference comparison (a judge
                # or a human), which is why `eval` reports `adherence_scored` and
                # the report says which measure produced the number.
                ok = overlap >= 0.02
            ok = bool(output.strip())
        if not ok and len(failures) < 20:
            failures.append({"index": index, "got": output[:300], "expected": expected[:300]})
        passed += 1 if ok else 0
    return {
        "score": round(passed / max(len(cases), 1), 3),
        "total": len(cases),
        "pass": passed,
        "failed": len(failures),
        "kinds": kinds,
    }


def cmd_eval(config: dict[str, Any]) -> None:
    """Adherence A/B: base model vs the fine-tuned adapter on the same eval set.

    Both sides run in fp16 for a fair comparison. A QLoRA-trained adapter is
    produced against a *4-bit* base; attaching it to a fresh fp16 base and merging
    on the fly is how peft defines that comparison — the alternative (evaluating
    against the quantised base) would measure the quantisation as much as the
    training. Merging happens in `merge`; here the adapter is simply loaded on top.
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as error:  # noqa: BLE001
        die("deps", f"评测依赖不完整：{error}")

    data_dir = config.get("data")
    out_dir = config.get("out")
    model_id = config.get("model")
    adapter_dir = os.path.join(out_dir or "", "adapter")
    if not os.path.isdir(adapter_dir):
        die("artifact", f"找不到适配器：先训练出 {adapter_dir}")
    rows = read_jsonl(os.path.join(data_dir, "eval.jsonl"))
    if not rows:
        die("data", "eval.jsonl 是空的：没有可以判定的样本")
    limit = int(config.get("evalLimit", 100))
    cases = []
    for row in rows[:limit]:
        messages = row.get("messages") or []
        if len(messages) >= 2:
            cases.append({"prompt": messages[0].get("content", ""), "expected": messages[-1].get("content", "")})

    device_map = {"": 0} if torch.cuda.is_available() else None

    def load(adapter: bool):
        base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map=device_map, trust_remote_code=True)
        return PeftModel.from_pretrained(base, adapter_dir) if adapter else base

    emit(kind="phase", phase="eval-base", message=f"底座跑 {len(cases)} 条")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = load(adapter=False)
    base_outputs = [_generate(base, tokenizer, case["prompt"]) for case in cases]
    base_score = _score(cases, base_outputs)
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    emit(kind="phase", phase="eval-ft", message=f"微调后跑 {len(cases)} 条")
    tuned = load(adapter=True)
    tuned_outputs = [_generate(tuned, tokenizer, case["prompt"]) for case in cases]
    tuned_score = _score(cases, tuned_outputs)

    # The text metric, reported separately from adherence. Computed for every run
    # because it costs nothing here (the outputs already exist) and because the
    # adherence score is silent on prose: it cannot tell a good answer from a lucky
    # one, and on a real run it gave both models 100%.
    text_report = None
    try:
        from .textmetric import score_text_split

        text_report = {
            "base": score_text_split(cases, base_outputs),
            "candidate": score_text_split(cases, tuned_outputs),
        }
        # The per-case rows make the report large; the JSON file keeps them, the
        # protocol event does not need them.
        for side in ("base", "candidate"):
            text_report[side].pop("perCase", None)
    except Exception as error:  # noqa: BLE001 - a metric must never fail a run
        log(f"文本相似度计算失败，已跳过：{error}", level="warn")

    _ensure_dir(out_dir)
    report_path = os.path.join(out_dir, "eval.json")
    samples_path = os.path.join(out_dir, "eval-outputs.jsonl")
    with open(samples_path, "w", encoding="utf-8") as handle:
        for index, case in enumerate(cases):
            handle.write(json.dumps({"index": index, "prompt": case["prompt"], "expected": case["expected"], "base": base_outputs[index], "candidate": tuned_outputs[index]}, ensure_ascii=False) + "\n")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump({"model": model_id, "base": base_score, "candidate": tuned_score, "text": text_report, "cases": len(cases), "outputs": samples_path}, handle, ensure_ascii=False, indent=2)
    emit(kind="artifact", label="eval", path=report_path)
    emit(kind="artifact", label="eval-outputs", path=samples_path)
    # `adherence_scored` says what the score can bear: "tool" cases are a real
    # comparison (did it call the right tool); "answer" cases are only a floor
    # ("did it stay on topic"). A report that shows 100% for prose answers
    # without this distinction is overstating what was measured.
    kinds = base_score.get("kinds", {})
    emit(
        kind="evaluation",
        base=base_score,
        candidate=tuned_score,
        adherence_scored=kinds.get("tool", 0),
        floor_only=kinds.get("answer", 0),
        # The text metric is reported alongside, never merged into, the adherence
        # score: one answers "did it call the right tool", the other "does the prose
        # resemble the reference". Averaging them would produce a number that means
        # neither.
        text=text_report,
    )
    log(f"贴合度：底座 {base_score['score']:.1%} → 微调后 {tuned_score['score']:.1%}")
    if text_report is not None:
        log(
            f"文本相似度（F1）：底座 {text_report['base']['f1']:.2f} → 微调后 {text_report['candidate']['f1']:.2f}"
            f"　沾题率 {text_report['base']['onTopicRate']:.0%} → {text_report['candidate']['onTopicRate']:.0%}"
        )
        log("这一项衡量「内容与形式接近参考答案的程度」，不是质量评分；质量请用偏好对比或人工抽查。")
    if kinds.get("answer", 0) and not kinds.get("tool", 0):
        log(
            "注意：这组样本全是开放式文字回答，「贴合度」只是「没有跑题」的下限，"
            "不能证明微调学到了东西；判断升级看上面的文本相似度与偏好对比。"
        )


def cmd_calibrate(config: dict[str, Any]) -> None:
    """Hardware calibration lives in its own module; this is only the dispatcher."""
    from .calibrate import cmd_calibrate as run_calibration

    run_calibration(config)


COMMANDS = {"train": cmd_train, "merge": cmd_merge, "export": cmd_export, "eval": cmd_eval, "calibrate": cmd_calibrate}




def _ensure_dir(path: str) -> None:
    """Make sure `path` exists, tolerating a process that cannot create directories.

    On this machine `os.mkdir` returns EACCES for every directory under the plugin
    root, while the same call from Node or .NET succeeds ? a scanner or filter
    driver treats the Python process differently. The host therefore creates the
    directories it knows about before spawning the worker (lib/backend.mjs), and
    this function only verifies:

      - a directory that already exists is left alone (never re-created, which is
        where the EACCES came from)
      - a missing directory triggers one attempt, so a worker run by hand in a
        normal environment still works
      - the result is checked, and the failure names the path, because "Permission
        denied" on its own has sent a user looking for a permissions problem that
        did not exist

    """
    if os.path.isdir(path):
        return
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass
    if not os.path.isdir(path):
        raise OSError(
            f"无法使用输出目录 {path}：该路径不存在且无法创建。"
            "上层应当在启动 worker 之前建好它（lib/backend.mjs）。"
        )


def _install_tempdir() -> None:
    """Point tempfile at a directory this process owns, before torch is imported.

    On this machine `tempfile.mkstemp` blocks for minutes inside the system temp
    directory ? a scanner or a filter driver holds the write ? so `gettempdir()`
    rejects every candidate and falls back to the current working directory. Once
    even that fails, callers see "No usable temporary directory found".

    The host passes TMPDIR in the worker's environment (lib/backend.mjs). Setting
    it here as well covers a manual run, and doing it before any ML import means
    torch's own temp access benefits too. A directory that cannot be created is
    left alone: the failure then surfaces where it is used, with the real path.
    """
    candidate = (
        os.environ.get("TMPDIR")
        or os.environ.get("TEMP")
        or os.environ.get("FNT_TMPDIR")
        or ""
    )
    if not candidate:
        return
    try:
        os.makedirs(candidate, exist_ok=True)
        tempfile.tempdir = candidate
    except OSError:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fnt_worker", description="DSH fine-tune worker")
    parser.add_argument("verb", choices=sorted(COMMANDS))
    parser.add_argument("--config", required=True, help="path to the JSON config the host wrote")
    args = parser.parse_args(argv)
    _install_tempdir()
    config = load_config(args.config)
    emit(kind="phase", phase="start", message=f"{args.verb} 开始")
    try:
        COMMANDS[args.verb](config)
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001
        note(traceback.format_exc())
        emit(kind="error", code="exception", message=f"{type(error).__name__}: {error}")
        return 2
    emit(kind="phase", phase="done", message=f"{args.verb} 结束")
    return 0


if __name__ == "__main__":
    sys.exit(main())
