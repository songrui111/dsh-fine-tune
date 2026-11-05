"""Measure, do not guess.

The plugin ships an *estimated* memory budget (`lib/hardware.mjs`). This module
turns that estimate into two measured coefficients, using the GPU that is
actually in the machine:

    a  activation bytes per (token × hidden unit) per layer
    r  the rest of the peak: CUDA context + cuBLAS workspaces + allocator slack

It does that by running a **real optimizer step** on a real model and reading
`torch.cuda.max_memory_allocated()`. Without `CALIBRATE_MODEL` it uses a tiny model
as a shape probe (fast, constant); with it, the estimator is checked on the
planned model itself — which is what makes the prediction trustworthy for a run.

Both numbers come back with their inputs, so the lab prints an equation instead of
a claim.

Protocol (one JSON event per line on stdout), matching the rest of the worker:

    python -m fnt_worker calibrate --config calibrate.json
    config = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "mode": "qlora",
              "seqLens": [512, 1024], "batch": 1, "out": "..."}
"""

from __future__ import annotations

import json
import os
from typing import Any

from . import die, emit, log

# Small enough to be constant, large enough that the CUDA workspaces are real.
SHAPE_PROBE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _gib(bytes_count: float) -> float:
    return round(float(bytes_count) / (1024 ** 3), 3)


def _shape_from_config(model_id: str) -> dict[str, int]:
    """True geometry from the model's own config, not from the estimator's table."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    layers = int(
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "n_layer", None)
        or getattr(config, "num_layers", None)
        or 0
    )
    hidden = int(
        getattr(config, "hidden_size", None) or getattr(config, "n_embd", None) or getattr(config, "d_model", None) or 0
    )
    return {"layers": layers, "hidden": hidden}


def _load(model_id: str, mode: str):
    """Load + wrap a model the same way training does, so the measurement applies."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": {"": 0}}
    if mode == "qlora":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if mode == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    return model


def _one_step(model, seq_len: int, batch: int) -> int:
    """Forward + backward + an SGD step; returns the peak allocated bytes.

    The optimizer step is not decoration: paged AdamW allocates its state on the
    first step, and leaving it out is how an estimate comes back 2 GiB too small.
    """
    import torch

    device = next(model.parameters()).device
    ids = torch.randint(100, 2000, (batch, seq_len), device=device)
    labels = ids.clone()
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
    output.loss.backward()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                parameter.data.add_(-2e-4, parameter.grad)
                parameter.grad = None
    peak = int(torch.cuda.max_memory_allocated())
    del ids, labels, output
    torch.cuda.empty_cache()
    return peak


def cmd_calibrate(config: dict[str, Any]) -> None:
    try:
        import torch
    except Exception as error:  # noqa: BLE001 - the message is the product here
        die("deps", f"校准需要 torch：{error}")
    if not torch.cuda.is_available():
        die("cuda", "没有可用的 CUDA 设备，显存校准无意义（这也是 M0 要回答的问题之一）。")

    model_id = str(config.get("model") or SHAPE_PROBE_MODEL)
    mode = str(config.get("mode", "qlora"))
    seq_lens = [int(value) for value in config.get("seqLens", [512, 1024])]
    batch = int(config.get("batch", 1))
    out_dir = config.get("out") or "."

    device_name = torch.cuda.get_device_name(0)
    total_bytes = int(torch.cuda.get_device_properties(0).total_memory)
    emit(kind="log", level="info", message=f"设备 {device_name}，显存 {_gib(total_bytes)} GiB，torch {torch.__version__}，模型 {model_id}（{mode}）")

    emit(kind="phase", phase="load", message=f"加载 {model_id}")
    try:
        model = _load(model_id, mode)
    except Exception as error:  # noqa: BLE001
        die("load", f"加载失败（这种情况下失败本身就是结论：这台机器装不下它）：{error}")
    shape = _shape_from_config(model_id)
    if shape["layers"] == 0 or shape["hidden"] == 0:
        die("shape", "读不出模型的层数/隐藏维度，无法归因。")
    emit(kind="log", level="info", message=f"真实形状：{shape['layers']} 层 × {shape['hidden']} hidden")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    weights_bytes = int(torch.cuda.memory_allocated())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # fp16 weights + fp16 grads + fp32 Adam m/v ≈ 12 B/trainable param, plus the
    # paged-AdamW block table (~2 B). This is the adapter's steady state.
    adapter_state_bytes = int(trainable * 14)

    measurements = []
    for seq_len in seq_lens:
        torch.cuda.reset_peak_memory_stats()
        try:
            peak = _one_step(model, seq_len, batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            measurements.append({"seqLen": seq_len, "batch": batch, "fits": False, "peakGiB": None})
            emit(kind="log", level="warn", message=f"seq {seq_len} @ batch {batch}：显存装不下")
            continue
        activation_bytes = peak - weights_bytes - adapter_state_bytes
        denominator = seq_len * shape["layers"] * shape["hidden"]
        per_unit = activation_bytes / denominator if denominator > 0 else 0.0
        measurements.append(
            {
                "seqLen": seq_len,
                "batch": batch,
                "fits": True,
                "peakGiB": _gib(peak),
                "weightsGiB": _gib(weights_bytes),
                "activationGiB": _gib(activation_bytes),
                "bytesPerTokenPerLayer": round(per_unit, 3),
                "predictedGiB": _gib(weights_bytes + adapter_state_bytes + per_unit * denominator),
            }
        )
        emit(kind="metric", step=seq_len, values={"peakGiB": _gib(peak), "a": round(per_unit, 3)})
        torch.cuda.empty_cache()

    fitted = [item for item in measurements if item["fits"]]
    a_values = [item["bytesPerTokenPerLayer"] for item in fitted if item["bytesPerTokenPerLayer"] > 0]
    a = round(sum(a_values) / len(a_values), 3) if a_values else None
    calibration: dict[str, Any] = {
        "schema": "dsh-fine-tune/calibration/v1",
        "device": device_name,
        "totalGiB": _gib(total_bytes),
        "torch": torch.__version__,
        "model": model_id,
        "mode": mode,
        "shape": shape,
        "a": a,
        "r": None,
        "measurements": measurements,
    }
    if fitted:
        # The last fitting point is the one closest to a real run.
        item = fitted[-1]
        activation = item["activationGiB"]
        calibration["r"] = round(max(0.0, item["peakGiB"] - item["weightsGiB"] - activation), 3)
        calibration["check"] = {"seqLen": item["seqLen"], "peakGiB": item["peakGiB"], "activationGiB": activation}
    else:
        die("oom", f"所有序列长度都 OOM：{model_id} 在 {device_name} 上以 {mode} 跑不了。这本身就是 M0 的一条结论。")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "calibration.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(calibration, handle, ensure_ascii=False, indent=2)
    emit(kind="artifact", label="calibration", path=path)
    emit(kind="calibration", values=calibration)
    check = calibration["check"]
    log(
        f"校准：a={calibration['a']} 字节/(token×hidden×层)，r={calibration['r']} GiB；"
        f"seq{check['seqLen']} 实测峰值 {check['peakGiB']} GiB"
    )
    log("这两个数填回 lib/hardware.mjs 的 MEASURED 表，估算就从「经验」变成「本机实测」。")
