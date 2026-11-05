# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Remote training over SSH.** A run can execute on another machine's GPU.
  Datasets and the worker package are uploaded before the run and the adapter is
  pulled back after it. Uses the system `ssh`/`sftp` so the user's existing
  `~/.ssh/config`, keys and ProxyJump keep working; host-key verification is
  never configurable from plugin settings.
- `scripts/setup-remote.sh` — prepares and *verifies* a Linux host in one command
  (`--check` verifies without installing). The GPU test is a real allocation and
  matmul plus an nf4 round-trip, because `torch.cuda.is_available()` can be true on
  a driver that cannot run a kernel.
- The training panel gained the LLaMA-Factory parameter set: LR scheduler,
  warmup, optimiser choice, weight decay, grad clipping, Adam β2, seed, rsLoRA,
  DoRA, target-module preset, precision, gradient checkpointing, FlashAttention-2,
  packing, loss masking, early stopping, merge and GGUF export.
- Hardware capabilities are reported to the panel (`capability` action) so knobs
  the machine cannot honour are shown disabled **with the reason**, not hidden.
- `npm run sync` — copies the source into the DSH profile that loads it. The
  profile installs a *copy*, so editing the source alone changes nothing.
- `npm run test:ssh` and `npm run test:ssh-train` — end-to-end transport and
  full-remote-run tests against a throwaway local SSH server.

### Fixed

- **The memory estimate under-reported peak usage by roughly 20×.** It omitted the
  logits term (`vocab × seq × 4` bytes, upcast to fp32 for the loss and kept until
  backward) and carried an activation coefficient of 4 bytes/token/layer/hidden
  against a measured ~210. Both are corrected and the estimator now lands within
  +10–17% of two real measurements (`M0-LAB.md` §7).
- **A report of 100% adherence that measured nothing.** Prose answers were scored
  as "non-empty passes", so base and fine-tuned both scored 100%. `eval` now emits
  `adherence_scored` (tool-call cases, a real comparison) and `floor_only` (prose
  cases, a floor) separately.
- **92% of usable training data was discarded.** Tool calls were checked against a
  registry that listed 5 tools while the session used 14, so real calls were
  dropped as hallucinations. A call is now judged by what the tool layer returned:
  38 → 2737 samples on the same logs.
- **Samples whose answer did not fit the sequence were silently kept** with a fully
  masked label, producing `grad_norm = 0` and a NaN eval loss. They are dropped and
  counted now.
- `DataCollatorForSeq2Seq` cast `input_ids` to float, which an embedding lookup
  rejects; replaced with an explicit padding collator.
- Validation had no batch size, so a run could train successfully and then OOM
  during evaluation.
- A LoRA adapter hard-coded Qwen's module names, which is wrong for Llama, Falcon
  and ChatGLM; it now uses `all-linear`.

### Changed

- `M0-LAB.md` records what the first real run measured, including the conclusions
  it overturned (1.5B does not do a *useful* run on 6 GB; open-ended prose has no
  decidable adherence metric).

## [0.1.0]

### Added

- Initial plugin: session-log mining with redaction and dedupe, dataset cards and
  preview, QLoRA and LoRA training on a local GPU, adherence evaluation, a
  four-step preference-comparison flow, and the publish gate.
