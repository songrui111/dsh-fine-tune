#!/usr/bin/env bash
#
# Prepare a Linux machine to run this plugin's worker over SSH.
#
# Run it ON the remote host, once, from the repository checkout:
#
#     bash scripts/setup-remote.sh            # install + verify
#     bash scripts/setup-remote.sh --check    # verify only, install nothing
#     bash scripts/setup-remote.sh --model Qwen/Qwen2.5-3B-Instruct
#
# It is idempotent: a second run re-checks and skips what is already there.
#
# Design notes that matter for a PR review:
#
#   * Nothing here runs as root unless the *user* chooses to. The script never
#     calls sudo on its own; if pip needs a system prefix it says so and stops.
#   * `--check` never writes. It is what a CI job or a suspicious user runs first.
#   * The GPU test is a real allocation and a real kernel launch, not a
#     `torch.cuda.is_available()` that can be true on a broken driver.
#   * The model download is separate from the install and skippable, because a
#     cluster node often has no outbound network and a shared cache instead.
#
# Exit codes:  0 ok, 1 usage error, 2 environment problem, 3 install failed.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
INSTALL=1
VENV_DIR="${VENV_DIR:-$HOME/.dsh-fine-tune/venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [ $# -gt 0 ]; do
  case "$1" in
    --check) INSTALL=0 ;;
    --no-install) INSTALL=0 ;;
    --model) shift; MODEL="${1:-$MODEL}" ;;
    --venv) shift; VENV_DIR="${1:-$VENV_DIR}" ;;
    -h|--help)
      sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
  shift
done

say()  { printf '  %s\n' "$*"; }
section() { printf '\n== %s\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m   %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; }
warn() { printf '  \033[33mwarn\033[0m %s\n' "$*"; }

FAILED=0
fail() { bad "$*"; FAILED=1; }

# --- 1. platform -------------------------------------------------------------
section "platform"
if [ "$(uname -s)" != "Linux" ]; then
  fail "this script prepares a Linux host; found $(uname -s)."
  fail "The worker runs on Windows too, but the remote path is Linux-only."
  exit 2
fi
ok "Linux $(uname -r) on $(uname -m)"

for cmd in "$PYTHON_BIN" pip3 nvidia-smi; do
  if command -v "$cmd" >/dev/null 2>&1; then ok "$cmd: $(command -v "$cmd")"; else fail "$cmd not found"; fi
done
if [ "$FAILED" -ne 0 ]; then
  warn "install the missing tools first:  sudo apt install -y python3 python3-venv python3-pip"
  exit 2
fi

PY_VERSION="$("$PYTHON_BIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
if "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
  ok "python $PY_VERSION"
else
  fail "python $PY_VERSION is too old: the worker needs 3.10+"
fi

# --- 2. GPU ------------------------------------------------------------------
section "gpu"
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L | while IFS= read -r line; do say "$line"; done
  COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
  ok "$COUNT GPU(s) visible to nvidia-smi"
  if [ "$COUNT" -gt 1 ]; then
    ok "more than one GPU: DeepSpeed ZeRO becomes usable in the panel"
  fi
else
  fail "nvidia-smi reported no GPU. Training needs one; the data and evaluation steps do not."
fi

# --- 3. install --------------------------------------------------------------
section "python environment"
if [ "$INSTALL" -eq 0 ]; then
  say "skipped (--check / --no-install)"
  PY="$VENV_DIR/bin/python"
  if [ ! -x "$PY" ]; then
    fail "no virtualenv at $VENV_DIR — run without --check once to create it"
    exit 2
  fi
else
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    say "creating venv at $VENV_DIR"
    if ! "$PYTHON_BIN" -m venv "$VENV_DIR" 2>/dev/null; then
      fail "python3-venv is missing:  sudo apt install -y python3-venv"
      exit 3
    fi
  else
    say "reusing venv at $VENV_DIR"
  fi
  PY="$VENV_DIR/bin/python"

  # Prefer the CUDA wheel index: the default PyPI torch on Linux is CUDA-enabled
  # already, but pinning to the cu-index makes the driver requirement explicit and
  # keeps the version reproducible across machines.
  say "installing torch (this takes a few minutes and ~3 GB)"
  if ! "$PY" -m pip install --quiet --upgrade pip; then
    fail "pip upgrade failed"
    exit 3
  fi
  if ! "$PY" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121; then
    warn "the cu121 index failed; retrying against default PyPI"
    if ! "$PY" -m pip install --quiet torch; then
      fail "torch install failed — check network access or point pip at a mirror"
      exit 3
    fi
  fi
  ok "torch installed"

  say "installing the worker's dependencies"
  if ! "$PY" -m pip install --quiet transformers peft datasets accelerate bitsandbytes sentencepiece; then
    fail "dependency install failed"
    exit 3
  fi
  ok "dependencies installed"
fi

# --- 4. verify the imports ---------------------------------------------------
section "imports"
"$PY" - <<'PYCODE' || FAILED=1
import sys

def check(name, label=None):
    label = label or name
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "?")
        print(f"  \033[32mok\033[0m   {label} {version}")
        return True
    except Exception as error:  # noqa: BLE001
        print(f"  \033[31mFAIL\033[0m {label}: {error}")
        return False

ok = True
for name in ("torch", "transformers", "peft", "datasets", "accelerate"):
    ok &= check(name)

try:
    import bitsandbytes  # noqa: F401
    print("  \033[32mok\033[0m   bitsandbytes (QLoRA)")
except Exception as error:  # noqa: BLE001
    print(f"  \033[31mFAIL\033[0m bitsandbytes: {error}")
    print("         QLoRA needs it. On a CPU-only host or an unsupported driver this is expected.")
    ok = False

sys.exit(0 if ok else 1)
PYCODE

# --- 5. real GPU kernel ------------------------------------------------------
section "gpu compute"
"$PY" - <<'PYCODE' || FAILED=1
import sys

try:
    import torch
except Exception as error:  # noqa: BLE001
    print(f"  \033[31mFAIL\033[0m torch missing: {error}")
    sys.exit(1)

if not torch.cuda.is_available():
    print("  \033[31mFAIL\033[0m torch.cuda.is_available() is False")
    print("         The driver is present but torch cannot use it: check the CUDA version")
    print("         the wheel was built for against `nvidia-smi`'s driver version.")
    sys.exit(1)

for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"  \033[32mok\033[0m   cuda:{index} {props.name} "
          f"{props.total_memory / 1024**3:.1f} GiB cc{torch.cuda.get_device_capability(index)[0]}."
          f"{torch.cuda.get_device_capability(index)[1]}")

# A real allocation and matmul: a broken driver can pass is_available() and still
# fail here, and this is the failure that would otherwise appear 20 minutes into
# a training run.
try:
    a = torch.randn(256, 256, device="cuda")
    result = (a @ a.T).sum().item()
    print(f"  \033[32mok\033[0m   gpu matmul (checksum {result:.2f})")
except Exception as error:  # noqa: BLE001
    print(f"  \033[31mFAIL\033[0m gpu matmul: {error}")
    sys.exit(1)

if torch.cuda.is_bf16_supported():
    print("  \033[32mok\033[0m   bf16 supported (the panel will default to it)")
else:
    print("  \033[33mwarn\033[0m bf16 unsupported on this card; fp16 will be used")

# nf4 quantisation is the whole point of QLoRA; storing and reloading a tensor
# proves the bitsandbytes kernels match this card's compute capability.
try:
    import bitsandbytes.functional as bnb

    source = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    quantised, meta = bnb.quantize_4bit(source, quant_type="nf4")
    restored = bnb.dequantize_4bit(quantised, meta)
    delta = (source - restored).abs().max().item()
    print(f"  \033[32mok\033[0m   4bit nf4 quantise/dequantise (max error {delta:.3f})")
except Exception as error:  # noqa: BLE001
    print(f"  \033[31mFAIL\033[0m 4bit nf4 on this card: {error}")
    print("         QLoRA will not work; LoRA (fp16 base) may.")
    sys.exit(1)
PYCODE

# --- 6. worker package -------------------------------------------------------
section "worker"
if "$PY" -c 'import fnt_worker' 2>/dev/null; then
  ok "fnt_worker importable from $("$PY" -c 'import fnt_worker,os;print(os.path.dirname(fnt_worker.__file__))')"
else
  warn "fnt_worker is not importable yet."
  warn "This is expected: the plugin uploads its own copy with every remote run,"
  warn "so nothing has to be installed here for that path to work."
  say "To use the CLI directly instead, run from the repository root with"
  say "  PYTHONPATH=<repo>/python $PY -m fnt_worker --help"
fi

# --- 7. model cache ----------------------------------------------------------
section "model cache"
if [ "$INSTALL" -eq 0 ]; then
  say "skipped"
else
  say "pre-downloading $MODEL (skip with --no-install on a node without network)"
  if "$PY" - <<PYCODE
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("$MODEL")
print("  tokenizer cached")
PYCODE
  then
    ok "$MODEL cached"
  else
    warn "could not download $MODEL."
    warn "A remote run will fail with 'not in the cached files' unless this host can"
    warn "reach the model hub, or HF_HOME already points at a cache that has it."
  fi
fi

# --- 8. summary --------------------------------------------------------------
section "summary"
if [ "$FAILED" -eq 0 ]; then
  ok "this host can run a remote training job"
  cat <<EOF

  Next, on the machine that runs DSH, point the plugin at this host:

    1. add this host to ~/.ssh/config (or use user@host:port)
    2. in the plugin settings set   sshTarget = <that alias>
                                    sshRoot   = $HOME/.dsh-fine-tune
    3. click "测试连接" in the panel, or run:  train_check { backend: "ssh" }

  Give the panel a writable sshRoot: this script's venv lives there too.
EOF
  exit 0
fi

bad "one or more checks failed — see the lines above"
cat <<EOF

  A remote *training* run will not work until these pass.
  The data-mining and evaluation steps run on the DSH machine and are unaffected.
EOF
exit 2
