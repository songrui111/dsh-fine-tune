# dsh-fine-tune · a fine-tuning bench for DSH

**A fine-tuning pipeline for DSH: mine (scenario → right action) samples from real session logs, audit and dedupe them, run QLoRA on a local or remote GPU, then A/B the result against the base model before anything is published.**

[中文](README.zh-CN.md) · MIT · [Design](DESIGN.md) · [M0 lab](M0-LAB.md)

## Why

Training scripts are not the scarce thing. The chain around them is: what to train on, whether the data is clean, and whether the result is actually better.

1. **What to train on** — a DSH session log is already a corpus of `(situation → right action)`, mixed with failed retries, invented tool names and private data.
2. **Whether it is clean** — a dataset gets uploaded and baked into weights; redaction and dedupe have to happen *before* training, with evidence (a data card).
3. **Whether it is better** — "is this answer good" is not decidable, but "is the tool name right, are the arguments right, was a tool invented" is. A regression blocks publishing.

## What you get

One sidebar seat, five model-facing tools, laid out as data → train → evaluate.

| Entry | For |
|---|---|
| `train_prepare` | Mine sessions or a local JSONL into a dataset + data card (kept / deduped / every drop reason / redaction counts / token spread) |
| `train_check` | Will it fit here? GPU, component-by-component memory estimate, the smallest change that makes it fit |
| `train_run` | Start QLoRA — refuses before spawning when the estimate does not fit |
| `train_status` | State, loss, artifacts, evaluation verdict |
| `train_eval` | How this run is judged, and what the last verdict was |
| **Panel** | Session picking, data card, memory gauge, live loss curve, runs, verdict, publish prep |

## Install

```sh
dsh plugin --profile desktop add file:C:/path/to/fine-tune
```

Restart the profile. **Read-only by default**: browsing data and probing the GPU work; training and publishing need their switches.

## Switches

| Switch | Default | Opens |
|---|---|---|
| `allowSpawn` | off | Starting training, probing Python |
| `allowPublish` | off | Publish prep, and only with a green verdict plus a `.gguf` |
| `allowRemotePanel` | off | Serving the panel beyond loopback |

## Config

```yaml
- id: fine-tune
  config:
    pythonPath: C:\\path\\to\\venv\\Scripts\\python.exe
    workerRoot: C:\\path\\to\\fine-tune\\python   # default: the bundled python/
    dataDir: ""                                  # default: ~/.dsh/fine-tune
    allowSpawn: false
    allowPublish: false
    nvidiaSmi: nvidia-smi
    sshTarget: ""                                # M4: remote training
```

## Honest notes

- **Whether a fine-tune actually helps is unverified until M0 is run.** On a 6 GB card: 1.5B is comfortable, 4B is worth trying, 7B is refused. See [M0-LAB.md](M0-LAB.md).
- Adherence is a **mechanical check** (tool name / arguments / format), not a taste score.
- The **remote SSH backend's execution hook is not wired** to the host yet; only the local backend runs. See [DESIGN.md](DESIGN.md) §10.
- Session decoding depends on clearing one reserved bit in the first zstd frame. A log that cannot be decoded says so, and is never counted as an empty session.

## Develop

```sh
npm test                                              # 31 unit + route-level host + client bundle
cd python && python -m unittest discover -s tests      # worker protocol (8)
```

No build step, no runtime dependencies.

## License

MIT
