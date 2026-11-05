## What changed

<!-- One paragraph. What a user can do after this PR that they could not before. -->

## Why

<!-- The problem this solves. If it is a bug, what was the user-visible symptom? -->

## How it was verified

<!-- Tick what you ran. CI runs the first three; the rest need hardware. -->

- [ ] `npm test`
- [ ] `npm run test:py`
- [ ] `npm run test:ssh`
- [ ] `npm run test:ssh-train` (needs a CUDA GPU)
- [ ] `bash scripts/setup-remote.sh --check` on a real remote host
- [ ] Manual check in the panel

### Numbers, if you added any

<!--
If this PR states a memory figure, a percentage or a speed, say where it was
measured — model, sequence length, GPU, and the command. "Faster" without a
measurement is not reviewable, and this repository has already shipped a memory
estimate that was wrong by 20×.
-->

## Review checklist

- [ ] New strings that reach a shell go through `shellQuote` / `validateRemotePath`
- [ ] New training parameters are validated **and** honoured by the worker
- [ ] New remote capabilities appear in `setup-remote.sh --check`
- [ ] Documentation updated in the same PR if a measurement contradicted it
- [ ] No large binaries, virtualenvs or pip temp directories staged
