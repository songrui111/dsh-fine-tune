# Contributing

Thanks for looking. This plugin touches training data, GPUs and a user's shell, so
the bar for a change is not "it works on my machine" — it is "someone else can
verify it". The notes below are what gets checked in review.

## Before opening a pull request

```bash
npm test              # unit + host routes + client bundle
npm run test:py       # worker protocol tests
npm run test:ssh      # real SSH+SFTP, against a local throwaway server
npm run test:ssh-train  # full remote run on a local GPU (needs CUDA)
npm run sync          # only if you are testing against an installed profile
```

All of them must pass. `test:ssh-train` exits 77 (skip) without a GPU, which is
fine in CI but should be run locally before a change to `lib/backend.mjs`,
`lib/ssh.mjs`, `lib/transfer.mjs` or `python/fnt_worker/`.

## Rules that are not negotiable

**Never write a claim you cannot point at.**

The repository has been through two rounds of a memory estimator that was wrong
by 20×, and a panel that reported 100% adherence for answers that were empty. Both
looked fine. If you add a number to a doc, a log line or a UI label, say where it
was measured (`M0-LAB.md` §7 has the format) or mark it as an estimate.

**A failure is never a training label.**

`lib/mine.mjs` labels only what the environment endorsed: a successful retry, a
result the tool layer returned, a turn the user accepted. If you are tempted to
label "what the model should have done", that is a different feature and needs its
own justification.

**Shell out through the quoting helpers, not around them.**

`validateTarget`, `shellQuote` and `validateRemotePath` exist because these strings
reach a remote shell. A new call site that builds a command with template literals
will be rejected in review even if the value "looks safe".

**The panel is gated, not filtered.**

Advanced knobs the machine cannot honour are rendered disabled *with the reason*
(`capabilityOf` in `lib/training.mjs`). Do not hide them: a hidden knob teaches
nobody why it is unavailable.

## Adding a training parameter

Three places, and the validator is the source of truth:

1. `lib/training.mjs` — add it to `validateTrainingPlan`, with a default and a
   sentence for a bad value
2. `python/fnt_worker/__main__.py` — actually honour it
3. `client/client.js` — the form field, in the matching group

Then add a case to `test/unit.mjs`. A parameter accepted by the validator but
ignored by the worker is worse than one that is rejected.

## Adding a remote capability

These run the user's code on a machine they own, so they need:

- a path through `validateRemotePath` if it reaches `sftp -b`
- a case in `test/ssh-e2e.mjs`
- an entry in the `--check` output of `scripts/setup-remote.sh`, so a user can
  see before a run whether their host supports it

## Documentation

`DESIGN.md` records decisions *and their reversals* (§7.2 and §10 are the honest
parts). If a measurement contradicts something written there, fix the document in
the same pull request — leaving both versions in place is how the 20× error
survived two revisions.

## Commit messages

One line, imperative, saying what changed for the user:

```
fix: predict the logits term so the memory estimate stops under-reporting
feat: run training on a remote host over ssh
docs: record the M0 measurements and the two estimator corrections
```

## Licence

By contributing you agree your work is released under the MIT licence in
`LICENSE`.
