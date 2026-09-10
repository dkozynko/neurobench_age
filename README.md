# NeuralBench Age — REVE study

This repository implements reproducible experiments on age information in REVE
EEG representations. The main question is whether increasingly expressive
representation heads improve reliably over a matched mean-pooled linear head
when the encoder, data split, preprocessing, checkpoint rule, and optimization
seeds are held fixed.

The primary prospective study freezes REVE, develops heads on HBN
train/validation subjects, and performs one sealed evaluation on an external
MIPDB cohort. Existing official NeuralBench full fine-tuning runs are retained
as secondary reproduction evidence. Previously opened HBN R5 results are
retrospective secondary evidence and must not be used for model or head
selection.

See [`ARTICLE_SCOPE.md`](ARTICLE_SCOPE.md) for the claim boundary and
[`docs/research/article_ready_protocol.md`](docs/research/article_ready_protocol.md)
for the execution contract.

## Repository layout

```text
src/neurobench_age/core/       benchmark contracts and evidence schemas
src/neurobench_age/heads/      REVE head implementations
src/neurobench_age/data/       metadata manifests and data adapters
src/neurobench_age/pipelines/  training, extraction, and evaluation logic
src/neurobench_age/research/   executable protocol and study lifecycle
src/neurobench_age/analysis/   reproducible statistical analysis
configs/article/               retained HBN reproduction configurations
configs/research/              prospective external-study configuration
scripts/                       command-line entry points
tests/                         contract and regression tests
docs/research/                 study design and evidence registry
results/canonical/             compact canonical evidence and artifact identities
```

Raw EEG, pretrained weights, representation caches, checkpoints, prediction
dumps, and launch logs must remain outside Git.

## Installation

For contract tests and synthetic verification:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

The official REVE integration and external BIDS tooling are optional:

```bash
python -m pip install -e '.[reve,external]'
```

No installation command downloads HBN, MIPDB, or pretrained weights.

## Supported experiment tracks

The prospective track is frozen representation probing. The encoder runs in
evaluation and inference mode, its parameters have no gradients, layers `-2`
and `-1` are cached once, and only the four predeclared heads are trained over
seeds 33 through 42.

The final head-training contract is explicit in
`configs/research/neuralbench_frozen_probe_training.json`: global seeded window
shuffling, AdamW, OneCycleLR, gradient clipping, 40 epochs, and patience 7.
This makes the study NeuralBench-compatible at the optimization level while
preserving the deliberate frozen-probe boundary. Final schema-3 artifacts
carry separate representation-protocol and training-protocol hashes, so the
stopped constant-learning-rate pilot cannot be mixed into the final study.

The secondary track is official NeuralBench full fine-tuning. Because the REVE
encoder is trainable in that track, it is described as end-to-end age
prediction rather than representation probing.

The exact operational sequence, executable commands, validation gates,
statistical decision rule, and result placeholders are
maintained in
[`docs/research/article_ready_protocol.md`](docs/research/article_ready_protocol.md).

## Current evidence status

- Prospective external cohort: 75 primary subjects after target-free QC, plus a
  separate 10-subject engineering pilot and 20-subject extrapolation cohort.
- Completed frozen-head runs: 40 (four heads across seeds 33--42).
- External prediction inventory: 3,000 exact head--seed--subject records;
  inventory identity `3ec0042d613fc2d36937d7349081beb0739c4fb3da349e1229e17f226e187830`.
- Confirmatory analysis identity:
  `7747a16e11629164b524b2113ab2230d260012022fc46f68540c7f7578c2a3b6`.
- Confirmatory conclusion: no tested complex head established a stable external
  gain under the predeclared joint rule; this does not establish equivalence.

## Secondary capacity--data extension

A separate aggregate-only extension tests whether the paired behavior of two
more expressive heads changes with nested HBN training sizes of 200, 400, and
800 subjects. It contains 90 frozen-head runs across three heads and ten
seeds, followed by a sealed external evaluation on the same 75-subject primary
cohort. The resulting 6,750 subject-level predictions remain outside Git.
Aggregate tables, figures, and their hash-chain manifest are retained under
`results/extensions/capacity_data_regime_v3/`. This extension is exploratory
and does not replace or modify the primary confirmatory evidence above.

Only compact canonical evidence is retained here. Participant-level predictions,
checkpoints, caches, raw recordings, and logs remain outside Git.

The latest code-versus-evidence readiness boundary is recorded in
[`docs/research/readiness_audit.md`](docs/research/readiness_audit.md).
