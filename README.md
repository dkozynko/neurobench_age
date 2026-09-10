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

For a clean-server setup, public verification, private-input boundary, and
shutdown checklist, see [`docs/research/reproduction.md`](docs/research/reproduction.md).

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
statistical decision rule, and evidence boundaries are
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

## Secondary layer-wise extension

The layer-wise extension compares matched mean-linear probes at REVE layers
`-4`, `-3`, and `-2` with the final-layer (`-1`) baseline. It uses the same
sealed 75-subject MIPDB cohort and ten optimization seeds. The analysis is
exploratory: all three point estimates are positive, but their hierarchical
bootstrap intervals cross zero, so the extension does not establish a primary
layer superiority claim.

Aggregate-only outputs are retained under
`results/extensions/layerwise_probe_20260910/`. The analysis identity is
`f1f757ef53a8aad10615afd7fa0d6f8e2bc4e7f04ec33ce2d870f92e44706565`; the
asset-manifest identity is
`14b5afe5623f34fbaed4fcbe98ee8b59528769bf468bd1bcd83b941353a2be9d`.
Participant-level predictions and representation caches remain external.

## Secondary ds006780 transfer

The completed v5 extension evaluates the final-layer mean-linear and
rich-statistics residual heads at HBN training sizes 200 and 800 on a separate
126-subject ds006780 cohort. It contains 40 selected runs and 5,040 external
prediction identities; only aggregate evidence is retained under
`results/extensions/ds006780_external_v5/`. At `n=800`, the rich head improved
the Pearson point estimate by `0.0287` across all ten seeds, but the
subject-aware 95% interval was `[-0.0349, 0.0922]` (width `0.1272`) and failed
the predeclared precision threshold `0.06`. This is uncertainty-limited
secondary transfer evidence, not a stable-superiority claim or a replacement
for the primary MIPDB analysis.

## Audit and reproduction commands

The repository's contract and analysis checks can be run with:

```bash
uv run --frozen --with pytest --with matplotlib pytest -q
uv run --frozen --extra test python scripts/audit_layerwise_article.py
```

The layer-wise report can be recomputed only when the immutable external
artifact directory is available. All artifact, protocol, and output paths must
be absolute:

```bash
export PYTHONPATH="$PWD/src"
uv run --frozen --extra test --extra manuscript python scripts/analyze_layerwise_probe.py \
  --artifact-root /ABSOLUTE/PATH/layerwise_external_predictions_20260910 \
  --analysis-config "$PWD/configs/research/layerwise_probe_analysis.json" \
  --protocol "$PWD/configs/research/layerwise_probe.json" \
  --training-protocol "$PWD/configs/research/layerwise_probe_training.json" \
  --output /ABSOLUTE/OUTPUT/layerwise_analysis.json

uv run --frozen --extra test --extra manuscript python scripts/build_layerwise_assets.py \
  --analysis /ABSOLUTE/OUTPUT/layerwise_analysis.json \
  --output-root /ABSOLUTE/OUTPUT/layerwise_assets
```

The analysis CLI is exact-resume: an existing output is accepted only when its
content is byte-identical to the recomputed report. The asset builder consumes
only the aggregate report and rejects canonical or manuscript-generated output
roots.

Only compact canonical evidence is retained here. Participant-level predictions,
checkpoints, caches, raw recordings, and logs remain outside Git.

The latest code-versus-evidence readiness boundary is recorded in
[`docs/research/readiness_audit.md`](docs/research/readiness_audit.md).
