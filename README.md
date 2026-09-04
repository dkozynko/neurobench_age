# NeuralBench Age — REVE study

This repository contains the code and compact evidence for the paper
**“Stability of age probing in REVE and the limits of increasingly complex
representation heads.”** It is not a general-purpose NeuralBench fork and it
does not store HBN data, pretrained weights, or raw training outputs.

The central question is whether age probing remains stable across random seeds
when the REVE encoder and official NeuralBench Age protocol are fixed, and
whether increasingly expressive representation heads produce reliable gains
over the matched `mean_linear` baseline.

See [`ARTICLE_SCOPE.md`](ARTICLE_SCOPE.md) for the inclusion policy and
[`docs/research/article_evidence_registry.md`](docs/research/article_evidence_registry.md)
for the claim-to-evidence map.

## Repository layout

```text
src/neurobench_age/core/       benchmark contract and evidence schemas
src/neurobench_age/heads/      REVE head implementations
src/neurobench_age/data/       manifests and data acquisition
src/neurobench_age/pipelines/  official and independent runners
src/neurobench_age/training/   optional train-only extensions
src/neurobench_age/analysis/   reproducible metrics and figures
configs/article/               frozen configurations
scripts/                       portable entry-point wrappers
tests/                         contract and regression tests
docs/research/                 protocol and experiment registry
results/canonical/             compact evidence only
```

## Installation

For contract tests and local dry runs:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
```

The official REVE integration is optional and requires the NeuralBench stack:

```bash
python -m pip install -e '.[reve]'
```

No installation command downloads HBN data or pretrained weights.

## Contract check

From the repository root:

```bash
PYTHONPATH=src python -m neurobench_age.core.baseline --dry-run
PYTHONPATH=src python -m pytest -q
```

The dry run checks the crop/window geometry, channel count, R5 holdout
invariant, regression interface, and official preprocessing contract.

## Canonical experiment

The canonical metadata-only manifest is included at
`results/canonical/data/age_medium_1000_nested.csv`; prepare the matching HBN
recordings outside this repository. Then run the validation-only pipeline with
explicit paths:

```bash
MANIFEST="$PWD/results/canonical/data/age_medium_1000_nested.csv" \
DATA_ROOT=/path/to/neurobench_data_hbn \
OUTPUT_ROOT=/path/to/article-results/validation \
./scripts/run_article_experiment.sh
```

The launcher defaults to the matched `mean_linear` baseline, strict protocol,
deterministic settings, and validation-only mode. Override the head and seeds
explicitly for a predeclared screen:

```bash
MANIFEST="$PWD/results/canonical/data/age_medium_1000_nested.csv" \
DATA_ROOT=/path/to/neurobench_data_hbn \
OUTPUT_ROOT=/path/to/article-results/screen \
HEAD_VARIANT=mean_layer_linear \
SEEDS='33' \
PHASE=screen \
./scripts/run_article_experiment.sh --layer-index -2
```

Use `PHASE=confirmation` for the held-back confirmation seeds; the launcher
selects `configs/article/confirmation.json` automatically unless `CONFIG` is
overridden explicitly.

The launcher cannot open the sealed test. A final test is allowed only after
the validation gate is audited and a single finalist is frozen; use the
official CLI with the explicit final-test flag only in that controlled step.

## Evidence analysis

Analyze complete validation or final-test run directories with:

```bash
./scripts/run_article_analysis.sh \
  /path/to/run/seed33 \
  /path/to/run/seed34 \
  /path/to/run/seed35 \
  --output-dir /path/to/article-results/analysis
```

The analysis reports Pearson, MAE, RMSE, R², per-seed variability, paired
wins/losses, worst-seed deltas, and subject-level bootstrap intervals. It
records the input paths and hashes so that tables and figures are
traceable to exact runs.

Before any sealed evaluation, verify the matched validation gate:

```bash
python scripts/check_article_gate.py \
  --baseline-run /path/to/baseline/seed33 \
  --baseline-run /path/to/baseline/seed34 \
  --baseline-run /path/to/baseline/seed35 \
  --candidate-run /path/to/candidate/seed33 \
  --candidate-run /path/to/candidate/seed34 \
  --candidate-run /path/to/candidate/seed35 \
  --output /path/to/article-results/final_gate.json
```

The gate reads validation evidence only, pairs runs by seed, requires at least
two candidate wins, and rejects test-contaminated run directories.

## Data and artifacts

The canonical manifest is metadata-only and the HBN recordings remain
external. Large checkpoints, raw predictions, launch logs, and historical
exploration outputs are intentionally excluded from Git. Compact summaries,
negative-result records, figures, and provenance required by the paper live
under `results/canonical/`.

The staged protocol is validation screen → held-back confirmation → one sealed
finalist test. Test metrics must never be used to select a head or tune a
hyperparameter. See
[`docs/research/article_ready_protocol.md`](docs/research/article_ready_protocol.md)
for the complete contract.
