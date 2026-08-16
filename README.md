# NeuralBench Age: REVE baseline harness

This project captures the official NeuralBench Age and REVE configuration in
executable Python and exposes a real adapter to NeuralBench's official REVE
runner. The local dry run does not download HBN data or pretrained weights.

Run the local contract check from the repository root:

```bash
python -m neurobench_age.reve_baseline --dry-run
rtk python3 -m pytest -q tests/test_neurobench_age_parity.py
```

The dry run verifies:

- the 60-second crop and 2-second non-overlapping windows;
- the 129-channel, 400-sample raw window shape;
- the R5 test-release invariant;
- the one-output regression interface; and
- the official REVE preprocessing and linear-probe settings.

## Install the official REVE integration

Create an isolated environment and install the project plus the official
NeuralBench model stack:

```bash
uv venv
uv pip install -e '.[reve,test]'
```

The `reve` extra pins NeuralBench to version `0.2.3`. It installs the official
`NtReve` implementation and its runtime dependencies, but this project still
does not download HBN or model weights during installation.

The programmatic entry point is:

```python
from neurobench_age.reve_baseline import run_official_reve

# debug=True keeps the official runner in its reduced local mode.
# This call still requires prepared/downloaded task data.
results = run_official_reve(debug=True)
```

The equivalent command-line entry point is:

```bash
python -m neurobench_age.reve_baseline --official-run
```

Add `--download` and `--prepare` explicitly when you are ready to materialize
the Age task. Add `--debug` for a reduced local smoke run.

The adapter delegates to NeuralBench's public `run_benchmark` API with
`device='eeg'`, `task='age'`, and `model='reve'`; it does not reimplement or
silently replace the upstream REVE architecture.

## Run the independent reproduction

The independent runner does not call `neuralbench.run_benchmark`. It reads the
local HBN EEGLAB files, applies the same Age split and REVE preprocessing, loads
the same `brain-bzh/reve-base` encoder through `NtReve`, and trains the same
mean-pooling + linear regression head.

Install the complete runner in an isolated environment:

```bash
uv pip install -e '.[pipeline]'
```

After HBN is available under `DATA_ROOT/R1/download` through
`DATA_ROOT/R11/download`, run:

```bash
python -m neurobench_age.independent_pipeline \
  --data-root DATA_ROOT \
  --cache-dir DATA_ROOT/reve-preprocessed \
  --mapping PATH_TO/neuralbench/models/channel_mappings/reve.json
```

The command prints the independent test Pearson score. To compare it with the
official run's reported test score, pass that value explicitly:

```bash
python -m neurobench_age.independent_pipeline \
  --data-root DATA_ROOT \
  --cache-dir DATA_ROOT/reve-preprocessed \
  --mapping PATH_TO/neuralbench/models/channel_mappings/reve.json \
  --official-score OFFICIAL_TEST_PEARSON \
  --manifest-output DATA_ROOT/age-manifest.csv \
  --predictions-output DATA_ROOT/independent-test-predictions.csv \
  --official-predictions PATH_TO/official-test-predictions.csv
```

The runner defaults to the official Age configuration: 2 DataLoader workers,
the PyTorch default prefetch factor, persistent workers, pinned memory, and parent-process
preloading of aligned recordings. For a low-memory or single-process probe,
use `--num-workers 0 --no-preload`; worker-only options are then omitted from
PyTorch's DataLoader. The performance settings can be overridden with
`--num-workers`, `--prefetch-factor`, and `--no-persistent-workers`.

The comparison is only valid when both commands use the same prepared HBN
root, checkpoint, seed, 40-epoch configuration, and R5 test split. The cache
contains preprocessed continuous recordings and can be reused on later runs.
The manifest and prediction CSVs preserve release, subject, path, window start,
target age, and split so an official per-window export can be aligned and
checked with `--official-predictions`; the report then includes the prediction
comparison in addition to the aggregate score comparison.

The official reference commands remain:

```bash
neuralbench eeg age --download
neuralbench eeg age --prepare
neuralbench eeg age -m reve
```

## Upstream REVE head experiments

`official_reve_subset.py` runs the official NeuralBench Age stack on the fixed
manifest and changes only the downstream head. The existing `mean_linear`
path remains the NeuralBench reference. The new `last_avg`, `last`, and `all`
paths use the pinned upstream REVE classifier semantics; the backbone, data
split, optimizer, early stopping, and preprocessing remain shared.

Before an HBN run, exercise the installed official REVE/NeuralTrain interface
without downloading HBN recordings:

```bash
python official_reve_subset.py --smoke-head last_avg
python official_reve_subset.py --smoke-head last
python official_reve_subset.py --smoke-head all
```

Run one seed for a head comparison:

```bash
python official_reve_subset.py \
  --manifest artifacts/age_medium_500_seed33/manifest/age_medium_500_resting_manifest.csv \
  --data-root /workspace/hbn_subset \
  --output-dir artifacts/age_medium_500_seed33/reve_head_experiments \
  --config artifacts/age_medium_500_seed33/reve_head_experiments/config.json \
  --head-variant last_avg \
  --seeds 33
```

Use the same command with `last` or `all`. Confirmation runs use
`--seeds 33 34 35`; the report keeps model seed and fixed data seed separate.
Each run writes `report.json`, `epoch_test_metrics.jsonl`, a resolved
NeuralBench config, a source/runtime metadata sidecar, the selected checkpoint
hash, and NeuralBench's raw test prediction artifacts under
`<output>/<variant>/seed<seed>/`.

The upstream query token is initialized exactly as in the pinned
`ReveClassifier`: it is a new seeded downstream `torch.randn` parameter, not a
parameter expected in the pretrained encoder checkpoint (for `last_avg` it is
kept as an unused parameter, exactly as upstream). The regression linear
follows the upstream `cls_wrapper` truncated-normal initialization
(`std=512**-0.5`, cutoff `3`, zero bias). Test Pearson printed after each epoch
is diagnostic only; checkpoint and head selection use validation Pearson.

The independent reproduction remains a separate lane. This work does not
import or modify `independent_pipeline.py`.
