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
# Braindecode's pretrained REVE loader requires this optional dependency.
uv pip install 'safetensors==0.5.3'
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

`official_reve_subset.py` runs the official NeuralBench Age stack and changes
only the downstream head. By default, `--manifest` selects the fixed validated
subset used by the existing experiments. The mutually exclusive `--full-data`
mode leaves `Shirazi2024Hbn`'s official timeline discovery intact and lets the
NeuralBench Age task apply its own filters and splits to the complete
`--data-root`. The reference `mean_linear` path, backbone, optimizer, early
stopping, and preprocessing remain shared in both modes.

### Data regime and improvement claims

The fixed 500-subject manifest is a **screening** regime. Its results can rank
candidate heads within that subset, but they must not be described as official
full-data improvements. Before promoting a head, run a matched `mean_linear`
baseline with the same data mode, timeline/subject selection, split,
preprocessing, channel positions, source/config hashes, protocol, and seeds.
Report the per-seed deltas, sample standard deviation, and worst seed; one
strong seed cannot promote a head. Keep the final test sealed until the
candidate and all training choices are frozen.

The current matched selective full-data comparison is the authoritative head
decision: `mean_linear` scores `0.732363780339559`, while
`mean_rich_stats_residual` scores `0.7290825843811035` (delta
`-0.0032811959584554`, or `-0.448028%`). Therefore `mean_linear` remains the
official winner. The historical 500-subject rich-statistics gain is retained
as a promising screening result only.

### 1000-subject nested screening setup

The repository includes [`screening_1000_manifest.py`](screening_1000_manifest.py)
to build a deterministic 1000-subject manifest from the existing 500-subject
manifest. It preserves all 500 base rows, adds 50 new eligible recordings per
existing release, inherits the release-to-split mapping, and writes a manifest
SHA-256 plus an audit report. The generator fails closed on duplicate subjects
or recording paths, malformed release/path pairs, excluded subjects,
non-finite or short recordings, mixed release splits, insufficient eligible
recordings, or an existing output file.

Run it on the host containing the HBN recordings:

```bash
python screening_1000_manifest.py \
  --base-manifest /workspace/hbn_subset/age_medium_500_resting_manifest.csv \
  --data-root /workspace/neurobench_data_hl2 \
  --output-manifest /workspace/hbn_subset/manifests/age_medium_1000_nested_20260825.csv \
  --report /workspace/hbn_subset/manifests/age_medium_1000_nested_20260825.json \
  --target-subjects 1000
```

If the host's asynchronous `openneuro-py` transport cannot fetch large S3
objects, use the resumable direct-S3 downloader
[`download_hbn_resting_direct.py`](download_hbn_resting_direct.py). It queries
OpenNeuro for the current snapshot metadata and materializes all resting-state
files for R1--R10; completed files are verified by size and skipped on retry.
The matching Supervisor configuration is
[`supervisor_hbn_download_direct_20260825.conf`](supervisor_hbn_download_direct_20260825.conf).

To keep a screening host below the full-data footprint, pass `--manifest` to
the same downloader. It then materializes only the manifest's selected
subjects (plus small release-level metadata), while retaining the official
`R#/download/sub-*/eeg` layout and resumability:

```bash
python download_hbn_resting_direct.py \
  --data-root /workspace/neurobench_data_hl2 \
  --manifest /workspace/hbn_subset/manifests/age_medium_1000_nested_20260826.csv \
  --releases R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 \
  --workers 8
```

The prepared [`supervisor_screening_1000_mean_linear_20260825.conf`](supervisor_screening_1000_mean_linear_20260825.conf)
generates the manifest first and then runs the matched `mean_linear` control on
seeds 33/34/35 with strict validation-only evaluation. It intentionally omits
`--strict-final-test`; test data remain withheld. Candidate heads must reuse the
same generated manifest and output a separate directory under
`reve_head_experiments_screening_1000_nested_20260825/`.

For a candidate screen, reuse the same manifest with a fresh output directory
and validation-only seed 33:

```bash
python official_reve_subset.py \
  --manifest /workspace/hbn_subset/manifests/age_medium_1000_nested_20260825.csv \
  --data-root /workspace/neurobench_data_hl2 \
  --output-dir /workspace/hbn_subset/results/reve_head_experiments_screening_1000_nested_20260825 \
  --config /workspace/hbn_subset/results/reve_head_experiments_screening_1000_nested_20260825/HEAD_VARIANT_config.json \
  --head-variant HEAD_VARIANT \
  --evaluation-protocol strict \
  --seeds 33
```

Only after a candidate passes the declared validation gate should the same
frozen command be repeated for seeds 34/35. No candidate screen should add
`--strict-final-test`.

### Selective HBN task acquisition

For the Age experiment, the full HBN root is not required. The standalone
downloader fetches only `task-RestingState` from the eleven official HBN
releases. It keeps the official layout under `DATA_ROOT/R1/download` through
`DATA_ROOT/R11/download` and writes the current acquisition pointer to
`selective_task_provenance.json` plus its `.sha256` sidecar.

Validate one release first:

```bash
python selective_hbn_download.py \
  --data-root /workspace/neurobench_data_hbn \
  --releases R1 \
  --workers 8
```

Download and audit all releases:

```bash
python selective_hbn_download.py \
  --data-root /workspace/neurobench_data_hbn \
  --workers 8
```

The root provenance reports `complete=true` only after all R1–R11 releases
pass the file audit: each recording must be either an external `.set`/`.fdt`
pair or a valid embedded-data `.set`. A one-release or otherwise
partial acquisition is useful for validating the downloader but is rejected
by the benchmark runner. Existing downloaded files are preserved on provider,
audit, or serialization failures; the root JSON/sidecar is a mutable current
pointer and each benchmark seed receives its own create-only snapshot.

Run the rich-statistics residual head on this selective root:

```bash
python official_reve_subset.py \
  --selective-task \
  --data-root /workspace/neurobench_data_hbn \
  --output-dir /workspace/hbn_subset/results/selective_mean_rich_stats_residual \
  --config /workspace/hbn_subset/results/selective_mean_rich_stats_residual/config.json \
  --head-variant mean_rich_stats_residual \
  --evaluation-protocol strict \
  --strict-final-test \
  --seeds 33 34 35
```

Selective mode leaves official HBN timeline discovery intact, records both the
official per-run timeline provenance and the immutable acquisition snapshot,
and does not download the other HBN tasks.

Before an HBN run, exercise the installed official REVE/NeuralTrain interface
without downloading HBN recordings:

```bash
python official_reve_subset.py --smoke-head last_avg
python official_reve_subset.py --smoke-head last
python official_reve_subset.py --smoke-head all
```

### Evaluation protocol (strict by default)

New runs use a validation-only strict holdout by default. During training the
runner writes `epoch_validation_metrics.jsonl` and selects the checkpoint only
from `val/pearsonr`; training callbacks never iterate or read the test loader.
The seed
directory then contains an immutable `selection.json` with the checkpoint,
configuration, source provenance, validation-history, and SHA-256 hashes. A
manifest run records its manifest; a full-data run records
`full_data_provenance.json`, while a selective run records both
`selective_task_timeline_provenance.json` and its acquisition snapshot. Both
non-manifest modes leave manifest fields null. A strict
validation-only report has `test_status: withheld` and no test metric.

After the architecture and hyperparameters are frozen, a single predeclared
finalist may consume the holdout exactly once:

```bash
python official_reve_subset.py \
  --manifest PATH_TO/age_medium_500_resting_manifest.csv \
  --data-root /workspace/neurobench_data_hl2 \
  --output-dir /workspace/hbn_subset/results/strict_mean_linear \
  --config /workspace/hbn_subset/results/strict_mean_linear/config.json \
  --head-variant mean_linear \
  --evaluation-protocol strict \
  --strict-final-test \
  --seeds 33 34 35
```

To run a complete HBN root with the proposed rich-statistics residual head,
use `--full-data` instead of `--manifest` (or use `--selective-task` above for
the smaller RestingState-only acquisition):

```bash
python official_reve_subset.py \
  --full-data \
  --data-root /workspace/neurobench_data_hl2 \
  --output-dir /workspace/hbn_subset/results/full_mean_rich_stats_residual \
  --config /workspace/hbn_subset/results/full_mean_rich_stats_residual/config.json \
  --head-variant mean_rich_stats_residual \
  --evaluation-protocol strict \
  --strict-final-test \
  --seeds 33 34 35
```

The full-data run writes one `full_data_provenance.json` per seed. It records
the resolved data root, the exact timeline identities returned by the official
study iterator, their count, and the SHA-256 digest. This snapshot is audit
metadata only; it is never fed back as a replacement iterator. Use the same
command with `--head-variant mean_linear` to produce the full-data reference
for comparison with the rich-statistics head.

This gated run creates durable `test_started.json` and `test_completed.json`
markers, verifies that the selected checkpoint hash did not change, and records
the exact official result key `test/pearsonr` as the report's
`test_pearsonr`. A second test invocation in the same seed directory fails
closed. Each strict seed directory must also be empty before training; any
stale report, failure record, marker, config, or checkpoint makes the new
attempt fail closed instead of mixing evidence. Use
`--evaluation-protocol legacy` only for historical parity; it
explicitly enables `epoch_test_metrics.jsonl`, is labeled `legacy`, and cannot
support a strict holdout claim. `--strict-final-test` is rejected with legacy.

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
Each run writes `report.json`, a resolved NeuralBench config, a source/runtime
metadata sidecar, and provenance under `<output>/<variant>/seed<seed>/`.
Strict runs add `epoch_validation_metrics.jsonl` and `selection.json`; only a
gated strict finalist adds the two test markers and test result. Legacy runs
add `epoch_test_metrics.jsonl` and retain NeuralBench's raw diagnostic test
prediction artifacts.

The upstream query token is initialized exactly as in the pinned
`ReveClassifier`: it is a new seeded downstream `torch.randn` parameter, not a
parameter expected in the pretrained encoder checkpoint (for `last_avg` it is
kept as an unused parameter, exactly as upstream). The regression linear
follows the upstream `cls_wrapper` truncated-normal initialization
(`std=512**-0.5`, cutoff `3`, zero bias). Test Pearson printed after each epoch
is diagnostic only; checkpoint and head selection use validation Pearson.

The independent reproduction remains a separate lane. This work does not
import or modify `independent_pipeline.py`.
