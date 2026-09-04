# experiment evidence

Every new run should be treated as an evidence package, not only as a scalar
score. The public runner writes a schema-versioned `run_manifest.json`,
`complexity.json`, normalized `config.json`, `report.json`, and the validation
history. Strict validation also writes an immutable `selection.json`, a
train-only `analysis/train_age_reference.jsonl`, and selected-checkpoint
`predictions/validation.jsonl`, plus the actual `optimizer.json` and
`throughput.json` observed by Lightning. An authorized final-test run additionally
writes `test_started.json`, `predictions/test.jsonl`, and
`test_completed.json`.

## Lifecycle and holdout policy

The default mode is `validation_only`. It may select a checkpoint using
validation Pearson, but test data remain sealed and no test prediction file is
created. A final test requires both:

```text
--evaluation-mode final_test --allow-sealed-test-evaluation
```

`--strict-final-test` remains a compatibility alias for the final-test mode,
but the explicit authorization flag is still required. The start marker is
written before test evaluation and binds the test pass to the selection and
checkpoint hashes. The completion marker is written only after the checkpoint
hash and test metric have been verified.

## Required reproducibility fields

`run_manifest.json` records the run ID, seed, task, dataset and split
fingerprints, normalized config hash, protocol digest, comparison-config hash,
declared comparison-factor keys, command line, Git state, a source-tree
SHA-256, host/software versions, deterministic settings and policy, precision,
evaluation mode, test access, timestamps, and a concrete `missing` list. A run
with an interrupted or unavailable evidence phase must not silently be
presented as complete. scripts must pass `--deterministic`; runs
without it are explicitly labeled best-effort and are not bitwise-reproducible
claims.

`complexity.json` separates encoder, head, and auxiliary parameters into
total/trainable/frozen counts and enforces the invariant
`total = encoder + head + auxiliary`. It also records head operations and the
parameter-count formula, training/validation phase durations, throughput,
peak memory, hardware class, and optional cost (provided with
`--gpu-hourly-rate-usd`). Hardware class is the exact
accelerator model plus VRAM tier; different GPU models must be labeled
`hardware_mixed` in confirmatory comparisons.

Some NeuralBench releases expose the downstream head outside the model object
seen by the runtime counter. In that case the declared
`head_complexity.parameter_count` is the authoritative head contract. The
audit may normalize a zero-sized raw head bucket with that
contract, but it must record an audit warning; it must never silently discard
the discrepancy.

## Metrics and statistical conventions

There are two explicitly named metric tracks. The official track is the native
NeuralBench callback metric used in `val/pearsonr` selection and the
`test_pearsonr` completion marker. The subject-level diagnostic track contains
one row per subject; if a subject has several views, the declared aggregation
is the arithmetic mean before Pearson, MAE, RMSE, and R² are computed. Never
average batch-level Pearsons, and never substitute the subject-level diagnostic
Pearson for the official benchmark value. The normalized analysis tables keep
both `test_pearsonr` (official marker) and `test_prediction_pearsonr`
(subject-level export), together with their metric sources.

For a candidate/baseline pair, define a seed win as a strictly positive
candidate-minus-baseline Pearson delta. Report the sample SD across seeds and
the minimum paired seed delta (`worst_seed_delta`). Paired subject bootstrap
uses 10,000 resamples, percentile 95% intervals, and analysis RNG seed
`20260903` unless an analysis manifest explicitly records another setting.

Age groups use quartile thresholds derived from unique training subjects only.
The thresholds and their hash are persisted in `analysis_spec.json` and reused
for validation/test rows. They must not be recomputed from validation or test
ages.

Checkpoint selection is bound to the official `val/pearsonr` maximization
callback; the selection artifact records that tie handling is delegated to
that callback rather than silently imposing a different post-hoc rule.
The audit checks the selected score against the matching validation-history
record. It does not require equality with a subject-level validation export,
because those files may use a different prediction unit.

## Reporting language

Three seeds provide useful evidence of run-to-run behavior but are not a
population-level stability proof. Use wording such as “evidence of
instability” when seed deltas disagree, and state the number of seeds. When a
head is more expensive but does not improve Pearson consistently, report that
the additional parameters and aggregation complexity did not yield a stable
gain proportional to their cost. Complexity-adjusted summaries expose both
the raw extra-parameter denominator and
`delta_per_extra_head_parameter`; the latter is null when the denominator is
zero or negative.
