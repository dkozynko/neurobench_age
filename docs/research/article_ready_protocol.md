# REVE age study execution protocol

The executable representation configuration at
`configs/research/external_frozen_probe.json` and the separate head-training
configuration at
`configs/research/neuralbench_frozen_probe_training.json` are authoritative.
The first fixes the encoder, data, preprocessing, and cohort identities; the
second fixes the final NeuralBench-compatible optimization contract. This
document explains the evidence boundary, command sequence, current integration
status, and interpretation rules. Paths shown below are placeholders; raw data
and run artifacts must live outside the repository.

## Evidence boundary

The primary prospective study compares four heads trained on frozen REVE
representations. HBN train subjects fit head parameters, HBN validation subjects
select the earliest best checkpoint, and the sealed external holdout is MIPDB.
The matrix contains exactly four heads and seeds 33 through 42, giving 40 matched
runs.

Official NeuralBench full fine-tuning is secondary reproduction evidence. It is
end-to-end age prediction, not representation probing, because the encoder is
trainable. The primary study is intentionally not an exact end-to-end
NeuralBench rerun: it keeps the encoder frozen, excludes Cz, uses cached
representations, and selects checkpoints with subject-level validation
Pearson. Only the head optimization dynamics are aligned with NeuralBench.

Existing HBN R5 results are retrospective secondary evidence because R5 has
already been used in repeated finalist decisions. They must not be used for
model or head selection and cannot provide untouched confirmation.

## Fixed protocol

- Checkpoint: `brain-bzh/reve-base`, constructed with protocol seed `0` and
  hashed in full at execution time. Seeded construction is isolated from the
  training RNG stream and stabilizes the unused randomly initialized task
  layer included in the provenance hash.
- Cached layers: `-2` and `-1` from one deterministic frozen-encoder pass.
- Encoder input layout: exactly ordered `E1` through `E128` for both HBN and
  MIPDB. Extra HBN `Cz` is excluded; required channels cannot be padded or
  silently omitted.
- Heads: `mean_linear`, `mean_layer_linear`,
  `mean_rich_stats_residual`, and `multi_query_rich_stats`.
- Seeds: every integer from 33 through 42, with no missing or extra run.
- Head optimization: global seeded window shuffling, AdamW with learning rate
  `1e-4`, weight decay `0.05`, OneCycleLR with `max_lr=1e-4`, `pct_start=0.1`,
  cosine annealing, gradient clipping at `1.0`, batch size 64, and MSE loss.
- Training budget: at most 40 epochs with early-stopping patience 7 for every
  head, matching the canonical NeuralBench budget.
- Checkpoint selection: maximum HBN validation Pearson, ties resolved by the
  earliest epoch.
- External adaptation: none. MIPDB must never be used for fitting, checkpoint
  selection, recalibration, hyperparameter tuning, or post-hoc QC thresholds.
- Primary unit: one age and one arithmetic-mean prediction per subject.

The primary MIPDB cohort is the non-pilot cohort inside the age support of the
HBN training set. Subjects outside that support are secondary extrapolation
data. If fewer than 50 primary subjects survive the predeclared QC contract, the
run is labelled underpowered and cannot establish confirmatory superiority.

## Runtime paths

Set explicit absolute paths in the shell that will execute the study. Do not
copy data or outputs into `results/canonical/`.

```bash
export RESEARCH_PROTOCOL="$PWD/configs/research/external_frozen_probe.json"
export TRAINING_PROTOCOL="$PWD/configs/research/neuralbench_frozen_probe_training.json"
export MIPDB_ROOT="/absolute/path/to/mipdb"
export MIPDB_DRAFT_MANIFEST="/absolute/path/to/study/mipdb_draft_manifest.json"
export MIPDB_PILOT_QC="/absolute/path/to/study/mipdb_pilot_qc.json"
export MIPDB_COHORT_QC="/absolute/path/to/study/mipdb_cohort_qc.json"
export MIPDB_MANIFEST="/absolute/path/to/study/mipdb_final_manifest.json"
export HBN_ROOT="/absolute/path/to/hbn"
export HBN_SUBJECT_MANIFEST="$PWD/results/canonical/data/age_medium_1000_nested.csv"
export HBN_PREPROCESSING_CACHE="/absolute/path/to/study/hbn_preprocessed"
export HBN_CACHE="/absolute/path/to/study/hbn_representations"
export HBN_TRAINING_MANIFEST="/absolute/path/to/study/hbn_training_manifest.json"
export REVE_CHANNEL_MAPPING="/absolute/path/to/neuralbench/models/channel_mappings/reve.json"
export HEAD_RUNS="/absolute/path/to/study/head_runs"
export CHECKPOINT_INVENTORY="$HEAD_RUNS/checkpoint_inventory.json"
export STUDY_LOCK="/absolute/path/to/study/study_lock.json"
export ENVIRONMENT_LOCK="/absolute/path/to/study/environment.lock"
export MIPDB_CACHE="/absolute/path/to/study/mipdb_representations"
export EXTERNAL_OUTPUT="/absolute/path/to/study/external_predictions"
export ANALYSIS_OUTPUT="/absolute/path/to/study/confirmatory_analysis"
```

Every output location is create-only or exact-resume. A changed protocol,
manifest, source tree, environment, checkpoint, subject inventory, or existing
artifact is a hard error.

## Execution sequence

### 1. Content-addressed MIPDB inventory

The command derives the support interval from unique HBN training subjects in
the canonical manifest; validation, HBN R5, and MIPDB outcomes cannot override
its endpoints:

```bash
python scripts/build_mipdb_manifest.py \
  --protocol "$RESEARCH_PROTOCOL" \
  --bids-root "$MIPDB_ROOT" \
  --hbn-manifest "$HBN_SUBJECT_MANIFEST" \
  --output "$MIPDB_DRAFT_MANIFEST"
```

This stage does not load model code or compute outcomes. It hashes the dataset
metadata, every selected subject acquisition file, BrainVision companions, and
BIDS sidecars. It deterministically allocates ten pilot subjects and emits draft
evaluation cohorts.

### 2. Pilot QC

```bash
python scripts/run_mipdb_pilot.py \
  --protocol "$RESEARCH_PROTOCOL" \
  --bids-root "$MIPDB_ROOT" \
  --mipdb-manifest "$MIPDB_DRAFT_MANIFEST" \
  --mapping "$REVE_CHANNEL_MAPPING" \
  --output "$MIPDB_PILOT_QC" \
  --device cuda
```

The pilot may check only loading, block selection, channel labels, resampling,
filtering, finite windows, geometry, and model-output shape. It must not compute
or retain an age-prediction metric. The report is create-only and contains no
age, target, prediction, or metric field.

### 3. Finalize the MIPDB cohort with target-free QC

```bash
python scripts/finalize_mipdb_cohort.py \
  --protocol "$RESEARCH_PROTOCOL" \
  --bids-root "$MIPDB_ROOT" \
  --draft-manifest "$MIPDB_DRAFT_MANIFEST" \
  --qc-output "$MIPDB_COHORT_QC" \
  --output "$MIPDB_MANIFEST"
```

This command applies the already frozen event, channel, duration, finite-signal,
resampling, and window rules to every non-pilot candidate. It never loads REVE
and never computes predictions. Failed candidates receive the fixed exclusion
reason `predeclared_signal_qc_failed`; list hashes and the underpowered flag are
recomputed in the create-only finalized manifest.

### 4. HBN representation extraction

```bash
python scripts/cache_hbn_representations.py \
  --protocol "$RESEARCH_PROTOCOL" \
  --subject-manifest "$HBN_SUBJECT_MANIFEST" \
  --data-root "$HBN_ROOT" \
  --preprocessing-cache-root "$HBN_PREPROCESSING_CACHE" \
  --representation-cache-root "$HBN_CACHE" \
  --training-manifest "$HBN_TRAINING_MANIFEST" \
  --mapping "$REVE_CHANNEL_MAPPING" \
  --device cuda
```

The command validates release-level splits, excludes R5 before resolving or
opening signal files, requires disjoint train/validation subjects, and caches
only layers `-2` and `-1`. Existing cache entries are reused only after exact
identity, payload, layer, and extraction-evidence validation.

The HBN dataset identity includes hashes of the selected raw recordings, and
both preprocessing and representation caches therefore fail closed after any
same-path file replacement.

### 5. Head-only training

```bash
python scripts/run_frozen_probe.py \
  --protocol "$RESEARCH_PROTOCOL" \
  --training-protocol "$TRAINING_PROTOCOL" \
  --training-manifest "$HBN_TRAINING_MANIFEST" \
  --cache-root "$HBN_CACHE" \
  --output-root "$HEAD_RUNS" \
  --device cuda
```

This command has no head or seed override: it requires the exact four-head by
ten-seed matrix and writes one audited checkpoint inventory.

### 6. Seal the study

Sealing is artifact-derived and requires a clean Git worktree. The command
independently hashes and cross-checks the protocol, environment, HBN acquisition
and training manifests, all 40 run manifests and checkpoints, finalized MIPDB
manifest, pilot QC, cohort QC, encoder state, cohort lists, source tree, and
absolute external output root:

```bash
python scripts/seal_external_study.py \
  --protocol "$RESEARCH_PROTOCOL" \
  --training-protocol "$TRAINING_PROTOCOL" \
  --environment "$ENVIRONMENT_LOCK" \
  --hbn-subject-manifest "$HBN_SUBJECT_MANIFEST" \
  --hbn-training-manifest "$HBN_TRAINING_MANIFEST" \
  --hbn-data-root "$HBN_ROOT" \
  --checkpoint-root "$HEAD_RUNS" \
  --checkpoint-inventory "$CHECKPOINT_INVENTORY" \
  --mipdb-manifest "$MIPDB_MANIFEST" \
  --mipdb-bids-root "$MIPDB_ROOT" \
  --mipdb-pilot-qc "$MIPDB_PILOT_QC" \
  --mipdb-cohort-qc "$MIPDB_COHORT_QC" \
  --output-root "$EXTERNAL_OUTPUT" \
  --lock "$STUDY_LOCK"
```

Sealing creates an immutable lock and state sidecar. Any input change requires a
new study identity; a started study cannot be resealed.

### 7. Sealed external extraction and inference

Primary MIPDB representations must not be precomputed before the study start
marker. The command below validates the sealed inputs, writes
`evaluation_started.json`, and only then lazily loads, preprocesses, and passes
each missing primary subject through REVE. A complete hash-valid cache entry is
reused on an exact resume.

```bash
python scripts/run_external_holdout.py \
  --protocol "$RESEARCH_PROTOCOL" \
  --training-protocol "$TRAINING_PROTOCOL" \
  --lock "$STUDY_LOCK" \
  --checkpoint-root "$HEAD_RUNS" \
  --checkpoint-inventory "$CHECKPOINT_INVENTORY" \
  --mipdb-manifest "$MIPDB_MANIFEST" \
  --environment "$ENVIRONMENT_LOCK" \
  --cache-root "$MIPDB_CACHE" \
  --bids-root "$MIPDB_ROOT" \
  --mapping "$REVE_CHANNEL_MAPPING" \
  --output-root "$EXTERNAL_OUTPUT" \
  --device cuda
```

The runner evaluates only the locked primary subject inventory. It writes no
aggregate metric and allows only exact resume against immutable predictions.

Failures after the start marker append structured evidence under
`study_failures/` while preserving exact resume against the same lock.

### 8. Confirmatory analysis — only after completion

```bash
python scripts/analyze_confirmatory.py \
  --protocol "$RESEARCH_PROTOCOL" \
  --lock "$STUDY_LOCK" \
  --checkpoint-root "$HEAD_RUNS" \
  --checkpoint-inventory "$CHECKPOINT_INVENTORY" \
  --mipdb-manifest "$MIPDB_MANIFEST" \
  --prediction-root "$EXTERNAL_OUTPUT" \
  --output-root "$ANALYSIS_OUTPUT"
```

Analysis refuses an incomplete prediction inventory or mismatched provenance.
It reports Pearson, MAE, RMSE, R², calibration, resource use, exclusions,
cohort sizes, per-seed deltas, wins/ties/losses, worst-seed delta, and seed SD.
An underpowered cohort still receives descriptive estimates, but the adequate
cohort condition fails and `established_heads` must remain empty.

## Confirmatory decision rule

For each complex head, the estimand is its mean paired external Pearson delta
against `mean_linear` over the ten seeds. The analysis uses a hierarchical
paired bootstrap with 10,000 iterations and seed `20260903`, an exact one-sided
paired seed-randomization test, and Holm correction over the three comparisons.

A head establishes a stable external gain only when all four conditions hold:

1. Holm-adjusted one-sided p-value is below 0.05.
2. The 95% bootstrap interval lower bound is above zero.
3. The head wins at least 8 of 10 seeds.
4. Its worst seed delta is at least -0.01 Pearson.

If no candidate passes, the supported wording is: “No tested complex head
established a stable external gain under the predeclared protocol.” This does
not establish equivalence, a true zero effect, or the absence of benefit for an
untested method.

## Result record

Values remain placeholders until the sealed artifacts exist.

| Quantity | Value |
| --- | --- |
| Eligible MIPDB subjects before pilot allocation | `TBD_AFTER_EXECUTION` |
| Engineering pilot subjects | `TBD_AFTER_EXECUTION` |
| Primary MIPDB subjects before QC | `TBD_AFTER_EXECUTION` |
| Primary MIPDB subjects after QC | `TBD_AFTER_EXECUTION` |
| Extrapolation subjects | `TBD_AFTER_EXECUTION` |
| Underpowered flag | `TBD_AFTER_EXECUTION` |
| Completed HBN head runs | `TBD_AFTER_EXECUTION` |
| External prediction count | `TBD_AFTER_EXECUTION` |
| Per-head paired effect and interval | `TBD_AFTER_EXECUTION` |
| Adjusted p-values and decision | `TBD_AFTER_EXECUTION` |

## Limitations that must accompany interpretation

- Cross-dataset shift: HBN and MIPDB may differ in acquisition hardware,
  montage, demographics, recruitment, resting-state instructions, recording
  duration, and data quality. External performance mixes representation quality
  with those shifts.
- Encoder pretraining uncertainty: published REVE sources do not list MIPDB,
  but absence from that list is evidence rather than a cryptographic guarantee
  that no MIPDB-derived sample influenced pretraining.
- Channel-layout sensitivity: preserving each dataset's mapped scalp layout
  avoids invented interpolation but may change the encoder input distribution.
- Cohort support: the primary inference applies only inside HBN training-age
  support; older subjects are extrapolation evidence and younger out-of-support
  subjects are excluded by the predeclared rule.
- Statistical scope: ten optimization seeds characterize the selected training
  procedure, not all possible initializations, checkpoints, or head families.
- Underpower: fewer than 50 post-QC primary subjects permits descriptive output
  only and no confirmatory superiority claim.
- Retrospective context: HBN R5 comparisons can reveal implementation behavior
  but cannot restore an untouched holdout after repeated use.
