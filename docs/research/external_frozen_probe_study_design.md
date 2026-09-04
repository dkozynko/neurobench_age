# External frozen-probe study design

Status: approved design; implementation and study sealing are pending.

Date: 2026-09-04

## Decision

The primary paper experiment will be a frozen-encoder comparison of four
predeclared REVE representation heads. Models are developed and selected using
HBN train/validation data only, then evaluated without adaptation on a new
external MIPDB cohort. Existing end-to-end NeuralBench results remain secondary,
retrospective evidence and are not described as untouched confirmation.

No primary MIPDB participant may be passed through a study model until the
protocol lock described below has been sealed.

## Research question

When the pretrained REVE encoder, HBN development data, preprocessing,
checkpoint rule, and random seeds are fixed, do increasingly expressive
representation heads yield an improvement over mean-linear probing that is
stable across optimization seeds and transfers to an external developmental
EEG cohort?

The intended contribution is evidence about stability and limits. Failure to
establish a stable gain is a valid result; it is not evidence that the true
effect is exactly zero.

## Evidence boundary

The repository already contains repeated evaluations on the NeuralBench/HBN
test split. Those results may support retrospective descriptions, implementation
checks, and motivation, but they are not an untouched confirmatory test.

The new confirmatory boundary is MIPDB. Metadata-only inspection, download,
format validation, and model-free signal quality checks are allowed before
sealing. Model predictions and age-prediction metrics are prohibited for the
primary holdout before sealing.

The REVE NeurIPS 2025 appendix lists HBN releases and TDBRAIN among its
pretraining sources. MIPDB is not named in the published exhaustive list. The
paper must describe MIPDB's encoder-level independence as an inference from that
list, not as a guarantee from the REVE authors.

## Data sources

### Development data: HBN

- Use the canonical HBN manifest already retained by the repository.
- Train heads on the declared HBN training subjects.
- Select checkpoints on the declared HBN validation subjects.
- Do not use HBN R5 for a new primary decision, hyperparameter choice, or
  promotion decision.
- Keep all participant splits subject-disjoint and validate that invariant
  before representation extraction.

### External data: MIPDB

- Pin the NEMAR MIPDB dataset identifier `nm000153` and an exact immutable
  dataset version/manifest digest.
- Use task-free/resting EEG only.
- Require an individual subject identifier, finite chronological age, channel
  metadata, and sufficient valid recording duration.
- Define the primary age-support interval from HBN training ages before reading
  MIPDB predictions. MIPDB subjects outside that interval belong only to a
  secondary extrapolation analysis.
- Never train, fine-tune, recalibrate, or select a checkpoint using MIPDB.

Primary sources:

- MIPDB archive: https://www.nemar.org/dataset/nm000153
- MIPDB data descriptor: https://www.nature.com/articles/sdata201740
- REVE paper: https://papers.neurips.cc/paper_files/paper/2025/file/20a917f77773ac0fa8bea2bdd6606b66-Paper-Conference.pdf

## Engineering pilot and confirmatory cohort

After metadata eligibility checks, sort eligible subject IDs and assign an
engineering pilot deterministically. Select the ten IDs with the lowest SHA-256
value of:

```text
dataset_manifest_sha256 + NUL + subject_id + NUL + "mipdb-engineering-pilot-v1"
```

The pilot is permanently excluded from every confirmatory table. Pilot use is
limited to loader behavior, channel mapping, resampling, window geometry,
finite tensors, and model-output shape. Age-prediction metrics must not be
computed or retained for pilot subjects.

All remaining eligible subjects within the HBN training age-support interval
form the primary external cohort. The protocol lock records both lists and
their hashes. Any overlap, duplicate ID, or post-seal membership change is a
hard error.

If fewer than 50 primary external subjects remain after predeclared QC, the
study is labelled underpowered and no confirmatory superiority claim is made.
The data are still reported descriptively with that limitation.

## Experimental tracks

### Primary: frozen representation probing

- Use one pinned `brain-bzh/reve-base` checkpoint and record its SHA-256.
- Freeze every encoder parameter and keep the encoder in evaluation mode.
- Run one deterministic extraction pass per data window and cache every
  predeclared REVE layer needed by the four heads. Each head reads only its
  declared layer fields, while subject membership, windows, extraction code,
  and tensor ordering remain identical.
- Assert that encoder parameters have no gradients and that the encoder state
  hash is unchanged before and after every head-training run.
- Train only head parameters.

Caching representations is part of the scientific control: it prevents
candidate-specific encoder stochasticity and ensures that a head comparison is
about the aggregation/probe rather than a different encoder trajectory.

### Secondary: official end-to-end NeuralBench

Existing official NeuralBench runs remain a reproduction and sensitivity
analysis. Because their encoder parameters are trainable, they are described as
end-to-end age prediction, not pure representation probing. They do not select
or alter the primary frozen-probe candidates after the external protocol is
sealed.

## Predeclared heads

The primary family contains exactly four heads:

1. `mean_linear`: mean-token linear reference.
2. `mean_layer_linear` with layer index `-2`: intermediate-layer linear probe.
3. `mean_rich_stats_residual`: nonlinear statistical aggregation.
4. `multi_query_rich_stats`: attention-based high-complexity aggregation.

This set represents a small complexity ladder rather than a search over all
implemented variants. No other head may be added to this study after sealing.
Unlisted heads remain historical or future-work candidates.

For every head, retain trainable parameter count, total operation description,
representation layer(s), aggregation rule, and inference time. Complexity is a
reported property, not a post-hoc selection criterion.

## Seeds and pairing

Use exactly ten head-training seeds: integers 33 through 42 inclusive.

Within each seed, all heads use the same HBN split, representation cache,
minibatch order contract, epoch budget, checkpoint metric, and stopping rule.
The external cohort and its subject order are identical across all heads and
seeds. Any missing seed invalidates the complete confirmatory comparison; the
analysis must not silently use the intersection of available seeds.

## Preprocessing contract

The common REVE input contract is:

- scalp channels identified from BIDS channel metadata;
- channel positions supplied to REVE by canonical electrode label;
- no spatial interpolation solely to imitate the HBN channel count;
- 200 Hz output frequency;
- 0.5--99.5 Hz band-pass;
- no notch filter unless required and frozen before sealing for both datasets;
- `StandardScaler` normalization with clamp 15;
- non-overlapping two-second windows;
- no window may cross a resting-block boundary;
- deterministic acquisition-order selection of at most 120 valid seconds per
  subject for the primary metric;
- arithmetic mean of window predictions for one subject-level prediction.

The MIPDB adapter must record original frequency, included blocks, channel
labels, mapped-channel count, rejected channels, usable duration, window count,
and every QC reason. Candidate-specific preprocessing is forbidden.

Exact event names and the minimum mapped-channel threshold may be finalized
from metadata and model-free pilot inspection, but they must be present in the
sealed lock. After sealing they cannot change.

## Training and checkpoint selection

- The encoder is frozen and evaluated deterministically.
- The optimizer contains head parameters only.
- MSE is the training loss unless the sealed protocol specifies one common
  alternative for every head.
- Epoch budget, optimizer, learning rate, weight decay, scheduler, batch size,
  and patience are shared unless a difference is explicitly part of a
  predeclared head contract.
- Select one checkpoint per head and seed by maximum HBN validation Pearson.
- Persist the complete validation trajectory and deterministic tie behavior.
- Do not inspect HBN test or MIPDB metrics during selection.

The executable config is authoritative. Human-readable phase config files must
be schema-validated and must either populate or verify every runtime option;
passing an ignored `--config` is prohibited.

## Primary estimand and reporting

For each non-baseline head, compute the candidate-minus-`mean_linear`
difference in subject-level MIPDB Pearson for every matched seed. The primary
estimand is the arithmetic mean of those ten paired seed deltas.

Report, without selective omission:

- per-seed Pearson for candidate and baseline;
- per-seed paired delta;
- mean paired delta;
- sample SD across seed deltas;
- seed wins, ties, and losses;
- worst-seed delta;
- MAE, RMSE, and R-squared;
- calibration intercept and slope;
- head parameter count, runtime, and memory;
- all exclusions and effective subject count.

### Hierarchical paired bootstrap

Use 10,000 deterministic bootstrap iterations with RNG seed `20260903`.
For each iteration:

1. resample the ten seed IDs with replacement;
2. resample primary subject IDs with replacement, using the same sampled IDs
   for every head and baseline;
3. calculate each candidate and baseline metric on the same sampled subjects;
4. average paired seed deltas.

Return percentile 95% intervals and retain the full bootstrap specification,
not the full resample matrix. A failed or undefined resample is counted and its
handling is reported.

For each candidate, obtain an exact one-sided paired seed-randomization p-value
from the ten observed seed deltas. Enumerate all `2^10` candidate/baseline label
swaps (equivalently, sign flips of the paired seed deltas) and define the
p-value as the fraction of permuted mean deltas greater than or equal to the
observed mean delta. This treats optimization seed as the randomization unit;
the hierarchical bootstrap separately represents uncertainty from both seeds
and subjects.

Apply Holm's step-down correction to those three p-values in a fixed head order
recorded by the sealed protocol. Effect sizes and intervals are reported
regardless of adjusted significance. Raw and adjusted p-values, permutation
count, tail direction, and zero-delta handling are retained in the analysis
artifact.

A head may be described as establishing a stable external improvement only if:

- its Holm-adjusted one-sided improvement test is below 0.05;
- the hierarchical 95% interval for mean Pearson delta is above zero;
- it wins at least eight of ten seeds; and
- its worst seed delta is not below -0.01 Pearson.

If no head passes, the valid conclusion is that no tested complex head
established a stable external gain under the predeclared protocol. This does
not establish exact equivalence or absence of all possible benefit.

## Study lock and lifecycle

The confirmatory study has these states:

```text
draft -> sealed -> started -> completed
                         \-> failed
```

The sealed lock contains:

- study ID and schema version;
- source-tree SHA-256 and Git revision/dirty status;
- executable config digest;
- REVE checkpoint identity and SHA-256;
- HBN and MIPDB dataset versions and manifest digests;
- train, validation, pilot, primary, and extrapolation subject-list hashes;
- all four heads and their immutable settings;
- seeds 33--42;
- preprocessing and QC rules;
- checkpoint selection rule;
- metrics, bootstrap specification, correction method, and decision rule;
- expected output inventory;
- UTC seal timestamp.

Sealing is allowed only when all fields are complete. The external runner
refuses a draft or altered lock and writes `evaluation_started.json` before
loading primary EEG. It writes `evaluation_completed.json` only after all
expected predictions and hashes pass audit.

An infrastructure failure may be resumed only with the exact same lock,
checkpoints, code hash, and append-only evaluation ledger. Existing subject
predictions are immutable: a resume verifies their hashes and computes only
missing work. Aggregate primary metrics are not generated until the expected
prediction inventory is complete. If any partial primary metric has been
inspected before a protocol change, the confirmatory study is contaminated and
must not be relabelled as sealed.

## Components

### `data/mipdb.py`

Owns dataset inventory, BIDS metadata validation, deterministic pilot
assignment, participant manifests, resting-block discovery, channel mapping,
preprocessing metadata, and model-free QC.

### `research/protocol.py`

Defines schema-validated executable research configs and cross-field
invariants. It converts a declared config into normalized runtime values and
rejects undeclared CLI overrides.

### `research/study_lock.py`

Creates, seals, verifies, and advances the append-only confirmatory lifecycle.
It owns tamper detection and expected-artifact validation.

### `pipelines/frozen_probe.py`

Extracts frozen representations, trains heads on HBN train data, selects on HBN
validation, verifies encoder immutability, and writes checkpoint evidence.

### `pipelines/external_holdout.py`

Contains no optimizer or training API. It verifies the sealed study and fixed
checkpoints, performs one external prediction pass, and writes subject-level
evidence plus lifecycle markers.

### `analysis/confirmatory.py`

Requires exact seed and subject pairing, computes the predeclared statistics,
applies Holm correction, produces tables/figures, and writes a hashed analysis
manifest. It refuses a non-empty output directory unless it is an exact
idempotent resume of the same analysis manifest.

## Corrections to the existing pipeline

Implementation must also correct these audited issues:

1. Load and enforce the supplied article/research config instead of merely
   requiring its path.
2. Make the finalist gate verify dataset, split, protocol, source, deterministic
   policy, config, seed set, and prediction-subject identity.
3. Require exact candidate/baseline seed equality in every paired analysis.
4. Replace silent plotting exceptions with explicit failure or recorded
   unavailable status.
5. Reject stale analysis output directories.
6. Align package support with NeuralBench's Python requirement (Python 3.12 or
   newer) and provide reproducible dependency constraints/lock metadata.
7. Prevent raw runs, checkpoints, and predictions under versioned canonical
   result paths; require an explicit external output root.
8. Label all previously opened HBN test evidence as retrospective.

## Error handling

The pipeline fails closed on:

- dataset version or manifest hash mismatch;
- unknown, duplicate, overlapping, or changed subject IDs;
- pilot subjects in primary outputs;
- missing or non-finite ages;
- insufficient mapped channels or valid duration under the sealed rule;
- preprocessing drift between heads;
- trainable encoder parameters or changed encoder hash;
- checkpoint/config/source mismatch;
- missing or extra seeds;
- missing or extra primary subjects;
- unexpected training capability in the external runner;
- prior completed evaluation for the same study ID;
- non-finite predictions or incomplete evidence;
- post-seal changes to statistics or candidate membership.

Failures produce structured JSON evidence without presenting the run as
complete.

## Verification strategy

No automated test downloads MIPDB or pretrained weights. Synthetic BIDS and
representation fixtures cover:

- deterministic pilot assignment;
- exact participant partitioning and overlap rejection;
- dataset and config tamper detection;
- event/block boundary handling;
- channel mapping and QC contracts;
- frozen encoder gradients and before/after state hashes;
- optimizer ownership of head parameters only;
- exact seed and subject pairing;
- hierarchical bootstrap determinism and known synthetic effects;
- Holm correction against reference examples;
- lifecycle transitions, failure, exact resume, and second-run rejection;
- validation-only checkpoint selection;
- synthetic end-to-end external evaluation;
- required evidence and analysis-manifest hashes.

Acceptance requires the complete existing test suite plus new tests, Python
byte-compilation, shell syntax checks, clean editable installation in the
declared Python range, and dry-run generation/audit of a sealed synthetic study.

## Execution sequence

1. Correct existing reproducibility and comparison bugs.
2. Implement executable config and study-lock primitives.
3. Implement and test frozen representation extraction/training.
4. Implement and test MIPDB metadata, pilot, preprocessing, and QC adapter.
5. Run engineering pilot checks only.
6. Finalize event names and QC threshold from model-free pilot evidence.
7. Generate and independently audit the complete study lock.
8. Train all four heads for seeds 33--42 on HBN train/validation only.
9. Audit checkpoints and seal their hashes.
10. Start the one-time primary MIPDB evaluation.
11. Run the predeclared confirmatory analysis without changing parameters.
12. Publish all results, including negative or inconclusive outcomes.

## Paper structure enabled by this design

1. Motivation: single-seed head gains can be misleading.
2. Reproduction: official NeuralBench behavior and historical evidence.
3. Method: frozen REVE representations and four-head complexity ladder.
4. Internal results: HBN validation stability across ten seeds.
5. External confirmation: untouched MIPDB age-overlap cohort.
6. Complexity/stability analysis: gain, variance, runtime, and parameter cost.
7. Secondary analyses: extrapolation ages and retrospective end-to-end runs.
8. Limitations: cohort shift, modest MIPDB sample, pretraining-list inference,
   and generalization beyond the tested heads.

## Non-goals

- Claiming a new NeuralBench state of the art.
- Searching additional heads after external results are visible.
- Fine-tuning or calibrating on MIPDB.
- Treating three historical seeds as a population-level proof.
- Hiding failed, negative, or incomplete experiments.
