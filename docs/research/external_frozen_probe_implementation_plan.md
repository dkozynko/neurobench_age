# External Frozen-Probe Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, reproducible frozen-REVE probing pipeline whose
primary claim is evaluated once on an untouched external MIPDB holdout.

**Architecture:** Keep the existing official NeuralBench full-fine-tuning path
as secondary reproduction evidence. Add an independent research package for
protocol validation, dataset inventory, frozen representation extraction,
head-only training, external evaluation, and confirmatory statistics. Every
stage consumes immutable manifests and emits auditable JSON/JSONL artifacts.

**Tech stack:** Python 3.12+, PyTorch, NumPy, SciPy, MNE/MNE-BIDS, pytest.

**Repository rule:** Do not create Git commits. Leave all changes uncommitted
for manual review and commit.

**Design specification:**
`docs/research/external_frozen_probe_study_design.md`

---

## Intended file map

New production files:

- `src/neurobench_age/research/__init__.py`
- `src/neurobench_age/research/protocol.py`
- `src/neurobench_age/research/study_lock.py`
- `src/neurobench_age/analysis/comparison.py`
- `src/neurobench_age/analysis/confirmatory.py`
- `src/neurobench_age/data/mipdb.py`
- `src/neurobench_age/pipelines/frozen_probe.py`
- `src/neurobench_age/pipelines/external_holdout.py`
- `configs/research/external_frozen_probe.json`
- `scripts/build_mipdb_manifest.py`
- `scripts/seal_external_study.py`
- `scripts/run_frozen_probe.py`
- `scripts/run_external_holdout.py`
- `scripts/analyze_confirmatory.py`

Existing files expected to change:

- `pyproject.toml`
- `.gitignore`
- `src/neurobench_age/official.py`
- `src/neurobench_age/analysis/gate.py`
- `src/neurobench_age/analysis/paper_evidence.py`
- `scripts/run_article_experiment.sh`
- `README.md`
- `docs/research/article_ready_protocol.md`
- `docs/research/article_evidence_registry.md`

New or extended tests:

- `tests/test_package_contract.py`
- `tests/test_comparison_contract.py`
- `tests/test_research_protocol.py`
- `tests/test_study_lock.py`
- `tests/test_mipdb_inventory.py`
- `tests/test_mipdb_preprocessing.py`
- `tests/test_frozen_probe.py`
- `tests/test_external_holdout.py`
- `tests/test_confirmatory_analysis.py`
- existing gate, evidence, launcher, CLI, and documentation tests.

---

## Task 1: Repair package and output safety

**Files:** `pyproject.toml`, `.gitignore`,
`scripts/run_article_experiment.sh`, `tests/test_package_contract.py`, existing
launcher tests.

- [x] Write failing tests asserting that the supported Python range is
  compatible with `neuralbench==0.2.3`, pytest can import from `src` without a
  manually supplied `PYTHONPATH`, and external-data dependencies are declared
  separately.
- [x] Write a failing launcher test that requires an explicit output root and
  rejects any run directory inside `results/canonical`.
- [x] Add ignore rules for external datasets, representation caches,
  checkpoints, model formats, and `results/canonical/runs/` while preserving
  small canonical evidence files.
- [x] Set the package requirement to Python 3.12 or newer and configure pytest
  with `pythonpath = ["src"]`.
- [x] Add a named optional dependency group for MNE/MNE-BIDS rather than making
  external-dataset tooling an implicit environment dependency.
- [x] Make the article launcher require an explicit output root outside the
  tracked canonical evidence tree.
- [x] Run the focused tests, then resolve dependencies in a clean environment
  without downloading EEG data.

Verification:

```bash
python3 -m pytest -q tests/test_package_contract.py tests/test_launchers.py
uv lock --check
```

## Task 2: Enforce exact matched-run comparisons

**Files:** `src/neurobench_age/analysis/comparison.py`,
`src/neurobench_age/analysis/gate.py`,
`src/neurobench_age/analysis/paper_evidence.py`,
`tests/test_comparison_contract.py`, existing gate/evidence tests.

- [x] Write failing tests where a comparison is rejected for missing seeds,
  extra seeds, different subject IDs, different targets, different split hash,
  different dataset fingerprint, different source revision, different
  checkpoint, different preprocessing, or different determinism settings.
- [x] Define one shared `ComparisonContract` containing exact seed inventory,
  comparison-factor keys, evaluation mode, dataset/split/source/config hashes,
  checkpoint identity, determinism policy, ordered subject IDs, and true ages.
- [x] Require exact seed-set equality. Do not intersect seed sets.
- [x] Require identical ordered subjects and targets before computing paired
  metrics.
- [x] Route both the experiment gate and paper-evidence generation through this
  shared validator.
- [x] Emit a structured rejection report naming every mismatched field.

Verification:

```bash
python3 -m pytest -q tests/test_comparison_contract.py tests/test_gate.py tests/test_paper_evidence.py
```

## Task 3: Make analysis outputs fresh and observable

**Files:** `src/neurobench_age/analysis/paper_evidence.py`, analysis CLI tests.

- [x] Write failing tests for a non-empty output directory, a plotting failure,
  and a manifest containing stale files.
- [x] Reject every non-empty output directory; completed analysis directories
  are immutable and a rerun must use a fresh destination.
- [x] Add explicit plot policy values: `required`, `optional`, and `off`.
- [x] Under `required`, propagate plotting failures. Under `optional`, record a
  structured warning rather than silently swallowing an exception.
- [x] Build the output manifest only from files created in the current analysis
  transaction.
- [x] Write results atomically and record tool/library versions.

Verification:

```bash
python3 -m pytest -q tests/test_paper_evidence.py tests/test_analysis_cli.py
```

## Task 4: Turn the research protocol into executable configuration

**Files:** `src/neurobench_age/research/protocol.py`,
`configs/research/external_frozen_probe.json`, `src/neurobench_age/official.py`,
`tests/test_research_protocol.py`, official CLI tests.

- [x] Write failing tests for missing fields, unknown fields, wrong seeds,
  unapproved heads, an unfrozen encoder, changed layer selection, and changed
  inferential thresholds.
- [x] Implement frozen dataclasses for dataset, preprocessing, encoder, head,
  training, cohort, and statistical contracts.
- [x] Implement strict JSON parsing that rejects unknown keys and validates all
  invariants from the approved design.
- [x] Add the executable protocol with seeds 33–42, four approved heads,
  layer `-2`, bootstrap seed `20260903`, 10,000 iterations, and the full
  decision rule.
- [ ] Make every research CLI load the same protocol and record its SHA-256.
- [x] Replace the ambiguous ignored `--config` with an enforced
  `--phase-config` for article launches, and record its path, name, and SHA-256
  in run evidence. Generic diagnostic runs do not pretend to consume it.

Verification:

```bash
python3 -m pytest -q tests/test_research_protocol.py tests/test_official_cli.py
```

## Task 5: Implement immutable study lifecycle and sealing

**Files:** `src/neurobench_age/research/study_lock.py`,
`scripts/seal_external_study.py`, `tests/test_study_lock.py`.

- [x] Write failing transition tests for `draft -> sealed -> started ->
  completed`, plus terminal `failed`.
- [x] Write tampering tests for protocol, dataset manifest, source revision,
  encoder checkpoint, HBN split, head inventory, seed inventory, preprocessing,
  statistics, and output location.
- [x] Implement canonical JSON serialization and SHA-256 helpers.
- [x] Seal all required hashes into one immutable study lock.
- [x] Write state changes and sidecars atomically.
- [ ] Require a `started` marker before primary EEG samples are read.
- [ ] Permit only exact resume against the same lock and immutable prediction
  inventory.
- [x] Ensure failure writes a diagnostic artifact without rewriting the lock.

Verification:

```bash
python3 -m pytest -q tests/test_study_lock.py
```

## Task 6: Build metadata-only MIPDB inventory and engineering pilot

**Files:** `src/neurobench_age/data/mipdb.py`,
`scripts/build_mipdb_manifest.py`, `tests/test_mipdb_inventory.py`.

- [x] Create synthetic BIDS metadata fixtures covering valid participants,
  missing ages, duplicate IDs, malformed ages, and missing recordings.
- [x] Write failing tests for deterministic normalized subject ordering and
  manifest hashing.
- [x] Implement metadata-only inventory creation; this stage must not import the
  model, load signal arrays, or compute prediction metrics.
- [x] Select exactly ten engineering subjects using the approved SHA-256 rule:
  `dataset_manifest_sha256 + NUL + subject_id + NUL +
  "mipdb-engineering-pilot-v1"`.
- [x] Permanently exclude pilot subjects from the confirmatory cohort.
- [x] Split the remaining subjects into the predeclared in-support primary
  cohort and older extrapolation cohort.
- [x] Mark the study underpowered when fewer than 50 primary subjects remain.
- [x] Emit a manifest containing only metadata, exclusions, cohort membership,
  and cryptographic hashes—never model performance.

Verification:

```bash
python3 -m pytest -q tests/test_mipdb_inventory.py
```

## Task 7: Implement external preprocessing and QC contract

**Files:** `src/neurobench_age/data/mipdb.py`,
`tests/test_mipdb_preprocessing.py`.

- [x] Write synthetic-signal tests for 200 Hz resampling, 0.5–99.5 Hz
  filtering, StandardScaler semantics, clamp 15, two-second non-overlapping
  windows, 120-second cap, and block-boundary preservation.
- [x] Write tests for missing channels, duplicate channel names, non-finite
  samples, too-short recordings, and deterministic exclusions.
- [x] Implement a channel-layout-preserving adapter; do not interpolate MIPDB
  solely to imitate HBN.
- [x] Keep MNE and MNE-BIDS imports lazy so metadata/unit tests remain light.
- [x] Separate metadata validation, signal loading, preprocessing, windowing,
  and QC into independently testable functions.
- [x] Emit per-subject QC records before representation extraction.

Verification:

```bash
python3 -m pytest -q tests/test_mipdb_preprocessing.py
```

## Task 8: Guarantee genuinely frozen representation extraction

**Files:** `src/neurobench_age/pipelines/frozen_probe.py`,
`tests/test_frozen_probe.py`.

- [x] Build a tiny synthetic encoder with dropout and multiple hidden layers.
- [x] Write failing tests that detect train mode, encoder gradients, encoder
  parameters in the optimizer, state mutation, repeated stochastic extraction,
  and wrong cached layers.
- [x] Load `brain-bzh/reve-base`, set all encoder parameters to
  `requires_grad=False`, and hold the encoder in evaluation mode.
- [x] Extract under `torch.inference_mode()` and hash encoder state before and
  after extraction.
- [x] Cache final and penultimate representations once per subject in a
  deterministic external cache keyed by protocol, checkpoint, data manifest,
  preprocessing, subject, and source hashes.
- [ ] Make every head consume declared cached fields only; it may not invoke or
  mutate the encoder.
- [x] Reject incomplete, stale, or hash-mismatched cache entries.

Verification:

```bash
python3 -m pytest -q tests/test_frozen_probe.py
```

## Task 9: Train the four predeclared heads on HBN train/validation only

**Files:** `src/neurobench_age/pipelines/frozen_probe.py`,
`scripts/run_frozen_probe.py`, training tests.

- [ ] Write tests that reject any head outside `mean_linear`,
  `mean_layer_linear`, `mean_rich_stats_residual`, and
  `multi_query_rich_stats`.
- [ ] Write leakage tests proving the training path has no test-split loader or
  test metric available during fitting and checkpoint selection.
- [ ] Implement the four heads against the same cached representation contract.
- [ ] Train each head with seeds 33–42 and validation-only checkpoint selection.
- [ ] Persist head-only parameter counts, runtime, peak memory, selected epoch,
  optimizer settings, validation history, checkpoint hash, and seed.
- [ ] Require an exact 40-run inventory before the study can be sealed.
- [ ] Add exact-resume behavior that reuses only hash-valid completed runs.

Verification:

```bash
python3 -m pytest -q tests/test_frozen_probe.py tests/test_frozen_probe_cli.py
```

## Task 10: Implement one-time external holdout inference

**Files:** `src/neurobench_age/pipelines/external_holdout.py`,
`scripts/run_external_holdout.py`, `tests/test_external_holdout.py`.

- [ ] Write a static/API test proving this module has no optimizer, scheduler,
  backward pass, calibration fit, or model-selection entry point.
- [ ] Write failing tests for an unsealed study, wrong checkpoint inventory,
  wrong subject inventory, wrong seed inventory, pilot contamination, and
  partially overwritten predictions.
- [ ] Preflight the lock, code revision, environment, cache identity, 40 head
  checkpoints, and ordered external subject list.
- [ ] Transition to `started` before reading primary subject EEG.
- [ ] Write one immutable subject prediction record per head and seed, including
  target, prediction, QC status, and all relevant hashes.
- [ ] Resume only missing records with exact identity; never overwrite an
  existing record.
- [ ] Produce no aggregate metric until the full expected prediction inventory
  is complete.
- [ ] Transition to `completed` only after inventory and hash validation.

Verification:

```bash
python3 -m pytest -q tests/test_external_holdout.py
```

## Task 11: Implement predeclared confirmatory statistics

**Files:** `src/neurobench_age/analysis/confirmatory.py`,
`scripts/analyze_confirmatory.py`, `tests/test_confirmatory_analysis.py`.

- [ ] Add reference tests for Pearson, MAE, RMSE, R², calibration slope, and
  calibration intercept.
- [ ] Add exact-pairing tests over ten seeds and identical subject order.
- [ ] Implement the primary estimand: mean across seeds of candidate-minus-
  `mean_linear` subject-level external Pearson.
- [ ] Implement deterministic hierarchical paired bootstrap with 10,000
  iterations and RNG seed `20260903`, resampling seeds and subjects while
  preserving pairing.
- [ ] Implement exact one-sided paired seed randomization by enumerating all
  `2^10` sign flips.
- [ ] Implement Holm step-down correction over the three candidate comparisons.
- [ ] Implement the joint stable-improvement rule: adjusted p-value below 0.05,
  bootstrap lower bound above zero, at least 8/10 wins, and worst seed delta at
  least -0.01.
- [ ] Report wins/ties/losses, worst delta, seed SD, calibration, resource use,
  exclusions, cohort sizes, and power warning.
- [ ] If no candidate passes, state only that no tested complex head established
  a stable external gain; do not claim equivalence.
- [ ] Fail if predictions are incomplete or if any provenance field differs.

Verification:

```bash
python3 -m pytest -q tests/test_confirmatory_analysis.py
```

## Task 12: Correct the evidence taxonomy and documentation

**Files:** `README.md`, `docs/research/article_ready_protocol.md`,
`docs/research/article_evidence_registry.md`, documentation tests.

- [ ] Label existing HBN R5 results as retrospective/secondary because the test
  set has already been used in repeated finalist decisions.
- [ ] Describe frozen representation probing as the primary new study and
  official NeuralBench full fine-tuning as secondary reproduction evidence.
- [ ] Replace generic “probing” language for trainable-encoder runs with
  “end-to-end age prediction” or “full fine-tuning.”
- [ ] Document the exact commands for inventory, pilot QC, HBN extraction,
  head training, sealing, external inference, and analysis.
- [ ] Add placeholders for real cohort counts and results rather than inventing
  values before the study runs.
- [ ] Document the negative-result interpretation and all limitations, including
  cross-dataset shift and possible encoder pretraining uncertainty.

Verification:

```bash
python3 -m pytest -q tests/test_documentation.py
```

## Task 13: Full synthetic verification and readiness audit

**Files:** all files above; optional readiness report under `docs/research/`.

- [ ] Run the complete test suite without manual `PYTHONPATH`.
- [ ] Compile all Python modules.
- [ ] Syntax-check all shell launchers.
- [ ] Build/install in a clean Python 3.12 environment with the declared
  research extras.
- [ ] Run a tiny end-to-end synthetic sealed study covering inventory, pilot,
  representation caching, 40 miniature head runs, one-time prediction, and
  confirmatory analysis.
- [ ] Scan the repository for tracked datasets, checkpoints, archives, caches,
  secrets, and unexpectedly large blobs.
- [ ] Run `git diff --check` and inspect every uncommitted change.
- [ ] Write a readiness report that distinguishes code validation from real-data
  completion and explicitly says primary MIPDB inference has not started.

Verification:

```bash
python3 -m pytest -q
python3 -m compileall -q src scripts tests
bash -n scripts/*.sh
git diff --check
git status --short
```

## Task 14: Operational sequence for the real study

This task begins only after Tasks 1–13 pass and the artifacts are manually
reviewed. It is an execution checklist, not part of ordinary code testing.

- [ ] Download or mount MIPDB outside Git and record the exact release/source.
- [ ] Build the metadata-only dataset manifest.
- [ ] Materialize the deterministic ten-subject engineering pilot list.
- [ ] Run pilot loader/QC/shape checks only; do not compute age metrics.
- [ ] Freeze the QC policy and regenerate the confirmatory cohort manifest.
- [ ] Extract HBN train/validation representations once with the frozen encoder.
- [ ] Train all four heads over seeds 33–42 and verify the exact 40-run inventory.
- [ ] Audit every manifest, checkpoint, source, environment, and output hash.
- [ ] Seal the external study lock.
- [ ] Run external representation extraction and prediction exactly once, with
  exact-resume semantics only if interrupted.
- [ ] Verify completion before generating any aggregate metric.
- [ ] Run the predeclared confirmatory analysis without changing thresholds or
  exclusions.
- [ ] Freeze the evidence bundle and use it to write the Results, Discussion,
  Limitations, and Conclusion sections.

Expected paper narrative after execution:

1. Reproduce the established NeuralBench age-prediction setup as secondary
   context.
2. Test whether more expressive heads extract stable age information from a
   fixed REVE representation better than a linear mean-pooled probe.
3. Separate within-dataset validation from one-time external confirmation.
4. Report both positive and negative outcomes using the same predeclared rule.
5. Bound claims to the tested representations, heads, cohorts, and protocol.
