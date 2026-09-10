# Independent external replication: feasibility and implementation plan

Date: 2026-09-10.

**Status:** metadata/literature screening, target-free adapter validation, full
target-free signal QC, precision-gate approval, target-bearing cohort
finalization, lock-gated execution, aggregate analysis, and manuscript asset
generation for ds006780 are complete. The v5 precision gate failed; all
participant-level artifacts remain outside Git and the result is reported as
secondary uncertainty-limited transfer evidence.

**Goal:** test whether the observed REVE head comparison transfers to an
independently recruited, age-compatible EEG cohort without tuning on its targets.

**Architecture:** preserve all existing evidence; use a separate, prospectively
locked external extension with frozen HBN-trained checkpoints, an explicit
signal/coordinate contract, and paired subject-level evaluation.

**Tech stack:** existing Python, MNE, PyTorch, manifest hashing, checkpoint
inventories, paired analysis, and aggregate asset generation. Extend the current
evaluation components only after a dataset passes feasibility checks.

For agentic workers: use the executing-plans workflow to implement the gated
tasks below. Do not treat this planning document as a finalized execution lock.
Leave changes uncommitted; do not modify primary study locks or inventories.

## Why this extension

The primary comparison, capacity-data analysis, and layer-wise analysis all
evaluate the same 75 MIPDB subjects. More predictions, layers, and optimization
seeds do not supply independent external participants. The next useful test is
whether a fixed head contrast recurs elsewhere, not another MIPDB-selected winner.

This remains an extension of the current investigation. The original primary
result remains unchanged. The new experiment is motivated by already observed
MIPDB results and must be described as such; only its new cohort evaluation can
be prospective. A new encoder and full-HBN retraining are not prerequisites.

## Candidate screening

This is a targeted screen, not an exhaustive census. Published total counts are
not eligible evaluation counts. Exact age support, signal QC, independence, and
access must still be checked before declaring a usable cohort.

| Candidate | Verified source facts | Decision for the current transfer contract |
| --- | --- | --- |
| Kang et al., *Development of EEG connectivity from preschool to school-age children* | 253 children aged 3-10; eyes-open EEG acquired with a 128-channel HydroCel net at 1,000 Hz. The authors offer raw data on request. Their analyzed derivative uses filtering and a reduced channel set. | Best acquisition match among these screened candidates, but not execution-ready: request original raw channels, continuous age at recording, recording boundaries, and reuse conditions. Ages below the HBN support are ineligible for the primary replication. |
| OpenNeuro ds006780, *SFARI_EEG multi-paradigm dataset* | Public raw BIDS EEG from 136 listed participants. In the pinned `1.0.0` signal/metadata snapshot, 128 candidates satisfy the target-free `run-01` recording contract; 126 have exact usable ages after two explicit metadata exclusions. Acquisition is BioSemi ActiveTwo, 64 EEG channels at 512 Hz, plus eight EMG channels and a trigger channel. Resting state is eyes-open, approximately 62-second blocks, with 1-8 runs observed and up to six blocks described across two sessions. The cohort contains typically developing, ASD, and ASD-sibling groups. A pinned audit found all 686 resting channel sidecars exactly match the canonical BioSemi-64 channel set and order; the target-free adapter and full signal QC passed. | Completed v5 secondary transfer candidate. The checked BIDS tree has no subject-specific `electrodes.tsv` or `coordsystem.json`, the sidecar leaves EEG reference as `n/a`, duration is variable, and clinical/sibling grouping creates domain shift. The fixed canonical BioSemi-64 approximation and preserve-acquisition reference policy are reported as limitations; the precision gate failed. |
| Iranian 6-11 years population-based EEG, ERP, and cognition dataset | 100 non-clinical children aged 6-11 years, with age recorded in months. Raw EDF recordings include four minutes eyes-open and four minutes eyes-closed resting EEG. Acquisition uses 19 scalp electrodes in the 10-20 system, 250 Hz sampling, and linked-ear reference. The authors describe controlled Synapse access; the institute download portal additionally requires name, email, and acceptance of research-use/no-redistribution terms. | Strong developmental candidate for a separate low-density cross-montage transfer: exact age and raw resting data are attractive, but 19-channel 10-20 data are not a matched HydroCel replication. Do not access or execute until the reuse terms and participant-level age/QC manifest are independently confirmed. |
| OpenNeuro ds006923, *Dataset of Electroencephalograms of Juvenile Offenders* | 140 participants; BioSemi ActiveTwo 128 channels; exact age in years is available for every participant, with support 14-19 years. The public derivative contains 128-channel resting-state recordings at 128 Hz, including a 1-40 Hz preprocessed derivative, per-subject channel/electrode/coordinate metadata, and no listed original BDF/FDT acquisition files. The cohort is all male and contains 74 juvenile offenders plus 66 non-offender controls. NEMAR lists CC0 and an approximately 8 GB download. | No-go as a drop-in replication. The age support is narrow relative to HBN, the cohort is all male, and the available signal is a derivative with a distinct BioSemi/coordinate/preprocessing contract. It can be considered only as a separately specified cross-montage transfer or domain-shift stress test, not as a silent replacement for the primary external replication. |
| MPI-LEMON | The project describes 228 participants, ages 20-35 or 59-77, with 62-channel resting EEG and public download links. | Not the preferred developmental replication: almost all advertised age support lies outside 5.06-21.67 years, and the montage differs. Do not reinterpret older-adult extrapolation as the same in-support task. |

Sources for the corresponding rows:

- [Kang et al.: acquisition and data availability](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1277786/full).
- [ds006780 source README](https://github.com/OpenNeuroDatasets/ds006780),
  [ds006780 participants metadata](https://github.com/OpenNeuroDatasets/ds006780/blob/main/participants.tsv).
- [Iranian dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11846862/),
  [author access portal](https://ibmhi.ir/iranian-child-brain-dataset/),
  [controlled Synapse record](https://www.synapse.org/Synapse:syn64112114).
- [ds006923 source README](https://github.com/OpenNeuroDatasets/ds006923),
  [participants metadata](https://github.com/OpenNeuroDatasets/ds006923/blob/main/participants.tsv),
  [dataset description](https://github.com/OpenNeuroDatasets/ds006923/blob/main/dataset_description.json),
  [NEMAR distribution record](https://nemar.org/dataset/on006923).
- [MPI-LEMON project description](https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html).

The metadata-only audit of ds006923 is complete, and ds006780 plus the Iranian
cohort were added as further candidates after repository/access metadata
screens. Those metadata screens were intentionally performed without downloading
EEG or running a model. A later target-free ds006780 pilot is documented below;
it does not approve confirmatory inference. The public metadata are sufficient
to reject ds006923 as a drop-in replication, while ds006780 and the Iranian
cohort remain secondary transfer candidates that require a frozen adapter,
reference policy, run/session aggregation, and precision validation.

### Metadata-only screen of ds006780

The public repository exposes raw BIDS files and participant ages. The
execution snapshot used OpenNeuro version `1.0.0` at source commit
`799d1502296ba5f74033734e149160c3d333e470` and found 136 participant rows,
135 non-missing ages from 7.9 to 14.5 years, and 128 target-free `run-01`
recording candidates. Of those candidates, 126 have a usable exact age; one
participant row is absent and one age is `n/a`, both recorded as exclusions.
The checked resting-state sidecar specifies continuous
approximately 62-second eyes-open recordings at 512 Hz with 64 EEG channels,
eight EMG channels, and one trigger channel. The repository contains 686 BDF
sidecars/files across 131 subjects, with variable run counts; it contains no
subject-specific `electrodes.tsv` or `coordsystem.json` files. The sidecar also
leaves the EEG reference as `n/a` and does not declare software filters.

Up to six resting blocks can occur across two sessions according to the README,
but the BIDS tree has no explicit `ses-*` directories. Subject-level
run/session aggregation and the treatment of incomplete runs must therefore be
fixed from the manifest before any external evaluation. The cohort is
clinically heterogeneous (ASD, typical development, and ASD-sibling groups),
which is a domain-shift factor rather than a nuisance to hide.

The current suitability decision is **conditional**:

- target availability and age support: pass for 126 subjects, with two explicit metadata exclusions;
- raw EEG and subject-level resting files: pass;
- independent cohort and mixed-sex coverage: pass provisionally;
- canonical BioSemi-64 channel mapping: pass as a standardized montage
  approximation; subject-specific coordinates remain unavailable;
- EEG reference provenance: unknown in the source sidecar, with the fixed
  `preserve_acquisition` policy recorded explicitly;
- fixed run/session aggregation: resolved for the predeclared one-subject
  `run-01` contract;
- precision for the eligible subset: approved maximum 95% CI width `0.06`.

The coordinate mapping gate is resolved only as a canonical standardized
approximation, and the source reference remains unknown. The completed v5
execution preserves that policy and is not a claim of harmonized acquisition.
Because the extension was motivated by earlier results and its precision gate
failed, it is reported as secondary transfer evidence rather than a new
confirmatory claim.

### Target-free technical pilot of ds006780

The pilot used OpenNeuro version `1.0.0` and one resting BDF file outside the
repository. It did not read participant age, create targets, compute
predictions, or calculate any metric. The following checks passed:

- MNE read 73 raw channels and 31,744 samples at 512 Hz for a 62-second run.
- The `channels.tsv` order matched the raw BDF order; 64 EEG channels were
  selected and the eight EMG plus trigger channels were excluded.
- The EEG order matched the canonical `biosemi64` montage, and MNE accepted
  all 64 finite standard positions without interpolation.
- The sole `Recording_start` event at 2.802734375 seconds was used as the
  block boundary. No resting condition marker was present in this run.
- The common target-free preprocessing produced 29 finite windows of shape
  `64 x 400` at 200 Hz, using the existing 0.5--99.5 Hz band-pass,
  non-overlapping two-second windows, and no cross-block windows.
- A frozen REVE forward pass accepted a batch of shape `2 x 64 x 400` and
  returned both declared layers with shape `2 x 128 x 512`. The encoder stayed
  in evaluation/inference mode and its state hash was unchanged.

The sidecar reports `EEGReference: n/a`. Two engineering-only preprocessing
variants were compared to expose the risk of guessing: no additional
rereference and average reference. Both produced finite windows, but average
referencing changed the standardized windows materially (mean absolute
difference `0.361`, maximum absolute difference `30.0` after the declared
clamp). Therefore average reference is not silently enabled. The primary
candidate policy is to preserve the stored acquisition with no additional
rereference, matching the existing HBN reader behavior when no explicit
reference channel is declared; this remains a design decision to validate and
freeze before external inference. Average reference may be retained only as a
predeclared robustness analysis applied consistently to every cohort.

### Target-free adapter implementation gate

The repository now contains a dedicated `ds006780` adapter package for the
pre-inference gate. It is intentionally separate from the MIPDB adapter and
does not read `participants.tsv`, age values, predictions, or metrics during
manifest construction or signal QC. The target-free manifest records the
OpenNeuro version, source commit, per-file size and SHA-256 identities,
environment/lockfile identity, canonical BioSemi-64 montage identity, ordered
channel/event inventories, explicit reference policy, and out-of-scope runs.

The first execution contract is one `task-Restingstate`, `run-01` recording
per subject. Other runs and session variants are recorded as out of scope.
The adapter requires 512 Hz source data, exactly 64 canonical BioSemi-64 EEG
channels, no spatial interpolation, and the BIDS recording boundaries. The
montage coordinates are a standardized BioSemi-64 approximation supplied by
the pinned MNE montage; they are not subject-specific measured electrode
locations.

The primary reference policy is `preserve_acquisition`: when the sidecar says
`n/a`, the adapter performs no additional rereferencing and records unknown
reference provenance. `average_reference` exists only as an explicitly
declared sensitivity policy. The QC artifact is target-free and create-only;
age-bearing finalization is a separate command that joins exact subject IDs
against a separately hashed participant metadata source.

The adapter gate was not itself an inference approval. The precision gate was
approved with a maximum primary confidence-interval width of `0.06`, which
authorized target-bearing cohort finalization and the separately locked v5
execution. The observed v5 width exceeded that threshold, so the resulting
evidence is uncertainty-limited and cannot support stable superiority.

The implemented adapter was then exercised on one locally retrieved BDF
outside the repository. Manifest verification passed, target-free QC produced
29 finite windows of shape `64 x 400`, and the frozen REVE forward pass
returned `29 x 128 x 512` for both declared layers (`-2` and `-1`). The encoder
remained frozen, in evaluation/inference mode, and its state digest was
identical before and after the pass. The resulting manifest and QC artifacts
remain in staging outside Git; no age, prediction, or metric artifact was
created.

The same target-free contract was then applied to the complete selectively
downloaded `run-01` signal tree for the current public snapshot. The staging
download contained 648 files and approximately 1.1 GB of signal/sidecar data;
`participants.tsv` was not downloaded or read. The sealed manifest contained
128 eligible recording candidates and one structured exclusion: `sub-1536`
was out of scope because its required `events.tsv` was missing. All 128
eligible recordings passed signal QC, yielding 4,121 finite non-overlapping
two-second windows in total. The observed per-recording window counts were
29 (111 recordings), 60 (13), 30 (3), and 32 (1); this variability is retained
as a QC fact and was not used to select subjects. Every QC artifact used the
declared `preserve_acquisition` policy, shape `64 x 400`, source frequency
512 Hz, and had no forbidden target-bearing fields. At that stage these results
authorized only the next precision and inference-lock review; the later v5
execution used the resulting frozen contract.

### Planning-only precision simulation

Before any participant metadata or external predictions were accessed, a
deterministic simulator was added for the exact hierarchical paired bootstrap
used by the primary contrast. It was run against the verified target-free
manifest count (`n=128`) with 100 synthetic cohort replications and 1,000
bootstrap iterations per replication. The scenario file is explicitly marked
planning-only; its correlations, paired-error correlation, and seed variation
are assumptions rather than external observations.

The illustrative grid produced valid results for all replications. Across the
four scenarios, median 95% interval widths ranged from `0.0554` to `0.0582`,
and the 95th-percentile widths ranged from `0.0659` to `0.0677`. The normal and
uniform synthetic target-shape scenarios were similar. These numbers are
useful for choosing a scientifically meaningful precision threshold, but they
are not a power claim or an external result. The threshold was subsequently
approved at `0.06`; the planning artifact remains outside Git.

### Target-bearing cohort finalization

After the precision threshold was approved, the public `participants.tsv` was
retrieved separately from the signal tree and hashed outside Git. The finalizer
joined it to the target-free manifest and all 128 signal-QC reports without
reading or changing the EEG artifacts. It produced a target-bearing manifest
with 126 subjects having exact ages inside the HBN support `[5.06, 21.67]`
years. Two candidates were retained in the release ledger as exclusions:
one `missing_participant_metadata` and one `missing_age`. No age was imputed,
rounded into eligibility, or silently dropped. The target-bearing manifest is
staging-only and is not a public aggregate release.

### Execution lock and external runner

The repository now contains a separate `ds006780` lock and runner. The seal
binds the approved external configuration, target-free manifest, target-bearing
manifest, the complete 90-run capacity inventory, and the predeclared 40-run
subset (`n=200` and `n=800` for `mean_linear` and
`mean_rich_stats_residual`, seeds 33--42). It also records the checkpoint core
provenance, REVE encoder state hash, execution source hash, subject inventory,
and expected 5,040 subject-by-run predictions for the current 126-subject
cohort.

The runner refused altered identities, recomputed and compared stored
target-free signal QC, extracts final-layer representations once per subject,
and evaluates all selected heads from the same frozen tensors. It writes only
immutable subject-level prediction files and a complete prediction inventory;
aggregate metrics are intentionally deferred to the separate analysis step.
The command names are `scripts/seal_ds006780_external.py` and
`scripts/run_ds006780_external.py`. The completed v5 output is summarized in
`results/extensions/ds006780_external_v5/`; raw predictions and target-bearing
manifests remain in the external workspace.

### Metadata-only screen of the Iranian 6-11 cohort

The paper and author access portal report 100 non-clinical children aged 6-11,
with age recorded in months, two four-minute resting conditions, and raw EDF
files. The acquisition uses 19 electrodes in the international 10-20 layout at
250 Hz with linked-ear reference. This makes the cohort attractive for testing
whether the head comparison survives a much lower-density sensor layout, but it
does not provide a matched HBN/HydroCel replication. The access portal requires
identifying contact information and acceptance of research-use and
no-redistribution terms, so no data were requested or downloaded during this
screen.

The candidate remains blocked until the exact permitted use, participant-level
age/QC metadata, and a source-preserving 19-channel adapter are confirmed.

### Metadata-only audit of ds006923

The audit queried only public text metadata and repository file listings. No EEG
signal, participant-level file, or model output was written to this repository.
The following facts are now verified:

- `participants.tsv` has 140 rows, an `Age` field in years with no missing
  values, and values from 14 through 19 years. The age counts are 2, 22, 35,
  42, 33, and 6 for ages 14, 15, 16, 17, 18, and 19, respectively.
- The cohort is all male and comprises 74 offenders and 66 non-offender
  controls. The group/age composition is asymmetric, so a transfer result
  must not be described as a clean age-only replication without reporting this
  domain shift.
- Each of the 140 subject folders exposes `channels.tsv`, `electrodes.tsv`,
  and `coordsystem.json`, with 128 EEG channels. The checked sidecars specify
  128 Hz sampling, average reference, and a 1-40 Hz FIR-filtered continuous
  derivative; a separate four-minute epoched derivative is also listed.
- `dataset_description.json` marks the dataset as a `derivative`, and the
  public file inventory lists `.set` derivatives rather than the original
  acquisition format. The adapter therefore must preserve the source
  coordinate/signal provenance and cannot pretend that this is the MIPDB raw
  recording contract.

The audit exit condition is therefore **screened, not approved**. The next
decision is between (a) keeping ds006923 as a predeclared secondary
cross-montage stress test, or (b) continuing the search/request process for an
independent developmental cohort with broader age support and a closer
acquisition contract.

## Compatibility and independence are separate checks

The current evaluation uses ordered EGI E1-E128 channels, 200 Hz samples,
0.5-99.5 Hz filtering, recording-level target-free normalization, and
non-overlapping two-second windows within condition boundaries. Identical
channel counts do not imply identical electrode positions. Upsampling a
128 Hz, 1-40 Hz derivative cannot restore its missing spectral information.

REVE itself accepts spatial coordinates and variable electrode layouts; that
capability does not establish that this repository's fixed-layout contract or
trained heads transfer unchanged. Its published pretraining inventory includes
HBN releases and many other datasets. A fresh dataset identifier, later mirror
date, or lack of a name match does not prove encoder-level independence.
[REVE methods and Appendix B](https://papers.neurips.cc/paper_files/paper/2025/file/20a917f77773ac0fa8bea2bdd6606b66-Paper-Conference.pdf).

Record two distinct assessments: participant/recording independence from HBN
and MIPDB, and potential exposure during encoder pretraining. Check original
study names, aliases, releases, acquisition provenance, and any overlap
statements. Unknowns remain explicit; do not claim certified absence of overlap.
Do not attempt to reidentify participants across public pseudonyms.

## Minimal experimental design

Use a small replication matrix before considering a second encoder:

- Heads: final-layer `mean_linear` and `mean_rich_stats_residual`.
- HBN training sizes: 200 and 800 subjects from the existing capacity-data run.
- Seeds: the same ten seeds, 33-42, paired between heads and sizes.
- Inventory: 40 existing checkpoints; zero new head-training runs if those
  exact checkpoint files remain available and pass their recorded hashes.
- Primary contrast on the new cohort: rich minus linear subject-level Pearson
  at 800 training subjects, averaged over the ten paired seeds.
- Secondary contrasts: the same comparison at 200 subjects and the direct
  difference of paired advantages between 800 and 200. These test the observed
  training-regime pattern without choosing a new optimum on external results.
- Secondary descriptive metrics: MAE, RMSE, R-squared, and calibration, including
  paired MAE differences. Do not replace Pearson after seeing the new results.

The two-head choice is informed by MIPDB and is not a reconstruction of the
original prespecification. This minimal matrix tests one head contrast and a
size contrast; it cannot replicate the multi-query versus rich-statistics
metric-ranking difference. Replicating that particular ranking would require
adding the already trained multi-query head before the new evaluation lock.

Retain subject-level aggregation and paired resampling across every compared
head, seed, and size. Repeated windows or conditions from one person remain one
external unit; account for families or other recruitment clusters if present.
Analyze the new cohort separately from MIPDB. A pooled correlation is not a
replication test and can be distorted by differing age distributions.

Before execution, fix the confidence procedure, bootstrap seed/count, exact
test family, and decision rule in the new analysis lock. The existing
conjunctive rule can be reused for the single primary contrast, with one primary
test rather than the original three-candidate Holm family. Secondary contrasts
remain exploratory and cannot rescue a failed primary decision. A zero-crossing
interval means uncertainty, not practical equivalence or absence of benefit.

## Gated tasks

### 1. Resolve access and metadata

- Obtain permission to send an enquiry if external contact is needed. Signing
  agreements, institutional representations, and accepting new conditions are
  user decisions, not automatic execution steps.
- Verify exact ages at EEG acquisition, units, repeated visits, resting-state
  conditions, original sampling/filtering, channel labels/positions/reference,
  and raw-versus-derivative status. Age bins are not exact regression targets.
- Pin a specific dataset snapshot and document access/attribution conditions.
  Keep participant metadata and signals outside Git. Do not reopen the deferred
  demographic-supplement release workflow as a prerequisite for this enquiry.
- Determine eligible count and age coverage without generating age predictions.
  The published total is an upper bound, not a sample-size promise.

Exit: documented access and metadata decisions. A request-only source stays
blocked until the data holder supplies the necessary material.

### 2. Check precision and freeze the design

- Define an acceptable primary-contrast interval width, justified by the
  scientific effect of interest, before running precision simulations. Record
  the decision criterion, scenario assumptions, and required sensitivity checks.
- Simulate paired-contrast precision using the projected post-pilot, post-QC
  cohort, its age distribution, and any recruitment clusters, over plausible
  effect/correlation scenarios. Recheck with the final eligible manifest in
  Task 3. Report assumptions and sensitivity, not a guaranteed power claim.
- If the precision criterion fails, defer the confirmatory replication or
  explicitly redesign it as estimation-only before any external predictions.
  Record which outcome applies; do not quietly lower the criterion to proceed.
- The existing 50-subject floor alone is not a power calculation. Do not change
  cohort inclusion or continue recruiting until a desired p-value appears.
- Reserve a small deterministic engineering pilot, exclude it from the final
  evaluation, and keep target-based model outputs out of the pilot workflow.
- Verify the selected capacity-data checkpoint inventory. If weights are
  unavailable, stop before promising zero retraining; any reconstruction needs
  a documented development-only protocol and its own identities.
- Finalize eligibility, QC, conditions, aggregation, contrasts, analysis rule,
  exclusions, and stopping rule before target/prediction joining.

Exit: a new versioned design specification with an explicit precision decision,
not yet an execution lock and not edits to existing locks. Adapter validation
and final cohort QC still precede execution sealing.

### 3. Implement only the necessary adapter

- Start from the existing manifest, extraction, and external evaluation code;
  do not force a new dataset through an MIPDB-specific parser.
- Add an explicit dataset adapter if needed, preserving the source coordinate
  system and signal units. Reject unsupported layouts or derivatives instead
  of renaming channels or inventing positions.
- Test with synthetic fixtures: channel-order invariance when labels/positions
  are reordered together, wrong/missing positions, units, sampling bandwidth,
  repeated subjects, condition boundaries, missing ages, and age-support rules.
- Test fail-closed behavior for checkpoint/hash mismatches, mixed dataset
  identities, incomplete inventories, and duplicate subjects.
- Test paired analysis with known synthetic predictions, clustered windows,
  all planned contrasts, and refusal to analyze an incomplete matrix.
- After the target-free pilot and predeclared signal QC, verify final cohort
  eligibility and recheck the fixed precision criterion. Freeze the validated
  adapter, preprocessing, analysis implementation, and environment identities.
- Create the execution lock only now: bind those validated identities, the
  original REVE encoder identity, all 40 checkpoint hashes, the dataset
  snapshot, final cohort manifest, pilot exclusions, and analysis specification.
  Do not produce final-cohort model predictions before this lock exists.

Exit: adapter and analysis tests pass, and the target-free pilot validates the
declared signal contract; the precision gate passes for a confirmatory run, or
an estimation-only design is explicitly locked. Any necessary montage or
bandwidth change is a new transfer design, not an undocumented repair to the
old one. A deferred design does not advance to Task 4.

### 4. Resource preflight and evaluation

- Estimate disk from actual eligible raw recordings, staged preprocessing,
  representation cache shape/dtype, and safe headroom. Do not use a catalogue
  archive size as the full working-space estimate.
- Extract once per subject/layer. These two heads can use their required pooled
  statistics; avoid retaining full token caches when they are not consumed.
- Lock the exact inventory before joining targets, evaluate all planned
  checkpoints, and produce an immutable complete prediction inventory.
- Publish progress as completed subjects/checkpoints, failures, throughput,
  remaining work, disk/RAM, and ETA uncertainty. No indefinite blind retries.

Exit: complete hash-valid evaluation or a documented failure; no partial-result
winner selection or checkpoint replacement after external inspection.

### 5. Analyze and retain bounded conclusions

- Report independent subject counts, eligibility/QC flow, cohort-specific
  metrics, paired intervals, and direct size contrasts, including null or
  unfavorable results. Explain age-range restriction when interpreting Pearson.
- Keep MIPDB discovery evidence and the new replication separate. Agreement
  across two cohorts is stronger evidence, not proof over all populations.
- Export only permitted aggregate artifacts and provenance. Retain existing
  primary, capacity-data, and layer-wise evidence byte-for-byte.

Exit: a reproducible external extension, irrespective of the direction of the
result. A second encoder is a subsequent option, not an automatic extra matrix.

## Access enquiry draft (not sent)

Subject: Access enquiry for developmental resting-state EEG replication

We are investigating chronological-age decoding from frozen EEG
representations and would like to assess whether your developmental cohort
could support an independent external evaluation. Would it be possible to
obtain the original resting-state EEG, continuous age at each recording, and
the acquisition/channel/condition metadata under your applicable reuse terms?
We would not train or select models using the external age targets. Raw
signals and participant-level metadata would not be redistributed through our
code repository.

Could you confirm whether the original full-channel recordings are available,
which data-access or ethics requirements apply, and whether the recordings
have previously been released under another dataset name or supplied for EEG
foundation-model pretraining? An initial metadata-only response would be
sufficient to establish feasibility; no participant-level data are requested
by email.
