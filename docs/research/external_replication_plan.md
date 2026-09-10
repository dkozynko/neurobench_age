# Independent external replication: feasibility and implementation plan

Date: 2026-09-10.

**Status:** metadata/literature screening completed for the candidates below;
no second cohort is approved for execution, downloaded, or evaluated.

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
| OpenNeuro ds006923, *Dataset of Electroencephalograms of Juvenile Offenders* | 140 participants; BioSemi ActiveTwo 128 channels. The README describes derivatives downsampled to 128 Hz and filtered to 1-40 Hz. NEMAR lists CC0 and an approximately 8 GB download. | No-go as a drop-in replication. Different coordinates and irreversibly different preprocessing require a separately specified cross-montage/preprocessing experiment. Exact age availability remains unverified. |
| MPI-LEMON | The project describes 228 participants, ages 20-35 or 59-77, with 62-channel resting EEG and public download links. | Not the preferred developmental replication: almost all advertised age support lies outside 5.06-21.67 years, and the montage differs. Do not reinterpret older-adult extrapolation as the same in-support task. |

Sources for the corresponding rows:

- [Kang et al.: acquisition and data availability](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1277786/full).
- [ds006923 source README](https://github.com/OpenNeuroDatasets/ds006923),
  [NEMAR distribution record](https://nemar.org/dataset/on006923).
- [MPI-LEMON project description](https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html).

The preferred next action is an access/metadata enquiry for the first candidate,
not bulk downloading the second candidate. No enquiry has been sent. If raw
data are unavailable, record a no-go and continue the cohort search; do not
silently relax the protocol to make an available dataset fit.

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
