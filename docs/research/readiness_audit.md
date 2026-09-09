# Scientific experiment readiness audit

Date: 2026-09-09

## Verdict

The sealed prospective frozen-representation study is evidence-complete. The
exact four-head by ten-seed checkpoint inventory contains 40 runs, external
evaluation contains 3,000 predictions for 75 primary MIPDB subjects, and the
confirmatory analysis is complete. No tested complex head established a stable
external gain under the predeclared joint rule; this does not establish
equivalence.

The operational workflow is closed. Any additional experiment must receive a
new protocol and evidence identity rather than modifying the sealed result.

## Completed controls

- The executable representation protocol fixed the encoder, layers, four heads,
  preprocessing, cohorts, inferential procedure, and decision thresholds.
- A separate final training protocol fixed seeds 33--42, global seeded window
  shuffling, AdamW, OneCycleLR, gradient clipping, a 40-epoch maximum, patience
  7, and earliest-best validation checkpoint selection.
- MIPDB inventory, pilot allocation, acquisition files, target-free QC, and
  cohort lists were content-addressed before primary inference.
- The ten-subject pilot was permanently disjoint from primary and extrapolation
  cohorts and produced no age-prediction metric.
- HBN representation materialization excluded R5, verified raw-file identities,
  cached only layers -2 and -1, and supplied immutable train/validation inputs
  to head fitting.
- Artifact-derived sealing verified the encoder state, environment, source tree,
  HBN manifests, MIPDB acquisition/QC manifests, subject lists, all 40 run
  manifests, and all selected checkpoint hashes.
- External evaluation wrote its start marker before lazy primary-cohort
  extraction, allowed exact resume only, and created an immutable inventory of
  3,000 unique head--seed--subject records.
- Confirmatory analysis started only after completion and applied the
  predeclared hierarchical bootstrap, paired seed-level sign-flip test, Holm
  correction, and conjunctive stability decision.
- A strict compact importer now recomputes the complete hash chain and decision
  before retaining aggregate evidence. It excludes sample-level predictions,
  participant IDs, external paths, model binaries, caches, logs, and secrets.
- The metadata-first MIPDB aggregate workflow is implemented separately from
  the primary estimand. It validates a concrete NEMAR release, the completed
  study lock, finalized cohort-QC hashes, target-free window counts, optional
  allowlisted sex categories, $k=5$ suppression, secondary suppression, and a
  controlled-review approval plus append-only release ledger. No restricted
  aggregate has been generated or placed in Git without the corresponding
  access-controlled source artifacts.

## Audited production identities

| Artifact | Identity |
| --- | --- |
| Study lock | `3694bcb11480aefdb1f29248ff456a8e918adba56025dfc582fb5d6f5a9b4174` |
| Checkpoint inventory | `dcd8a18b74c4b0373527b2c01330099bb2c76e819013e2396790004a9e3b24f5` |
| Prediction inventory | `3ec0042d613fc2d36937d7349081beb0739c4fb3da349e1229e17f226e187830` |
| Confirmatory analysis | `7747a16e11629164b524b2113ab2230d260012022fc46f68540c7f7578c2a3b6` |
| Compact artifact manifest | `472784323d244712bc1abe590c3081df05a1609231371f59d939ceefc4d2460a` |

## Result boundary

The baseline mean Pearson was 0.644. All three candidates had positive mean
paired deltas and passed the Holm-adjusted one-sided threshold, but every 95%
hierarchical-bootstrap interval crossed zero. The result supports a bounded
statement about failure to establish stable external improvement. It does not
show that the probes are equivalent, that the true effects are zero, or that
other heads cannot help.

The 75-subject primary cohort exceeded the prespecified minimum of 50. This is
a minimum-size decision gate, not a formal power analysis; the broad bootstrap
intervals remain the relevant uncertainty statement.

MIPDB is an external frozen-probe evaluation, not an official NeuralBench score.
Official NeuralBench full fine-tuning remains secondary reproduction evidence,
and previously inspected HBN R5 results remain retrospective.

## Remaining scientific limitations

- Cross-dataset shift mixes representation accessibility with differences in
  acquisition hardware, montage, recruitment, demographics, duration, and
  signal quality.
- Ten seeds characterize only the selected optimization procedure and produce a
  discrete exact randomization distribution.
- Primary inference is restricted to ages inside HBN training support; the 20
  extrapolation subjects are outside the confirmatory estimand.
- The published REVE source list does not name MIPDB, but this is evidence rather
  than a cryptographic guarantee of encoder-level independence; residual encoder
  pretraining uncertainty therefore remains.
- Correlation is not calibration: negative external R-squared and slopes above
  one caution against clinical brain-age interpretation.
- The retained HBN snapshot identities were revalidated during production, but
  raw acquisitions and full training manifests remain external and must be
  preserved with their content hashes.
