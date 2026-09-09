# Evidence registry

This registry separates prospective evidence from retrospective and
reproduction material. Raw data, model weights, caches, full predictions, and
logs remain external artifacts whose identities are retained by hash.

| Evidence class | Source | Retained evidence | Allowed interpretation |
| --- | --- | --- | --- |
| Primary prospective study | sealed lock, checkpoint and prediction inventories, confirmatory analysis | compact protocol identities, 75-subject cohort summary, 40-run matrix, 3,000-record inventory identity, paired statistics | Stable external improvement only when every predeclared criterion passes |
| Secondary reproduction evidence | official NeuralBench full fine-tuning path and `src/neurobench_age/pipelines/independent.py` | protocol checks, compact metrics, implementation comparison | Reproduction and sensitivity context for end-to-end age prediction |
| Retrospective secondary evidence | historical HBN R5 records under `results/canonical/` | per-seed metrics, gates, and compact negative results | Motivation and retrospective description only; not model or head selection |
| Contract evidence | `tests/`, `src/neurobench_age/research/`, and `src/neurobench_age/analysis/` | synthetic workflows, strict schemas, tamper checks | Demonstrates code behavior, not empirical model performance |

## Primary prospective artifact map

| Claim component | Retained verifier | Audited value |
| --- | --- | --- |
| Frozen encoder and immutable representations | study-lock protocol, encoder, preprocessing, HBN/MIPDB manifest, and source hashes | complete lock `3694bcb11480aefdb1f29248ff456a8e918adba56025dfc582fb5d6f5a9b4174` |
| Exact head-training matrix | checkpoint inventory identity | 40 unique runs; digest `dcd8a18b74c4b0373527b2c01330099bb2c76e819013e2396790004a9e3b24f5` |
| One-time external evaluation | prediction inventory identity | 75 primary subjects and 3,000 unique records; digest `3ec0042d613fc2d36937d7349081beb0739c4fb3da349e1229e17f226e187830` |
| Confirmatory inference | compact aggregate analysis and source identity | complete; digest `7747a16e11629164b524b2113ab2230d260012022fc46f68540c7f7578c2a3b6` |

The retained bundle is rooted at `results/canonical/prospective/`. Its terminal
manifest digest is
`472784323d244712bc1abe590c3081df05a1609231371f59d939ceefc4d2460a`.
The importer independently checks the exact four-head by ten-seed checkpoint
matrix, the common ordered 75-subject cohort, all 3,000 unique inventory
identities, the canonical hash chain, Holm adjustment, every condition in the
joint decision, and the bounded conclusion.

## Confirmatory result

The mean-pooled linear baseline reached mean external Pearson 0.644. Candidate
paired mean deltas were +0.008, +0.032, and +0.010 for the earlier-layer linear,
rich-statistics residual, and multi-query rich-statistics heads. Although the
one-sided Holm-adjusted tests and most seed-stability checks were favorable, all
three hierarchical-bootstrap intervals included zero. Therefore no tested
complex head established a stable external gain under the predeclared protocol.
This does not establish equivalence or a true zero effect.

## Provenance and privacy boundary

The sealed source artifacts retain content identities for the HBN acquisition
manifest, the separately rebuilt HBN training manifest, MIPDB acquisition and
QC records, the encoder state, source tree, environment, and preprocessing.
Production execution revalidated that rebuilt training snapshot before it was
bound into the final lock. The repository retains only those identities and
aggregate counts; participant IDs, participant-level predictions, remote paths,
and credentials are excluded.

## Retrospective HBN/R5 boundary

Files outside `prospective/` remain useful for auditing earlier experiments,
but repeated access to HBN R5 removes any untouched-holdout interpretation.
Their gates and finalist labels describe historical workflow state only. They do
not authorize a new primary claim and did not alter candidate membership in the
sealed frozen-representation study.
