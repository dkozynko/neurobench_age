# Evidence registry

This registry separates prospective evidence from retrospective and
reproduction material. Raw data, model weights, caches, full predictions, and
logs remain external artifacts whose identities are retained by hash.

| Evidence class | Source | Retained evidence | Allowed interpretation |
| --- | --- | --- | --- |
| Primary prospective study | sealed lock, checkpoint and prediction inventories, confirmatory analysis | compact protocol identities, 75-subject cohort summary, 40-run matrix, 3,000-record inventory identity, paired statistics | Stable external improvement only when every predeclared criterion passes |
| Secondary reproduction evidence | official NeuralBench full fine-tuning path and `src/neurobench_age/pipelines/independent.py` | protocol checks, compact metrics, implementation comparison | Reproduction and sensitivity context for end-to-end age prediction |
| Retrospective secondary evidence | historical HBN R5 records under `results/canonical/` | per-seed metrics, gates, and compact negative results | Motivation and retrospective description only; not model or head selection |
| Capacity--data extension | finalized 90-run extension lock, prediction inventory, exploratory analysis, and aggregate asset manifest | 3 training sizes, 3 heads, 10 seeds, 6,750 external prediction identities, aggregate tables and figures | Exploratory evidence about paired head behavior across the tested data regimes; does not modify the primary confirmatory claim |
| Layer-wise extension | finalized four-layer external prediction inventory, exploratory analysis, and aggregate asset manifest | layers -4/-3/-2 versus -1, 10 seeds, 75 external subjects, aggregate table and figures | Exploratory evidence about representation depth; does not modify the primary confirmatory claim |
| ds006780 external transfer v5 | sealed v5 execution lock, aggregate analysis, precision gate, and manuscript asset manifest | 2 heads, 2 training sizes, 10 seeds, 126 external subjects, 5,040 prediction identities, aggregate table and figure | Secondary cross-cohort transfer evidence; the precision gate failed, so it does not establish stable superiority or modify the primary claim |
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

## Capacity--data extension artifact map

The extension is retained as aggregate evidence under
`results/extensions/capacity_data_regime_v3/`. Its manifest binds the 90-run
analysis to the finalized lock, checkpoint inventory, prediction inventory,
and exploratory inference specification. The analysis digest is
`1bd8143b8d917d36082f23e5e936bf2b5f873f3e8803e144b96d3e2297f8d974` and the
asset-manifest digest is
`95d68b95bd621f547f168a51a86323b97268c8ae73f3182828f957bc196e955b`.
The stored outputs contain only aggregate metrics, seed-level contrasts,
intervals, tables, and figures; no participant-level rows, identifiers, or
raw signals are released.

## Layer-wise extension artifact map

The layer-wise extension is retained as aggregate evidence under
`results/extensions/layerwise_probe_20260910/`. Its report binds the completed
75-subject, four-layer by ten-seed external prediction inventory to the
layer-wise analysis configuration, representation protocol, and training
protocol. The analysis digest is
`f1f757ef53a8aad10615afd7fa0d6f8e2bc4e7f04ec33ce2d870f92e44706565` and the
asset-manifest digest is
`14b5afe5623f34fbaed4fcbe98ee8b59528769bf468bd1bcd83b941353a2be9d`.
The retained table and figures contain aggregate metrics only. Mean paired
deltas were +0.042, +0.030, and +0.008 for layers -4, -3, and -2; every
hierarchical-bootstrap interval crossed zero. Exact sign-flip p-values and
Holm-adjusted values are descriptive exploratory inference and do not modify
the primary confirmatory claim.

## ds006780 external-transfer artifact map

The completed v5 transfer is retained as aggregate-only evidence under
`results/extensions/ds006780_external_v5/`. Its execution lock is
`a68338a1569d73fbbc46cc6a56c8e4ed3b39770edbf71ba85cbe284ec2b499df`; the
encoder checkpoint identity is
`4dbd8c07f2322e9f8db156692d5572a653bff83d2b0fd895e41fc2d632cf9986`; and the
prediction inventory identity is
`8fccac2fdfce62015bdd3b7ebc729e92beb4469eaf46064657d9b4e5a4b6df6a`.
The aggregate analysis identity is
`a357dc6a46492e6e23184dafe6b5ea28ad30419b1df83abd9f820364d34a3a25`.

The v5 primary n=800 rich-versus-linear contrast was +0.0287 with a 95%
hierarchical paired seed--subject interval of [-0.0349, 0.0922]. The observed
width 0.1272 exceeded the predeclared 0.06 threshold; the precision-gate
artifact therefore records `failed` and the allowed interpretation is
uncertainty-limited transfer evidence. Only aggregate JSON, LaTeX, figure, and
hash metadata are retained. Target-bearing manifests, participant IDs,
subject-level predictions, checkpoints, and raw signals remain in the
access-controlled external workspace.

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
