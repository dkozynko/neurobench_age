# Evidence registry

This registry separates prospective evidence from retrospective and
reproduction material. Raw data, model weights, caches, full predictions, and
logs remain external artifacts whose identities are retained by hash.

| Evidence class | Source | Retained evidence | Allowed interpretation |
| --- | --- | --- | --- |
| Primary prospective study | `configs/research/external_frozen_probe.json`, sealed lock, external prediction inventory, confirmatory analysis | protocol and artifact hashes, cohort/QC counts, 40-run checkpoint inventory, paired statistics | Stable external improvement only when every predeclared criterion passes |
| Secondary reproduction evidence | official NeuralBench full fine-tuning path and `src/neurobench_age/pipelines/independent.py` | protocol checks, compact metrics, implementation comparison | Reproduction and sensitivity context for end-to-end age prediction |
| Retrospective secondary evidence | `results/canonical/`, including HBN R5 finalist records | historical per-seed metrics, gates, compact negative results | Motivation and retrospective description only; must not be used for model or head selection |
| Contract evidence | `tests/`, `src/neurobench_age/research/`, and `src/neurobench_age/analysis/` | synthetic tests, strict schemas, tamper checks | Demonstrates code behavior, not empirical model performance |

## Primary prospective artifact map

| Claim component | Required external artifact | Repository verifier | Current value |
| --- | --- | --- | --- |
| Frozen encoder and immutable representations | HBN and MIPDB cache metadata with state/config hashes | `src/neurobench_age/pipelines/frozen_probe.py` | `TBD_AFTER_EXECUTION` |
| Exact four-head × ten-seed training matrix | checkpoint inventory and 40 immutable run manifests | `src/neurobench_age/pipelines/frozen_probe_training.py` | `TBD_AFTER_EXECUTION` |
| One-time MIPDB evaluation | sealed lock, started/completed markers, prediction inventory | `src/neurobench_age/pipelines/external_holdout.py` | `TBD_AFTER_EXECUTION` |
| Stable external gain or bounded negative result | complete confirmatory analysis bundle | `src/neurobench_age/analysis/confirmatory.py` | `TBD_AFTER_EXECUTION` |

## Retrospective HBN/R5 boundary

Files under `results/canonical/` remain useful for auditing earlier experiments,
but repeated access to HBN R5 removes any untouched-holdout interpretation.
Their gates and finalist labels describe historical workflow state only. They do
not authorize a new primary claim and do not affect candidate membership in the
prospective frozen representation study.
