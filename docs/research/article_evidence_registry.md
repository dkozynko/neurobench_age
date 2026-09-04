# evidence registry

This registry is the index for the evidence retained in the repository. Raw
checkpoints, full prediction exports, and launch logs are external artifacts.

| purpose | Canonical source | Evidence retained | Interpretation |
| --- | --- | --- | --- |
| Official baseline contract | `src/neurobench_age/core/baseline.py` | dry-run contract tests and protocol docs | Defines the reference setup |
| Independent reproduction | `src/neurobench_age/pipelines/independent.py` | compact comparison report and manifest hash | Checks implementation parity |
| Head stability | `results/canonical/head_comparison.json` | per-seed metrics, SD, wins, deltas | Primary stability evidence |
| Complexity limits | `results/canonical/complexity_limits.json` | accepted, rejected, and inconclusive candidates | Prevents cherry-picking a single win |
| Validation protocol | `configs/article/` and `docs/research/article_ready_protocol.md` | frozen configs and protocol | Separates screening from confirmation |
| Finalist gate | `src/neurobench_age/analysis/gate.py` and `scripts/check_article_gate.py` | validation-only gate JSON | Prevents test leakage before finalist selection |
| Final claim | `results/canonical/finalist/` | pre-test gate, sealed test summary, figures | Only predeclared finalist may access test |

Historical experiment notes may be summarized here, but they are not part of
the canonical execution path.
