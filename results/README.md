# REVE Age head experiments

All runs use the same 500-subject resting-state manifest, protocol, and seed
33. The copied artifacts include reports, epoch-level test metrics, launch
logs, resolved configs, run metadata, and raw test predictions.

| Run | Head | Best validation Pearson | Final official test Pearson |
| --- | --- | ---: | ---: |
| sync5 | `last_avg` | 0.472 | 0.3001 |
| sync6 | `last` | 0.376 | 0.2097 |
| sync7 | `all` | 0.421 | 0.2668 |
| sync8 | `mean_linear` | 0.589 | 0.5942 |
| positions-fixed | `last_tuned` | — | 0.5734 mean across seeds 33/34/35 |
| positions-fixed | `last_avg` | — | 0.5219 mean across seeds 33/34/35 |
| positions-fixed | `last` | — | 0.5393 mean across seeds 33/34/35 |
| positions-fixed | `all` | — | 0.5658 mean across seeds 33/34/35 |
| positions-fixed | `mean_anchor` | — | 0.6077 ± 0.0554 across seeds 33/34/35 |

The `mean_anchor` run is the baseline-preserving residual head: it starts with
`gamma=0` (exact mean pooling) and initializes its trainable query from the
first training dummy sample. Its mean test Pearson is `0.6076949040`, which is
`+0.0103130738` above the same-seed `mean_linear` mean of `0.5973818302`.
The complete per-seed metrics are in
`artifacts/age_medium_500_seed33/reve_head_experiments/mean_anchor_train_dummy_positions_fixed_metrics.json`.

Large `*.ckpt` files are intentionally excluded from the repository copy;
the completed checkpoints remain on the Vast.ai workspace.
