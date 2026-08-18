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

Large `*.ckpt` files are intentionally excluded from the repository copy;
the completed checkpoints remain on the Vast.ai workspace.
