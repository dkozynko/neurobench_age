# Age medium-500 results

This directory contains the completed seed-33 comparison between the official
NeuralBench REVE baseline and the independent reproduction.

## Dataset and protocol

| Property | Value |
| --- | --- |
| Subjects | 500 |
| Split | 400 train / 50 validation / 50 test |
| Manifest SHA-256 | `66f9a041d1ed398bfc2723d7f129c3950681bf7b0adf4346edc038c5bc25e04a` |
| Seed | 33 |
| Model | `NtReve`, `brain-bzh/reve-base` |
| Windows | 30,000 total; 24,000 / 3,000 / 3,000 |
| Checkpoint selection | maximum validation Pearson |

## Final comparison

| Metric | Official NeuralBench REVE | Independent pipeline | Difference (official - independent) |
| --- | ---: | ---: | ---: |
| Best epoch | 6 | 6 | 0 |
| Validation Pearson | 0.6120882119 | 0.6120882119 | 0 |
| Test Pearson | 0.4635040462 | 0.4635041391 | -0.0000000929 |
| Test RMSE | 3.3231191635 | 3.3231192005 | -0.0000000370 |
| Test MAE | 2.5717103481 | 2.5717103408 | +0.0000000073 |
| Test R² | 0.0455873609 | 0.0455884622 | -0.0000011013 |

The test Pearson difference is approximately `9.3e-8`, so the independent
implementation reproduces the official score to numerical precision for this
run. The machine-readable version is in `comparison.json`.

## Why epoch 6 is used

The checkpoint is selected using `val/pearsonr` with `mode=max`; test Pearson is
logged only as a diagnostic after each epoch. The final test score therefore
uses epoch 6 even though the diagnostic test score at epoch 7 is higher. This
keeps the test set out of model selection.

The per-epoch diagnostic values are in
`reve_baseline/epoch_test_metrics.jsonl`:

| Epoch | Test Pearson | Selected |
| ---: | ---: | :---: |
| 1 | 0.5972986817 |  |
| 2 | 0.5044884086 |  |
| 3 | 0.2012218237 |  |
| 4 | -0.1154505238 |  |
| 5 | 0.3113195896 |  |
| 6 | 0.4635040462 | yes |
| 7 | 0.4980050325 |  |
| 8 | 0.4562997520 |  |
| 9 | 0.4292816818 |  |
| 10 | 0.4263353646 |  |
| 11 | 0.4342620671 |  |
| 12 | 0.4719432294 |  |
| 13 | 0.3926305771 |  |

## Files

- `comparison.json` — normalized side-by-side metrics and parity verdict.
- `manifest/age_medium_500_resting_manifest.csv` — the exact 500-subject
  manifest used by both runs.
- `independent_pipeline/` — independent predictions, window manifest, run log,
  and normalized report.
- `reve_baseline/` — official report, per-epoch test metrics, configuration,
  cache collection metadata, run log, and failed-attempt diagnostics.

The raw logs retain the original remote paths and execution details. No model
checkpoint or HBN recordings were copied into the repository.
