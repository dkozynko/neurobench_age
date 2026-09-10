# ds006780 precision-gate decision memo

Status: approved by the author; the external config records
`max_primary_confidence_interval_width = 0.06`.

The primary external estimand is the paired difference in subject-level
Pearson correlation between `mean_rich_stats_residual` and `mean_linear`,
averaged over the ten prespecified seeds. The precision gate is the maximum
allowed width of its two-sided 95% confidence interval. It must be chosen from
the scientific effect of interest, before reading external ages or producing
external predictions.

The planning-only simulation used the verified target-free count (`n=128`),
100 synthetic cohort replications, and 1,000 hierarchical paired-bootstrap
iterations per replication. It is sensitivity evidence, not an external
result:

| Scenario | Median width | 95th-percentile width |
| --- | ---: | ---: |
| Zero effect | 0.0582 | 0.0677 |
| Small effect (0.01) | 0.0579 | 0.0672 |
| MIPDB-scale effect (0.03) | 0.0554 | 0.0659 |
| Uniform target-shape sensitivity | 0.0556 | 0.0675 |

## Recommended criterion

Use `max_primary_confidence_interval_width = 0.06` if the scientific goal is
to estimate the head advantage to approximately ±0.03 Pearson units. This is
the stricter, effect-scale-aligned choice; the actual external interval may
fail it, in which case the study must be reported as estimation-only or
redesigned rather than silently weakening the criterion.

## Alternative criterion

Use `0.07` only if the author explicitly defines the goal as a wider
descriptive external estimate. It is close to the 95th-percentile planning
width and therefore should not be described as high-precision confirmation of
a small effect.

The selected threshold was written to the external config before target-bearing
finalization. The target-free manifest and QC were rebuilt under the resulting
identity, and the public participant metadata were joined in a separate
target-bearing staging manifest. The completed v5 execution then produced an
observed primary interval width of `0.1272`, so the precision gate failed. The
result is reported as uncertainty-limited secondary transfer evidence; raw
predictions and target-bearing manifests remain outside Git.
