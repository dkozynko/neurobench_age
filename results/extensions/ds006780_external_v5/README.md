# ds006780 external transfer v5

This directory retains the aggregate, non-participant-level evidence from the
completed ds006780 external transfer. The run evaluated the frozen
`brain-bzh/reve-base` encoder with `mean_linear` and
`mean_rich_stats_residual` heads at HBN training sizes 200 and 800, using seeds
33--42. It contains 40 selected head--seed runs and 5,040 subject-level
prediction identities in the external execution, but no prediction rows,
participant IDs, checkpoints, raw EEG, or target-bearing manifests are retained
here.

The sealed execution was complete for 126 eligible subjects. The primary
contrast is the n=800 rich-statistics residual head minus the matched n=800
mean-linear baseline. Its mean paired Pearson delta was `+0.0287113` with a
95% hierarchical paired seed--subject bootstrap interval of
`[-0.0349496, 0.0922320]`; the interval width was `0.1271816`.

The predeclared precision threshold was `0.06`, so the precision gate failed.
The correct interpretation is uncertainty-limited external evidence: all ten
n=800 seed deltas were positive, but the external-subject uncertainty does not
establish a stable superiority claim. The extension is secondary and does not
modify the primary MIPDB decision.

## Provenance

| Artifact | SHA-256 |
| --- | --- |
| `external_analysis.json` | `a357dc6a46492e6e23184dafe6b5ea28ad30419b1df83abd9f820364d34a3a25` (content identity) |
| `precision_gate.json` | `620588617a9c0700ddb996a24fedcb957b92e74d4d32b26a9d22ce72b8bd5db9` (artifact identity) |
| execution lock | `a68338a1569d73fbbc46cc6a56c8e4ed3b39770edbf71ba85cbe284ec2b499df` |
| prediction inventory | `8fccac2fdfce62015bdd3b7ebc729e92beb4469eaf46064657d9b4e5a4b6df6a` |
| encoder checkpoint | `4dbd8c07f2322e9f8db156692d5572a653bff83d2b0fd895e41fc2d632cf9986` |

The file hashes above distinguish the analysis content identity recorded
inside the JSON from the local JSON file hash. The participant-bearing lock,
target manifest, and prediction files remain in the access-controlled external
workspace.
