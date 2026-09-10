# Capacity--data regime aggregate evidence

This directory contains aggregate-only outputs from the secondary
capacity--data extension. The sealed extension identity remains
`reve_age_capacity_data_regime_v1`; `v3` identifies the runtime output
instance.

The evidence covers 90 frozen-head training runs: three nested HBN training
sizes (200, 400, and 800 subjects), three probe heads, and ten seeds (33--42).
Each selected checkpoint was evaluated remotely on the sealed 75-subject MIPDB
primary cohort, producing 6,750 subject-level predictions that are not stored
in this repository.

The CSV and LaTeX files contain only aggregate metrics, seed-level deltas, and
confidence intervals. The PDF files are deterministic publication figures.
`capacity_data_regime_asset_manifest.json` binds every generated output file to
its byte size and SHA-256 digest, as well as to the finalized lock, checkpoint
inventory, prediction inventory, and exploratory inference specification.
Participant identifiers, ages, raw EEG, representations, checkpoints, and
prediction rows are excluded.
