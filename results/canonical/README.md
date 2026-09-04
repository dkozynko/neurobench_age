# Canonical article evidence

This directory contains only compact evidence used by the paper. Start with
[`index.json`](index.json), then follow the claim-to-result mapping in
[`docs/research/article_evidence_registry.md`](../../docs/research/article_evidence_registry.md).

The records preserve per-seed metrics, protocol metadata, hashes, and decisions.
Raw checkpoints, HBN data, window-level predictions, and launch logs are stored
outside Git. `legacy_screening/` contains compact negative or historical metrics
only when they support the paper's limits-of-complexity claim.
