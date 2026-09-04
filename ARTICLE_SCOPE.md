# Article scope

This repository is the reproducible research package for the paper:

> **Stability of age probing in REVE and the limits of increasingly complex
> representation heads.**

## Research question

When the REVE encoder and the official NeuralBench Age protocol are held fixed,
how stable is age probing across random seeds, and do more expressive
representation heads improve over the matched `mean_linear` baseline reliably
enough to support an article-level claim?

## What is canonical

- the official REVE/NeuralBench Age protocol and its independent reproduction;
- the canonical 1000-subject nested manifest;
- validation-only screening, three-seed confirmation, and one sealed finalist
  test;
- matched controls, explicit complexity accounting, subject-level predictions,
  and provenance-rich evidence;
- compact metrics, tables, figures, and negative results needed to support the
  paper's conclusions.

## What is not part of the article package

Raw HBN data, pretrained weights, checkpoints, large prediction dumps, launch
logs, dated retry scripts, and obsolete exploratory run directories are kept
outside Git. Their compact hashes or summaries remain only when they support a
claim in the paper.

## Canonical claim boundary

The primary claim is about stability and limits, not about a single best score.
A candidate is not promoted because of one favorable seed or one screening
metric. Test data are opened only for a predeclared finalist after validation
confirmation, and a non-uniform or statistically weak improvement is reported
as inconclusive.

See [`docs/research/article_ready_protocol.md`](docs/research/article_ready_protocol.md)
for the execution contract and
[`docs/research/article_evidence_registry.md`](docs/research/article_evidence_registry.md)
for the mapping from claims to retained evidence.
