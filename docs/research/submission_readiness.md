# Submission readiness checklist

## Current scientific status

- [x] Primary frozen-probe comparison is complete: four heads, ten seeds, and
  75 sealed MIPDB subjects.
- [x] Capacity--data and layer-wise extensions are labelled secondary or
  exploratory and do not change the primary decision.
- [x] The ds006780 v5 transfer is integrated as a separate secondary cohort
  analysis.
- [x] The v5 n=800 rich-head point estimate is reported with the full
  subject-aware interval and the failed precision gate.
- [x] The manuscript does not claim equivalence, universal head superiority,
  encoder-unseen status, or an official NeuralBench MIPDB score.

## Reproducibility status

- [x] Aggregate v5 analysis, precision gate, table, figure, macros, and asset
  manifest are retained under
  `results/extensions/ds006780_external_v5/`.
- [x] Participant-level predictions, target-bearing manifests, checkpoints,
  raw EEG, credentials, and private paths remain outside Git.
- [x] Asset generation validates JSON self-hashes and provenance before
  rendering.
- [x] Full Python test suite passes and the repository's LaTeX verification
  target builds the
  manuscript without undefined references.
- [x] The PDF was rendered and visually checked for clipped tables, figures,
  unreadable labels, and broken page transitions.

## Required before an actual submission

These items require author decisions or an independent human review; they
cannot be inferred from the repository:

- [ ] Replace `Anonymous` with the final author list and affiliations, if the
  venue is not double-blind at the relevant stage.
- [ ] Add the venue-specific ethics/data-access statement, including the exact
  dataset citations, access conditions, and any institutional approval wording
  required by the venue or dataset terms.
- [ ] Have at least one independent reader check the methods, numerical claims,
  statistical interpretation, and privacy boundary.
- [ ] Select a venue and apply its official format, page limit, anonymization,
  supplementary-material, and code/data-availability rules.
- [ ] Decide whether to submit the current bounded empirical paper or first run
  a new prospectively powered external cohort. A new encoder is not required
  for the latter; the scientific bottleneck is external precision and
  independence, not model novelty.

## Claims that should remain in the final version

The defensible headline is that richer frozen-representation heads can change
the point estimate across data regimes, but repeated seed direction alone does
not establish a stable external gain when subject-level uncertainty is
propagated. The strongest current result is therefore methodological guidance
about specifying the head, training size, metric, cohort, and uncertainty—not a
universal ranking of REVE heads or a new encoder contribution.

## Final verification commands

```bash
MPLCONFIGDIR=/private/tmp/mplconfig MNE_DONTWRITE_HOME=true \
  .venv/bin/python -m pytest -q
UV_CACHE_DIR=/private/tmp/neurobench-uv-cache \
  MPLCONFIGDIR=/private/tmp/mplconfig MNE_DONTWRITE_HOME=true \
  make --directory=manuscript verify
```

No Git commit is created by the research workflow; the final diff remains for
manual author review.
