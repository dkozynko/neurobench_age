# Submission readiness checklist

Updated September 20, 2026. Editorial revisions and technical packaging are
complete. The HBN counts and selected ds006780 sample descriptions have now
been recovered with identity checks. Final author review and the remaining
limitations below still apply.

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
- [x] This revision adds only aggregate cohort-recovery information. The
  repository already contains the historical HBN subject/split manifest;
  participant-level prediction files and execution caches were not recovered.
- [x] Asset generation validates JSON self-hashes and provenance before
  rendering.
- [x] Full Python test suite: **813 passed** in 191.65 seconds.
- [x] After cohort recovery, all **29** affected manuscript, documentation and
  release tests passed; the 18-page PDF and clean archive build were rechecked.
- [x] The layer-wise article audit passes; retained comparisons and provenance
  remain consistent with the manuscript.
- [x] The LaTeX verification target builds the manuscript without undefined
  references, missing font shapes, or overfull boxes.
- [x] The PDF was rendered and visually checked for clipped tables, figures,
  unreadable labels, and broken page transitions.
- [x] All 61 files in the retained `results/` bundles match their pre-edit hashes.
- [x] The arXiv archive contains 39 source dependencies. A clean extraction
  builds with `pdflatex`, `bibtex`, and two further `pdflatex` passes; the text
  on all 18 pages matches the local PDF.

## Editorial changes completed

- [x] Rewritten title, abstract and section prose; shorter sentences and fewer
  repeated qualifications. The PDF decreased from 24 to 18 pages.
- [x] Explained head definitions and parameter counts, window aggregation,
  calibration direction, the shared participant resample across seeds,
  sign-flip assumptions, and Holm adjustment.
- [x] Corrected the exploratory training-size interval summary and distinguished
  the fixed positive-direction tests from two-sided questions about change.
- [x] Consolidated figures, moved supporting plots to the appendix, and replaced
  implementation identifiers in presentation tables with readable labels.
- [x] Replaced machine-specific fonts with Latin Modern and formatted references
  using a standard bibliography style.
- [x] Added the public code URL and the pinned ds006780 version 1.0.0 citation
  (2025), with source ethics, consent and licensing context.
- [x] Recovered HBN counts from a source manifest whose SHA-256 matches the
  primary study, and ds006780 demographics from the pinned public metadata
  with exact participant-list and age-vector hash matches.

## Required before an actual submission

The two earlier sample-description gaps are resolved. See the
[`cohort recovery audit`](cohort_metadata_recovery_20260920.md) for the evidence:

- [x] Primary HBN source cohort: 800 training and 100 validation participants,
  one recording each, with ages and release-level split membership recovered.
- [x] Selected ds006780 sample: 126 participants, age 10.60 +/- 1.57 years,
  range 7.9-14.5, 80 recorded male and 46 female; source groups TD 39, ASD 62,
  and ASD SIBLING 25. Recording and metadata exclusions were reconstructed.

HBN sex/group fields and ds006780 family identifiers remain unavailable in
these sources. The manuscript retains the corresponding limitations. Original
execution caches and participant-level predictions remain unavailable; cohort
recovery does not recreate the full experiment archive.

The remaining items are author decisions or external review:

- [x] Author details confirmed: Dmytro Kozynko, Independent Researcher,
  dmytro.kozynko@gmail.com.
- [ ] Confirm that the data-use statement accurately covers the author's
  institutional requirements and the terms under which the data were accessed.
- [ ] Have at least one independent reader check the methods, numerical claims,
  statistical interpretation, and privacy boundary. The additional agent review
  supplied useful corrections but stopped at a usage limit; it was not a
  completed independent review and does not replace a human reader.
- [ ] Choose the arXiv category and license, verify submission metadata and any
  endorsement requirement, and authorize the actual submission.

A new external experiment is an optional scientific extension, not a condition
imposed by this editorial revision. The current paper must retain its bounded
claims and the failed ds006780 precision gate.

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
.venv/bin/python scripts/audit_layerwise_article.py
MPLCONFIGDIR=/private/tmp/mplconfig MNE_DONTWRITE_HOME=true \
  make --directory=manuscript arxiv PYTHON=.venv/bin/python
```

The final command verifies the local PDF and writes `dist/arxiv-source.zip`.
Archive portability was checked separately from an empty extraction directory.
The assistant created no Git commits and did not submit the article.
