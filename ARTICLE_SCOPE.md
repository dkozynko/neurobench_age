# Research scope

## Research question

When frozen REVE representations, HBN development data, preprocessing,
checkpoint selection, and optimization seeds are held fixed, do increasingly
expressive age-prediction heads improve over `mean_linear` consistently and
transfer to an external developmental EEG cohort?

The scope is the stability and limits of increasingly expressive heads under a
fixed protocol, not the pursuit of one favorable score.

## Evidence taxonomy

The primary prospective study is frozen representation probing with HBN
train/validation development and one sealed MIPDB evaluation. MIPDB predictions
are not available for tuning, promotion, recalibration, or exclusion decisions.

Official NeuralBench full fine-tuning is secondary reproduction evidence. It is
end-to-end age prediction because the encoder is trainable.

All existing HBN R5 metrics are retrospective secondary evidence. R5 has been
opened during repeated finalist decisions, so those results must not be used for
model or head selection and cannot be represented as untouched confirmation.

## Included material

- executable protocol, immutable study lifecycle, and exact comparison rules;
- model-free content-addressed manifests and deterministic cohort assignment;
- model-free preprocessing and QC contracts;
- frozen representation extraction and cache-integrity checks;
- head-only training, external inference, and confirmatory statistics;
- compact retrospective HBN evidence with explicit provenance;
- tests and operational documentation needed to reproduce each claim.

## Excluded material

Raw EEG, pretrained weights, checkpoints, representation caches, prediction
dumps, launch logs, credentials, and obsolete exploratory run directories stay
outside Git. Only compact hashes, summaries, and audit records belong in the
repository.

## Claim boundary

The intended inference is limited to the four declared heads, REVE checkpoint,
HBN development split, MIPDB cohorts, seeds 33 through 42, and the sealed
protocol. A favorable single seed is insufficient. A null result means that no
tested complex head established a stable external gain under the predeclared
protocol; it does not establish equivalence or rule out gains from untested
representations, heads, datasets, or training regimes.

The layer-wise extension is a secondary exploratory comparison of matched
linear probes at layers -4, -3, and -2 against the final-layer baseline. It
uses the same sealed external cohort and seeds, and its aggregate results do
not modify the primary confirmatory estimand or decision rule.

The ds006780 extension is a separate secondary transfer evaluation of two
final-layer heads at HBN training sizes 200 and 800. It uses 126 eligible
subjects from a different public EEG cohort, but was motivated by earlier
results and therefore is not a new blinded confirmatory test. Its positive
`n=800` seed pattern is uncertainty-limited because the predeclared precision
gate failed; it does not establish cross-cohort superiority or an
encoder-unseen guarantee.

See [`docs/research/article_ready_protocol.md`](docs/research/article_ready_protocol.md)
for the execution contract and
[`docs/research/article_evidence_registry.md`](docs/research/article_evidence_registry.md)
for the claim-to-artifact mapping.
