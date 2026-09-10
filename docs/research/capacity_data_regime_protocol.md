# Capacity–Data Regime Extension Protocol

This document describes a secondary, mechanistic extension of the frozen REVE
age-probing study. It is not a replacement for the primary result and does
not change the primary locks, registries, generated assets, or claims.

## Scope

The primary study remains bounded to its sealed 1,000-subject HBN evidence.
This extension uses the same frozen representation and training protocol, with
nested HBN training cohorts of 200, 400, and 800 subjects. The validation set
is fixed across all sizes. The currently executable matrix is:

- three heads: `mean_linear`, `mean_rich_stats_residual`, and
  `mean_mlp_residual_matched(hidden_dim=4)`;
- ten predeclared seeds, 33 through 42;
- 90 training runs in total;
- secondary evaluation on the sealed 75-subject MIPDB primary cohort;
- 6,750 subject-level external predictions.

The extension does not claim performance on the full HBN dataset. A future
`all_available` endpoint is declared as a planning boundary only and is not
part of the current executable matrix.

## Execution order and gates

1. Validate the pinned representation cache, nested cohorts, training protocol,
   free-space/RAM preflight, and one-run pilot.
2. Train the exact 90-run matrix in a separate output root. Each run retains
   its cache window count, validation history, selected epoch, subject-level
   validation predictions, head complexity metadata, resource placement, and
   checkpoint hashes.
3. Create the immutable `checkpoint_sealed` lock. The secondary evaluator
   cannot start from a draft or fabricated final lock.
4. Re-read the primary study lock and primary prediction inventory immediately
   before external materialization and abort if either changed.
5. Evaluate only the declared MIPDB primary subjects. Extrapolation subjects
   and participant-level demographic release are outside this protocol.
6. Create the final lock only after the complete prediction inventory is
   present and hash-valid.
7. Run analysis and render assets only from the final lock and its inventory.

The storage preflight reserves the larger of a 12 GiB fixed floor or 25% of the
validated representation-cache footprint, plus twice the declared extension
output estimate. This is an operational guard for the cached extension; it does
not authorize deletion of primary evidence or raw data.

## Estimand and interpretation

For candidate head `h` and training size `n`, the primary extension summary is

\[
\Delta(h,n)=\frac{1}{10}\sum_s\left[
\mathrm{Pearson}_{\mathrm{external}}(h,n,s)-
\mathrm{Pearson}_{\mathrm{external}}(\mathrm{mean\_linear},n,s)
\right].
\]

The analysis reports the endpoint contrast 800 minus 200 and the two adjacent
contrasts 400 minus 200 and 800 minus 400 for each candidate head. The joint
bootstrap samples the ten seeds and 75 external subjects once per iteration
and reuses those draws across every cell and contrast. It uses 10,000
iterations, seed `20260909`, linear-interpolation percentile intervals, and a
minimum of 9,950 valid replicates.

These results are exploratory evidence about representation-head capacity and
data regime. They do not establish causality, universal scaling laws, or
superiority outside the declared protocol.

## Exploratory inference

The extension reports an exact seed-level sign-flip p-value for each of the six
head-by-training-size cells and for each of the six predeclared training-size
contrasts. The sign assignments enumerate all $2^{10}$ possibilities for the
ten paired seed deltas. Zero deltas are retained as nonnegative exceedances.
Holm step-down adjustment is applied separately within the cell family and the
contrast family, using the fixed order recorded in
`configs/research/capacity_data_regime_exploratory_inference.json`.

These p-values are descriptive exploratory inference. They do not expand the
sealed primary hypothesis family, change the primary stable-improvement rule,
or justify a superiority/equivalence claim. The joint seed--subject bootstrap
intervals and the primary predeclared decision boundary remain authoritative
for interpretation. The aggregate asset manifest records the analysis hash,
inference-specification hash, lock hashes, and file hashes so that regenerated
tables cannot silently use a different family or ordering.

## Data boundary

Only aggregate derived evidence is intended for publication. No participant-
level MIPDB metadata or unapproved demographics supplement is released by
this extension.
