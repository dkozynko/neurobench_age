# REVE age experiment protocol

This protocol belongs to the package described in
[`ARTICLE_SCOPE.md`](../../ARTICLE_SCOPE.md). It governs the paper's claims
about seed stability and the limits of complex representation heads.

This protocol is the default for every new claim about improving the
NeuralBench Age `mean_linear` reference.

## Comparison unit

Each candidate is compared with a matched `mean_linear` control using the same
canonical manifest, subject split, REVE checkpoint, channel positions,
preprocessing, optimizer, scheduler, epoch budget, checkpoint-selection rule,
and seed. The canonical 1000-subject nested manifest is the primary screening
regime; the historical 500-subject manifest is screening evidence only.

## Staged evaluation

1. Run a validation-only seed-33 screen. The test loader must not be opened.
2. Promote a candidate only when its predeclared gate is passed against the
   matched control.
3. Confirm the frozen candidate and control on seeds 34 and 35, still without
   test access.
4. Run one sealed final test only for the frozen finalist, with explicit
   `--evaluation-mode final_test --allow-sealed-test-evaluation`.

The validation screen is allowed to reject a candidate early. The test score is never used to choose a head, tune a hyperparameter, or rescue a failed screen.

## Required evidence

Every seed directory must contain the schema-versioned manifest, normalized
configuration, complexity accounting, validation history, immutable selection
record, train-only age reference, optimizer metadata, throughput metadata, and
subject-level validation predictions. A sealed finalist additionally contains
the one-time test markers and subject-level test predictions.

Report Pearson together with MAE, RMSE, and R². Report per-seed values, sample
standard deviation, candidate wins, worst-seed delta, and paired subject-level
bootstrap confidence intervals. Age-group thresholds are derived from unique
training subjects only and reused unchanged for validation and test.

Keep the two metric units explicit. `selection.json` and the official
`test_completed.json` marker contain the native NeuralBench callback metric;
the exported `predictions/*.jsonl` files contain subject-level arithmetic
means. These metrics can legitimately differ when the official loader evaluates
multiple windows per subject. Use the official selection/test marker for the
NeuralBench comparison, and use the subject-level export for age-group plots,
residuals, and paired subject bootstrap. The audit must reconcile an export
against its own `prediction_export.metrics`, but must not require it to equal
the native official marker.

## Compute and hardware

Record the exact accelerator model, VRAM tier, peak memory, training and
validation time, throughput, trainable/frozen parameter counts, and optional
hourly cost. Runs on different accelerator models are labelled
`hardware_mixed` in confirmatory summaries; hardware differences must not be
presented as a method effect.

## Reproducible analysis

After the runs complete, generate the tables and figures with:

```bash
PYTHONPATH=src python scripts/analyze_paper_evidence.py \
  /path/to/candidate/mean_linear/seed33 \
  /path/to/candidate/mean_linear/seed34 \
  /path/to/candidate/mean_linear/seed35 \
  --output-dir /path/to/analysis
```

The analysis directory is itself hashed and records the exact input run
directories, bootstrap seed, iteration count, age-group thresholds, and all
generated tables/plots.
