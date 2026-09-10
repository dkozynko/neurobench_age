from __future__ import annotations

import numpy as np

from neurobench_age.analysis.ds006780_external import analyze_external_series
from neurobench_age.analysis.confirmatory import PredictionSeries


def _series(head: str, size: int, seed: int) -> PredictionSeries:
    subjects = tuple(f"sub-{index:02d}" for index in range(8))
    targets = tuple(float(index) for index in range(8))
    baseline = np.asarray(targets, dtype=np.float64)
    if head == "mean_linear":
        predictions = baseline + 0.25 * np.sin(np.arange(8) + seed)
    else:
        predictions = baseline + 0.5 + 0.5 * np.sin(np.arange(8) + seed)
    return PredictionSeries(
        seed=seed,
        subject_ids=subjects,
        targets=targets,
        predictions=tuple(float(value) for value in predictions),
        provenance={"source": "synthetic"},
    )


def test_external_analysis_declares_primary_and_secondary_size_contrasts() -> None:
    series = {
        (size, head, seed): _series(head, size, seed)
        for size in (200, 800)
        for head in ("mean_linear", "mean_rich_stats_residual")
        for seed in range(33, 43)
    }

    result = analyze_external_series(
        series,
        lock_sha256="a" * 64,
        prediction_inventory_sha256="b" * 64,
        bootstrap_iterations=100,
        bootstrap_seed=20260910,
    )

    assert result["primary_contrast"]["training_size"] == 800
    assert result["primary_contrast"]["candidate_head"] == "mean_rich_stats_residual"
    assert result["secondary_contrasts"][0]["training_size"] == 200
    assert result["primary_contrast"]["bootstrap"]["iterations"] == 100
    assert len(result["cell_summaries"]) == 4


def test_external_analysis_rejects_incomplete_matrix() -> None:
    series = {
        (800, head, seed): _series(head, 800, seed)
        for head in ("mean_linear", "mean_rich_stats_residual")
        for seed in range(33, 43)
    }

    try:
        analyze_external_series(
            series,
            lock_sha256="a" * 64,
            prediction_inventory_sha256="b" * 64,
            bootstrap_iterations=10,
            bootstrap_seed=20260910,
        )
    except ValueError as error:
        assert "exact external matrix" in str(error)
    else:  # pragma: no cover - assertion is the test oracle
        raise AssertionError("incomplete matrix was accepted")
