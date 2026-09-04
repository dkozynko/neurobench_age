from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neurobench_age.core.predictions import (
    PredictionEvidenceError,
    aggregate_subject_predictions,
    compute_regression_metrics,
    write_prediction_jsonl,
)


def test_prediction_serializer_writes_one_valid_row_per_subject(tmp_path: Path) -> None:
    path = tmp_path / "predictions" / "validation.jsonl"

    rows = write_prediction_jsonl(
        path,
        [
            {"subject_id": "s-2", "true_age": 12.0, "predicted_age": 11.5, "split": "validation", "seed": 33},
            {"subject_id": "s-1", "true_age": 10.0, "predicted_age": 10.25, "split": "validation", "seed": 33},
        ],
    )

    assert rows == 2
    persisted = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["subject_id"] for row in persisted] == ["s-1", "s-2"]
    assert all(row["schema_version"] == "1.0" for row in persisted)


@pytest.mark.parametrize(
    "bad_rows",
    [
        [
            {"subject_id": "s-1", "true_age": 10.0, "predicted_age": 10.0, "split": "validation", "seed": 33},
            {"subject_id": "s-1", "true_age": 10.0, "predicted_age": 10.0, "split": "validation", "seed": 33},
        ],
        [{"subject_id": "s-1", "true_age": np.nan, "predicted_age": 10.0, "split": "validation", "seed": 33}],
        [{"subject_id": "s-1", "true_age": 10.0, "predicted_age": 10.0, "split": "test", "seed": 33}],
    ],
)
def test_prediction_serializer_rejects_duplicate_nonfinite_or_wrong_split(
    tmp_path: Path, bad_rows: list[dict[str, object]]
) -> None:
    with pytest.raises(PredictionEvidenceError):
        write_prediction_jsonl(tmp_path / "predictions.jsonl", bad_rows, expected_split="validation")


def test_view_aggregation_is_arithmetic_mean_and_requires_matching_subject_truth(
) -> None:
    rows = aggregate_subject_predictions(
        subject_ids=["s-2", "s-1", "s-2"],
        true_ages=[20.0, 10.0, 20.0],
        predictions=[21.0, 9.0, 19.0],
        split="validation",
        seed=33,
    )

    assert [(row["subject_id"], row["predicted_age"]) for row in rows] == [
        ("s-1", 9.0),
        ("s-2", 20.0),
    ]
    assert rows[1]["aggregation"] == "arithmetic_mean_per_subject"
    assert rows[1]["view_count"] == 2


def test_view_aggregation_rejects_inconsistent_truth_and_metrics_are_subject_level() -> None:
    with pytest.raises(PredictionEvidenceError, match="true age"):
        aggregate_subject_predictions(
            subject_ids=["s-1", "s-1"],
            true_ages=[10.0, 11.0],
            predictions=[10.0, 10.0],
            split="validation",
            seed=33,
        )

    metrics = compute_regression_metrics([10.0, 20.0, 30.0], [11.0, 19.0, 32.0])
    assert metrics["mae"] == pytest.approx(4.0 / 3.0)
    assert metrics["rmse"] == pytest.approx((6.0 / 3.0) ** 0.5)
    assert metrics["pearsonr"] > 0.99
    assert metrics["r2"] > 0.96
