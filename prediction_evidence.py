"""Validated subject-level prediction evidence for paper analyses."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "1.0"


class PredictionEvidenceError(ValueError):
    """Raised when prediction evidence cannot be audited safely."""


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise PredictionEvidenceError(f"{field} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise PredictionEvidenceError(f"{field} must be a finite number")
    return value


def _normalise_row(row: Mapping[str, Any], *, expected_split: str | None) -> dict[str, Any]:
    subject_id = row.get("subject_id")
    if not isinstance(subject_id, str) or not subject_id.strip():
        raise PredictionEvidenceError("subject_id must be a non-empty string")
    split = row.get("split")
    if not isinstance(split, str) or not split.strip():
        raise PredictionEvidenceError("split must be a non-empty string")
    if expected_split is not None and split != expected_split:
        raise PredictionEvidenceError(f"split {split!r} does not match expected split {expected_split!r}")
    seed = row.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PredictionEvidenceError("seed must be an integer")
    normalised = dict(row)
    normalised.update(
        {
            "schema_version": SCHEMA_VERSION,
            "subject_id": subject_id,
            "true_age": _finite_number(row.get("true_age"), field="true_age"),
            "predicted_age": _finite_number(row.get("predicted_age"), field="predicted_age"),
            "split": split,
            "seed": seed,
        }
    )
    return normalised


def validate_prediction_rows(
    rows: Iterable[Mapping[str, Any]], *, expected_split: str | None = None
) -> list[dict[str, Any]]:
    """Validate and sort one-row-per-subject prediction records."""

    normalised = [_normalise_row(row, expected_split=expected_split) for row in rows]
    subject_ids = [row["subject_id"] for row in normalised]
    if len(set(subject_ids)) != len(subject_ids):
        raise PredictionEvidenceError("prediction rows contain duplicate subject_id values")
    return sorted(normalised, key=lambda row: row["subject_id"])


def write_prediction_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_split: str | None = None,
) -> int:
    """Write deterministic JSONL and return its subject row count."""

    validated = validate_prediction_rows(rows, expected_split=expected_split)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in validated
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return len(validated)


def aggregate_subject_predictions(
    subject_ids: Sequence[Any],
    true_ages: Sequence[Any],
    predictions: Sequence[Any],
    *,
    split: str,
    seed: int,
    view_field: str = "view_count",
) -> list[dict[str, Any]]:
    """Aggregate recording/view predictions to one arithmetic-mean row per subject."""

    if not (len(subject_ids) == len(true_ages) == len(predictions)):
        raise PredictionEvidenceError("subject_ids, true_ages, and predictions must have equal lengths")
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for raw_subject, raw_true, raw_prediction in zip(subject_ids, true_ages, predictions):
        if not isinstance(raw_subject, str) or not raw_subject.strip():
            raise PredictionEvidenceError("subject_id must be a non-empty string")
        true_age = _finite_number(raw_true, field="true_age")
        prediction = _finite_number(raw_prediction, field="predicted_age")
        grouped[raw_subject].append((true_age, prediction))

    rows: list[dict[str, Any]] = []
    for subject_id, values in grouped.items():
        truths = {true for true, _ in values}
        if len(truths) != 1:
            raise PredictionEvidenceError(f"subject {subject_id!r} has inconsistent true age across views")
        rows.append(
            {
                "subject_id": subject_id,
                "true_age": next(iter(truths)),
                "predicted_age": sum(prediction for _, prediction in values) / len(values),
                "split": split,
                "seed": seed,
                "aggregation": "arithmetic_mean_per_subject",
                view_field: len(values),
            }
        )
    return validate_prediction_rows(rows, expected_split=split)


def read_prediction_jsonl(path: Path, *, expected_split: str | None = None) -> list[dict[str, Any]]:
    """Read and validate a prediction artifact."""

    if not path.is_file():
        raise PredictionEvidenceError(f"prediction file is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise PredictionEvidenceError(f"invalid prediction JSON at line {line_number}") from error
        if not isinstance(row, Mapping):
            raise PredictionEvidenceError(f"prediction row at line {line_number} is not an object")
        rows.append(dict(row))
    return validate_prediction_rows(rows, expected_split=expected_split)


def compute_regression_metrics(
    true_ages: Sequence[Any], predictions: Sequence[Any]
) -> dict[str, float | int]:
    """Compute subject-level Pearson, MAE, RMSE, and R²."""

    if len(true_ages) != len(predictions) or len(true_ages) < 2:
        raise PredictionEvidenceError("metrics require equal arrays with at least two subjects")
    truth = np.asarray([_finite_number(value, field="true_age") for value in true_ages], dtype=np.float64)
    prediction = np.asarray([_finite_number(value, field="predicted_age") for value in predictions], dtype=np.float64)
    truth_centered = truth - truth.mean()
    prediction_centered = prediction - prediction.mean()
    denominator = float(np.linalg.norm(truth_centered) * np.linalg.norm(prediction_centered))
    if denominator <= 1e-12:
        raise PredictionEvidenceError("Pearson/R² is undefined for a constant target or prediction")
    total_ss = float(np.square(truth_centered).sum())
    residual = prediction - truth
    metrics = {
        "n_subjects": int(len(truth)),
        "pearsonr": float(np.dot(truth_centered, prediction_centered) / denominator),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "r2": float(1.0 - np.square(residual).sum() / total_ss),
    }
    if not all(math.isfinite(float(metrics[key])) for key in ("pearsonr", "mae", "rmse", "r2")):
        raise PredictionEvidenceError("regression metrics are non-finite")
    return metrics
