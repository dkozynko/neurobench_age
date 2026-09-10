"""Confirmatory analysis for the sealed ds006780 external transfer matrix."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from neurobench_age.analysis.confirmatory import (
    PredictionSeries,
    calibration_parameters,
    exact_seed_randomization,
    hierarchical_paired_bootstrap,
    paired_seed_statistics,
    regression_metrics,
)
from neurobench_age.research.strict_json import canonical_sha256
from neurobench_age.research.strict_json import load_json_strict


class Ds006780ExternalAnalysisError(ValueError):
    """Raised when external predictions cannot support the declared analysis."""


EXTERNAL_HEADS = ("mean_linear", "mean_rich_stats_residual")
EXTERNAL_SIZES = (200, 800)
EXTERNAL_SEEDS = tuple(range(33, 43))


def _validate_hash(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise Ds006780ExternalAnalysisError(f"{field} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise Ds006780ExternalAnalysisError(f"{field} must be a SHA-256 hex digest") from error
    return value


def _validate_series_matrix(
    series: Mapping[tuple[int, str, int], PredictionSeries],
) -> tuple[tuple[str, ...], tuple[float, ...], dict[str, Any]]:
    expected_keys = {
        (size, head, seed)
        for size in EXTERNAL_SIZES
        for head in EXTERNAL_HEADS
        for seed in EXTERNAL_SEEDS
    }
    if set(series) != expected_keys:
        raise Ds006780ExternalAnalysisError("analysis requires the exact external matrix")
    reference_subjects: tuple[str, ...] | None = None
    reference_targets: tuple[float, ...] | None = None
    reference_provenance: dict[str, Any] | None = None
    for key in sorted(expected_keys):
        size, head, seed = key
        value = series[key]
        if not isinstance(value, PredictionSeries) or value.seed != seed:
            raise Ds006780ExternalAnalysisError(f"invalid prediction series for {key}")
        if (
            len(value.subject_ids) < 2
            or len(value.subject_ids) != len(value.targets)
            or len(value.targets) != len(value.predictions)
            or len(set(value.subject_ids)) != len(value.subject_ids)
        ):
            raise Ds006780ExternalAnalysisError(f"invalid prediction shape for {key}")
        if tuple(sorted(value.subject_ids, key=lambda item: item.encode("utf-8"))) != value.subject_ids:
            raise Ds006780ExternalAnalysisError(f"subject order is not canonical for {key}")
        targets = np.asarray(value.targets, dtype=np.float64)
        predictions = np.asarray(value.predictions, dtype=np.float64)
        if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
            raise Ds006780ExternalAnalysisError(f"non-finite predictions for {key}")
        provenance = dict(value.provenance)
        if reference_subjects is None:
            reference_subjects = value.subject_ids
            reference_targets = value.targets
            reference_provenance = provenance
        elif (
            value.subject_ids != reference_subjects
            or value.targets != reference_targets
            or provenance != reference_provenance
        ):
            raise Ds006780ExternalAnalysisError(
                "subject order, targets, or provenance differs across external cells"
            )
    assert reference_subjects is not None
    assert reference_targets is not None
    assert reference_provenance is not None
    return reference_subjects, reference_targets, reference_provenance


def _cell_summary(
    *, size: int, head: str, series: Mapping[int, PredictionSeries]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in EXTERNAL_SEEDS:
        metrics = regression_metrics(series[seed].targets, series[seed].predictions)
        calibration = calibration_parameters(
            series[seed].targets, series[seed].predictions
        )
        rows.append({"seed": seed, **metrics, "calibration": calibration})
    pearson = np.asarray([row["pearson"] for row in rows], dtype=np.float64)
    return {
        "training_size": size,
        "head": head,
        "seed_count": len(rows),
        "subject_count": len(series[EXTERNAL_SEEDS[0]].subject_ids),
        "mean_pearson": float(pearson.mean()),
        "seed_pearson_sd": float(pearson.std(ddof=1)),
        "min_pearson": float(pearson.min()),
        "max_pearson": float(pearson.max()),
        "per_seed": rows,
    }


def _contrast(
    *,
    size: int,
    series: Mapping[tuple[int, str, int], PredictionSeries],
    bootstrap_iterations: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, Any]:
    baseline = {
        seed: series[(size, "mean_linear", seed)] for seed in EXTERNAL_SEEDS
    }
    candidate = {
        seed: series[(size, "mean_rich_stats_residual", seed)]
        for seed in EXTERNAL_SEEDS
    }
    paired = paired_seed_statistics(candidate, baseline)
    bootstrap = hierarchical_paired_bootstrap(
        candidate,
        baseline,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    randomization = exact_seed_randomization(
        [row["pearson_delta"] for row in paired["per_seed"]]
    )
    return {
        "training_size": size,
        "baseline_head": "mean_linear",
        "candidate_head": "mean_rich_stats_residual",
        "estimand": "mean paired seed-level Pearson delta (candidate minus baseline)",
        "paired": paired,
        "bootstrap": bootstrap,
        "exact_seed_randomization": randomization,
    }


def analyze_external_series(
    series: Mapping[tuple[int, str, int], PredictionSeries],
    *,
    lock_sha256: str,
    prediction_inventory_sha256: str,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 2_026_0910,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Calculate the predeclared external head-complexity contrasts."""

    lock_sha256 = _validate_hash(lock_sha256, "lock_sha256")
    prediction_inventory_sha256 = _validate_hash(
        prediction_inventory_sha256, "prediction_inventory_sha256"
    )
    if (
        isinstance(bootstrap_iterations, bool)
        or not isinstance(bootstrap_iterations, int)
        or bootstrap_iterations <= 0
        or isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or not 0.0 < confidence < 1.0
    ):
        raise Ds006780ExternalAnalysisError("bootstrap specification is invalid")
    subjects, targets, provenance = _validate_series_matrix(series)
    cell_summaries = [
        _cell_summary(
            size=size,
            head=head,
            series={seed: series[(size, head, seed)] for seed in EXTERNAL_SEEDS},
        )
        for size in EXTERNAL_SIZES
        for head in EXTERNAL_HEADS
    ]
    primary = _contrast(
        size=800,
        series=series,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    secondary = _contrast(
        size=200,
        series=series,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed + 1,
        confidence=confidence,
    )
    body: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "analysis": "ds006780_external_head_complexity",
        "lock_sha256": lock_sha256,
        "prediction_inventory_sha256": prediction_inventory_sha256,
        "subject_count": len(subjects),
        "subjects_sha256": canonical_sha256(list(subjects)),
        "target_vector_sha256": canonical_sha256(list(targets)),
        "common_provenance": provenance,
        "primary_contrast": primary,
        "secondary_contrasts": [secondary],
        "cell_summaries": cell_summaries,
        "bootstrap_policy": {
            "iterations": bootstrap_iterations,
            "seed_primary": bootstrap_seed,
            "seed_secondary": bootstrap_seed + 1,
            "confidence": confidence,
            "resampling": "paired_seeds_and_subjects_with_replacement",
        },
    }
    return {**body, "analysis_sha256": canonical_sha256(body)}


def load_external_prediction_series(
    *,
    lock: Mapping[str, Any],
    inventory: Mapping[str, Any],
    output_root: Path,
) -> dict[tuple[int, str, int], PredictionSeries]:
    """Load and validate every immutable subject prediction from an inventory."""

    lock_sha256 = _validate_hash(lock.get("lock_sha256"), "lock_sha256")
    inventory_hash = inventory.get("prediction_inventory_sha256")
    if inventory_hash != canonical_sha256(
        inventory, exclude_fields=("prediction_inventory_sha256",)
    ):
        raise Ds006780ExternalAnalysisError("prediction inventory self-hash is invalid")
    inventory_hash = _validate_hash(inventory_hash, "prediction_inventory_sha256")
    if inventory.get("lock_sha256") != lock_sha256:
        raise Ds006780ExternalAnalysisError("prediction inventory lock differs from analysis lock")
    selected_runs = lock.get("selected_runs")
    subject_ids = lock.get("subject_ids")
    if not isinstance(selected_runs, list) or not isinstance(subject_ids, list):
        raise Ds006780ExternalAnalysisError("external lock inventory is invalid")
    expected_subjects = tuple(sorted((str(value) for value in subject_ids), key=lambda item: item.encode("utf-8")))
    expected_runs = {
        (int(run["training_size"]), str(run["head"]), int(run["seed"])): run
        for run in selected_runs
    }
    expected_keys = {
        (size, head, seed)
        for size in EXTERNAL_SIZES
        for head in EXTERNAL_HEADS
        for seed in EXTERNAL_SEEDS
    }
    if set(expected_runs) != expected_keys:
        raise Ds006780ExternalAnalysisError("external lock selected matrix is invalid")
    rows = inventory.get("predictions")
    if not isinstance(rows, list) or len(rows) != len(expected_subjects) * len(expected_keys):
        raise Ds006780ExternalAnalysisError("prediction inventory row count is invalid")
    grouped: dict[tuple[int, str, int], dict[str, dict[str, Any]]] = {}
    root = Path(output_root).resolve()
    for row in rows:
        if not isinstance(row, Mapping):
            raise Ds006780ExternalAnalysisError("prediction inventory row is invalid")
        try:
            key = (int(row["training_size"]), str(row["head"]), int(row["seed"]))
            subject_id = str(row["subject_id"])
            relative = Path(str(row["path"]))
        except (KeyError, TypeError, ValueError) as error:
            raise Ds006780ExternalAnalysisError("prediction inventory row identity is invalid") from error
        if key not in expected_runs or subject_id not in expected_subjects:
            raise Ds006780ExternalAnalysisError("prediction inventory contains an unexpected row")
        if relative.is_absolute():
            raise Ds006780ExternalAnalysisError("prediction inventory path must be relative")
        path = (root / relative).resolve()
        if root not in path.parents:
            raise Ds006780ExternalAnalysisError("prediction inventory path escapes output root")
        record = load_json_strict(path)
        if not isinstance(record, Mapping):
            raise Ds006780ExternalAnalysisError(f"prediction artifact is not an object: {path}")
        prediction_hash = row.get("prediction_sha256")
        if prediction_hash != record.get("prediction_sha256"):
            raise Ds006780ExternalAnalysisError(f"prediction hash differs from inventory: {path}")
        if prediction_hash != canonical_sha256(record, exclude_fields=("prediction_sha256",)):
            raise Ds006780ExternalAnalysisError(f"prediction self-hash is invalid: {path}")
        expected_identity = {
            "lock_sha256": lock_sha256,
            "training_size": key[0],
            "head": key[1],
            "seed": key[2],
            "subject_id": subject_id,
        }
        if any(record.get(field) != value for field, value in expected_identity.items()):
            raise Ds006780ExternalAnalysisError(f"prediction identity differs from inventory: {path}")
        prediction = record.get("prediction")
        target_age = record.get("target_age")
        if (
            isinstance(prediction, bool)
            or not isinstance(prediction, (int, float))
            or not math.isfinite(float(prediction))
            or isinstance(target_age, bool)
            or not isinstance(target_age, (int, float))
            or not math.isfinite(float(target_age))
            or record.get("qc_sha256") != record.get("signal_qc_sha256")
        ):
            raise Ds006780ExternalAnalysisError(f"prediction values are invalid: {path}")
        cell = grouped.setdefault(key, {})
        if subject_id in cell:
            raise Ds006780ExternalAnalysisError("duplicate subject in prediction cell")
        cell[subject_id] = dict(record)
    if set(grouped) != expected_keys or any(
        set(values) != set(expected_subjects) for values in grouped.values()
    ):
        raise Ds006780ExternalAnalysisError("prediction inventory cell matrix is incomplete")
    series: dict[tuple[int, str, int], PredictionSeries] = {}
    provenance = {
        "study_id": lock["study_id"],
        "lock_sha256": lock_sha256,
        "external_config_sha256": lock["external_config_sha256"],
        "target_free_manifest_sha256": lock["target_free_manifest_sha256"],
        "target_manifest_sha256": lock["target_manifest_sha256"],
        "encoder_checkpoint_sha256": lock["encoder_checkpoint_sha256"],
    }
    reference_targets: tuple[float, ...] | None = None
    for key in sorted(expected_keys):
        values = grouped[key]
        ordered = [values[subject_id] for subject_id in expected_subjects]
        targets = tuple(float(record["target_age"]) for record in ordered)
        if reference_targets is None:
            reference_targets = targets
        elif targets != reference_targets:
            raise Ds006780ExternalAnalysisError("target ages differ across prediction cells")
        series[key] = PredictionSeries(
            seed=key[2],
            subject_ids=expected_subjects,
            targets=targets,
            predictions=tuple(float(record["prediction"]) for record in ordered),
            provenance=provenance,
        )
    return series
