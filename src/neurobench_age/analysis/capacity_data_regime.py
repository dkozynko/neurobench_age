"""Predeclared secondary estimands for the capacity--data extension."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from neurobench_age.research.capacity_data_regime import CapacityDataRegimeProtocol
from neurobench_age.research.capacity_data_regime_lock import (
    CapacityDataRegimeLockError,
    load_final_lock,
    load_checkpoint_inventory,
    load_prediction_inventory,
)


class CapacityDataRegimeAnalysisError(ValueError):
    """Raised when finalized extension evidence cannot support the estimands."""


BASELINE_HEAD = "mean_linear"
CANDIDATE_HEADS = (
    "mean_rich_stats_residual",
    "mean_mlp_residual_matched(hidden_dim=4)",
)
SIZES = (200, 400, 800)
SEEDS = tuple(range(33, 43))
SUBJECT_COUNT = 75


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(values: Sequence[float], field: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CapacityDataRegimeAnalysisError(f"{field} must be numeric") from error
    if array.ndim != 1 or not np.isfinite(array).all():
        raise CapacityDataRegimeAnalysisError(f"{field} must be finite and one-dimensional")
    return array


def regression_metrics(
    targets: Sequence[float], predictions: Sequence[float]
) -> dict[str, float]:
    truth = _finite(targets, "true ages")
    estimate = _finite(predictions, "predictions")
    if truth.size != estimate.size or truth.size < 2:
        raise CapacityDataRegimeAnalysisError("metrics require equal arrays with at least two subjects")
    truth_centered = truth - truth.mean()
    estimate_centered = estimate - estimate.mean()
    denominator = math.sqrt(
        float(np.square(truth_centered).sum())
        * float(np.square(estimate_centered).sum())
    )
    if denominator <= 0.0:
        raise CapacityDataRegimeAnalysisError("Pearson is undefined")
    residual = estimate - truth
    return {
        "pearson": float(np.dot(truth_centered, estimate_centered) / denominator),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
    }


def calibration_parameters(
    targets: Sequence[float], predictions: Sequence[float]
) -> dict[str, float]:
    truth = _finite(targets, "true ages")
    estimate = _finite(predictions, "predictions")
    if truth.size != estimate.size or truth.size < 2:
        raise CapacityDataRegimeAnalysisError("calibration requires equal arrays with at least two subjects")
    centered = estimate - estimate.mean()
    denominator = float(np.square(centered).sum())
    if denominator <= 0.0:
        raise CapacityDataRegimeAnalysisError("calibration is undefined")
    slope = float(np.dot(centered, truth - truth.mean()) / denominator)
    intercept = float(truth.mean() - slope * estimate.mean())
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise CapacityDataRegimeAnalysisError("calibration is non-finite")
    return {"intercept": intercept, "slope": slope}


def _validated_arrays(
    inventory: Mapping[str, Any],
) -> tuple[dict[tuple[int, str, int], np.ndarray], np.ndarray, tuple[str, ...]]:
    predictions = inventory.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 6_750:
        raise CapacityDataRegimeAnalysisError("prediction inventory must contain exactly 6,750 rows")
    by_cell: dict[tuple[int, str, int], list[Mapping[str, Any]]] = {}
    for row in predictions:
        if not isinstance(row, Mapping):
            raise CapacityDataRegimeAnalysisError("prediction inventory row is invalid")
        key = (row.get("training_size"), row.get("head"), row.get("seed"))
        if key[0] not in SIZES or key[1] not in (BASELINE_HEAD, *CANDIDATE_HEADS) or key[2] not in SEEDS:
            raise CapacityDataRegimeAnalysisError("prediction inventory contains an unexpected cell")
        if row.get("split") != "mipdb_primary" or not isinstance(row.get("subject_id"), str):
            raise CapacityDataRegimeAnalysisError("prediction row split or subject identity is invalid")
        by_cell.setdefault(key, []).append(row)
    expected_cells = {
        (size, head, seed)
        for size in SIZES
        for head in (BASELINE_HEAD, *CANDIDATE_HEADS)
        for seed in SEEDS
    }
    if set(by_cell) != expected_cells or any(len(rows) != SUBJECT_COUNT for rows in by_cell.values()):
        raise CapacityDataRegimeAnalysisError("prediction inventory cell matrix is incomplete")
    reference_subjects: tuple[str, ...] | None = None
    reference_targets: np.ndarray | None = None
    arrays: dict[tuple[int, str, int], np.ndarray] = {}
    for key, rows in by_cell.items():
        rows = sorted(rows, key=lambda row: row["subject_id"].encode("utf-8"))
        subjects = tuple(str(row["subject_id"]) for row in rows)
        targets = _finite([row["true_age"] for row in rows], "true ages")
        values = _finite([row["prediction"] for row in rows], "predictions")
        if len(set(subjects)) != SUBJECT_COUNT:
            raise CapacityDataRegimeAnalysisError("prediction inventory contains duplicate subjects")
        if reference_subjects is None:
            reference_subjects = subjects
            reference_targets = targets
        elif subjects != reference_subjects or not np.array_equal(targets, reference_targets):
            raise CapacityDataRegimeAnalysisError("subject order or true ages differ across cells")
        arrays[key] = values
    assert reference_subjects is not None and reference_targets is not None
    return arrays, reference_targets, reference_subjects


def _pearson_batch(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    target_centered = targets - targets.mean(axis=-1, keepdims=True)
    prediction_centered = predictions - predictions.mean(axis=-1, keepdims=True)
    denominator = np.sqrt(
        np.square(target_centered).sum(axis=-1)
        * np.square(prediction_centered).sum(axis=-1)
    )
    values = np.full(denominator.shape, np.nan, dtype=np.float64)
    valid = denominator > 0.0
    values[valid] = (
        (target_centered * prediction_centered).sum(axis=-1)[valid] / denominator[valid]
    )
    return values


def _bootstrap_distribution(
    candidate: np.ndarray,
    baseline: np.ndarray,
    targets: np.ndarray,
    seed_indices: np.ndarray,
    subject_indices: np.ndarray,
) -> tuple[np.ndarray, int]:
    sampled_targets = targets[subject_indices]
    sampled_candidate = candidate[seed_indices[:, :, None], subject_indices[:, None, :]]
    sampled_baseline = baseline[seed_indices[:, :, None], subject_indices[:, None, :]]
    candidate_correlations = _pearson_batch(
        np.broadcast_to(sampled_targets[:, None, :], sampled_candidate.shape),
        sampled_candidate,
    )
    baseline_correlations = _pearson_batch(
        np.broadcast_to(sampled_targets[:, None, :], sampled_baseline.shape),
        sampled_baseline,
    )
    deltas = np.full(candidate_correlations.shape[0], np.nan, dtype=np.float64)
    valid = (
        np.isfinite(candidate_correlations).all(axis=1)
        & np.isfinite(baseline_correlations).all(axis=1)
    )
    deltas[valid] = np.mean(
        candidate_correlations[valid] - baseline_correlations[valid], axis=1
    )
    return deltas, int((~valid).sum())


def _interval(values: np.ndarray, *, confidence: float) -> tuple[float, float]:
    alpha = 1.0 - confidence
    return (
        float(np.quantile(values, alpha / 2.0, method="linear")),
        float(np.quantile(values, 1.0 - alpha / 2.0, method="linear")),
    )


def analyze_capacity_data_regime(
    *,
    final_lock: Mapping[str, Any],
    prediction_inventory: Mapping[str, Any],
    checkpoint_inventory: Mapping[str, Any] | None = None,
    protocol: CapacityDataRegimeProtocol | None = None,
) -> dict[str, Any]:
    """Validate final evidence and calculate the six predeclared contrasts."""

    try:
        validated_lock = load_final_lock(final_lock)
        validated_inventory = load_prediction_inventory(
            prediction_inventory,
            expected_core_digest=validated_lock["lock_core_sha256"],
            expected_checkpoint_digest=validated_lock["checkpoint_inventory_body_sha256"],
        )
    except CapacityDataRegimeLockError as error:
        raise CapacityDataRegimeAnalysisError(str(error)) from error
    if validated_inventory["prediction_inventory_body_sha256"] != validated_lock[
        "prediction_inventory_body_sha256"
    ]:
        raise CapacityDataRegimeAnalysisError("prediction inventory does not match final lock")
    training_evidence: dict[str, Any] | None = None
    if checkpoint_inventory is not None:
        try:
            validated_checkpoint_inventory = load_checkpoint_inventory(
                checkpoint_inventory,
                expected_core_digest=validated_lock["lock_core_sha256"],
            )
        except CapacityDataRegimeLockError as error:
            raise CapacityDataRegimeAnalysisError(str(error)) from error
        if validated_checkpoint_inventory["checkpoint_inventory_body_sha256"] != validated_lock[
            "checkpoint_inventory_body_sha256"
        ]:
            raise CapacityDataRegimeAnalysisError(
                "checkpoint inventory does not match final lock"
            )
        training_evidence = {
            "checkpoint_inventory_body_sha256": validated_checkpoint_inventory[
                "checkpoint_inventory_body_sha256"
            ],
            "run_count": validated_checkpoint_inventory["run_count"],
            "runs": validated_checkpoint_inventory["runs"],
        }
    arrays, targets, subject_ids = _validated_arrays(validated_inventory)
    iterations = 10_000 if protocol is None else protocol.bootstrap_iterations
    bootstrap_seed = 20260909 if protocol is None else protocol.bootstrap_seed
    confidence = 0.95 if protocol is None else protocol.bootstrap_confidence
    minimum_valid = 9_950 if protocol is None else protocol.minimum_valid_bootstrap_replicates
    rng = np.random.default_rng(bootstrap_seed)
    sampled_seed_indices = rng.integers(0, len(SEEDS), size=(iterations, len(SEEDS)))
    sampled_subject_indices = rng.integers(0, len(subject_ids), size=(iterations, len(subject_ids)))
    cell_results: list[dict[str, Any]] = []
    delta_by_head_size: dict[tuple[str, int], float] = {}
    bootstrap_by_head_size: dict[tuple[str, int], np.ndarray] = {}
    undefined_by_head_size: dict[tuple[str, int], int] = {}
    for head in CANDIDATE_HEADS:
        for size in SIZES:
            per_seed: list[dict[str, Any]] = []
            candidate_values: list[float] = []
            baseline_values: list[float] = []
            deltas: list[float] = []
            for seed in SEEDS:
                candidate_predictions = arrays[(size, head, seed)]
                baseline_predictions = arrays[(size, BASELINE_HEAD, seed)]
                candidate_metrics = regression_metrics(targets, candidate_predictions)
                baseline_metrics = regression_metrics(targets, baseline_predictions)
                candidate_calibration = calibration_parameters(targets, candidate_predictions)
                baseline_calibration = calibration_parameters(targets, baseline_predictions)
                delta = candidate_metrics["pearson"] - baseline_metrics["pearson"]
                candidate_values.append(candidate_metrics["pearson"])
                baseline_values.append(baseline_metrics["pearson"])
                deltas.append(delta)
                per_seed.append(
                    {
                        "seed": seed,
                        "candidate": {**candidate_metrics, "calibration": candidate_calibration},
                        "baseline": {**baseline_metrics, "calibration": baseline_calibration},
                        "pearson_delta": float(delta),
                    }
                )
            observed = float(np.mean(deltas))
            bootstrap_values, undefined = _bootstrap_distribution(
                np.stack([arrays[(size, head, seed)] for seed in SEEDS]),
                np.stack([arrays[(size, BASELINE_HEAD, seed)] for seed in SEEDS]),
                targets,
                sampled_seed_indices,
                sampled_subject_indices,
            )
            valid_bootstrap = np.isfinite(bootstrap_values)
            if int(valid_bootstrap.sum()) < minimum_valid:
                raise CapacityDataRegimeAnalysisError(
                    f"bootstrap valid replicate count is below threshold for {head}, n={size}"
                )
            delta_by_head_size[(head, size)] = observed
            bootstrap_by_head_size[(head, size)] = bootstrap_values
            undefined_by_head_size[(head, size)] = undefined
            cell_results.append(
                {
                    "training_size": size,
                    "head": head,
                    "baseline_head": BASELINE_HEAD,
                    "per_seed": per_seed,
                    "mean_candidate_pearson": float(np.mean(candidate_values)),
                    "mean_baseline_pearson": float(np.mean(baseline_values)),
                    "mean_delta": observed,
                    "wins": int(np.count_nonzero(np.asarray(deltas) > 0.0)),
                    "ties": int(np.count_nonzero(np.asarray(deltas) == 0.0)),
                    "losses": int(np.count_nonzero(np.asarray(deltas) < 0.0)),
                    "worst_seed_delta": float(np.min(deltas)),
                    "bootstrap": {
                        "iterations": iterations,
                        "valid_iterations": int(valid_bootstrap.sum()),
                        "undefined_iterations": undefined,
                        "seed": bootstrap_seed,
                        "confidence": confidence,
                        "ci_low": _interval(bootstrap_values[valid_bootstrap], confidence=confidence)[0],
                        "ci_high": _interval(bootstrap_values[valid_bootstrap], confidence=confidence)[1],
                        "percentile_method": "linear_interpolation",
                        "shared_seed_and_subject_draws": True,
                    },
                }
            )

    contrasts: list[dict[str, Any]] = []
    for head in CANDIDATE_HEADS:
        contrast_specs = (
            ("endpoint_800_minus_200", 800, 200),
            ("adjacent_400_minus_200", 400, 200),
            ("adjacent_800_minus_400", 800, 400),
        )
        for name, high, low in contrast_specs:
            observed = delta_by_head_size[(head, high)] - delta_by_head_size[(head, low)]
            bootstrap_values = bootstrap_by_head_size[(head, high)] - bootstrap_by_head_size[(head, low)]
            valid = np.isfinite(bootstrap_values)
            bootstrap_values = bootstrap_values[valid]
            undefined = int((~valid).sum())
            if int(valid.sum()) < minimum_valid:
                raise CapacityDataRegimeAnalysisError(
                    f"bootstrap valid replicate count is below threshold for contrast {head}, {name}"
                )
            low_ci, high_ci = _interval(bootstrap_values, confidence=confidence)
            contrasts.append(
                {
                    "head": head,
                    "name": name,
                    "estimand": "Delta(h,n_high) - Delta(h,n_low)",
                    "training_size_high": high,
                    "training_size_low": low,
                    "observed": float(observed),
                    "bootstrap": {
                        "iterations": iterations,
                        "valid_iterations": int(valid.sum()),
                        "undefined_iterations": undefined,
                        "seed": bootstrap_seed,
                        "confidence": confidence,
                        "ci_low": low_ci,
                        "ci_high": high_ci,
                        "percentile_method": "linear_interpolation",
                        "shared_seed_and_subject_draws": True,
                    },
                }
            )
    output_body = {
        "schema_version": 1,
        "status": "complete",
        "final_lock_sha256": validated_lock["lock_sha256"],
        "prediction_inventory_body_sha256": validated_inventory[
            "prediction_inventory_body_sha256"
        ],
        "provenance": {
            "extension_protocol_sha256": validated_lock["lock_core"][
                "extension_protocol_sha256"
            ],
            "representation_protocol_sha256": validated_lock["lock_core"][
                "representation_protocol_sha256"
            ],
            "training_protocol_sha256": validated_lock["lock_core"][
                "training_protocol_sha256"
            ],
            "training_source_sha256": validated_lock["lock_core"][
                "training_source_sha256"
            ],
            "environment_sha256": validated_lock["lock_core"]["environment_sha256"],
            "hardware_sha256": validated_lock["lock_core"]["hardware_sha256"],
            "preflight": validated_lock["lock_core"]["preflight"],
        },
        "training_evidence": training_evidence,
        "subject_count": len(subject_ids),
        "subject_ids_sha256": _canonical_sha256(subject_ids),
        "estimand": "Delta(h,n) = mean_s[Pearson_external(h,n,s) - Pearson_external(mean_linear,n,s)]",
        "baseline_head": BASELINE_HEAD,
        "candidate_heads": list(CANDIDATE_HEADS),
        "training_sizes": list(SIZES),
        "cells": cell_results,
        "contrasts": contrasts,
        "bootstrap": {
            "iterations": iterations,
            "seed": bootstrap_seed,
            "confidence": confidence,
            "minimum_valid_replicates": minimum_valid,
            "resampling": "same paired seed vector and subject vector reused across all cells",
            "percentile_method": "linear_interpolation",
        },
    }
    return {**output_body, "analysis_sha256": _canonical_sha256(output_body)}
