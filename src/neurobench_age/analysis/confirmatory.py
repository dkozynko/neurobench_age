"""Predeclared statistics for the sealed external frozen-probe study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from neurobench_age.pipelines.frozen_probe_training import APPROVED_HEADS
from neurobench_age.research.protocol import ProtocolError, StudyProtocol, load_study_protocol
from neurobench_age.research.study_lock import (
    StudyLockError,
    canonical_sha256,
    load_checkpoint_inventory,
    load_study_lock,
)


EXPECTED_SEEDS = tuple(range(33, 43))


class ConfirmatoryAnalysisError(ValueError):
    """Raised when confirmatory evidence or statistics violate the protocol."""


@dataclass(frozen=True)
class PredictionSeries:
    seed: int
    subject_ids: tuple[str, ...]
    targets: tuple[float, ...]
    predictions: tuple[float, ...]
    provenance: Mapping[str, Any]


def _finite_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ConfirmatoryAnalysisError(f"{name} must be numeric") from error
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ConfirmatoryAnalysisError(f"{name} must be a finite one-dimensional array")
    return result


def regression_metrics(
    targets: Sequence[float], predictions: Sequence[float]
) -> dict[str, float]:
    """Return subject-level Pearson, MAE, RMSE, and R-squared."""

    truth = _finite_vector(targets, name="targets")
    estimate = _finite_vector(predictions, name="predictions")
    if truth.size != estimate.size or truth.size < 2:
        raise ConfirmatoryAnalysisError(
            "regression metrics require equal arrays with at least two subjects"
        )
    truth_centered = truth - truth.mean()
    estimate_centered = estimate - estimate.mean()
    truth_ss = float(np.square(truth_centered).sum())
    denominator = math.sqrt(
        truth_ss * float(np.square(estimate_centered).sum())
    )
    if truth_ss <= 0.0 or denominator <= 1e-12:
        raise ConfirmatoryAnalysisError(
            "Pearson and R-squared require non-constant targets and predictions"
        )
    residual = estimate - truth
    result = {
        "pearson": float(np.dot(truth_centered, estimate_centered) / denominator),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "r2": float(1.0 - np.square(residual).sum() / truth_ss),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ConfirmatoryAnalysisError("regression metrics are non-finite")
    return result


def calibration_parameters(
    targets: Sequence[float], predictions: Sequence[float]
) -> dict[str, float]:
    """Fit observed age = intercept + slope * predicted age by least squares."""

    truth = _finite_vector(targets, name="targets")
    estimate = _finite_vector(predictions, name="predictions")
    if truth.size != estimate.size or truth.size < 2:
        raise ConfirmatoryAnalysisError(
            "calibration requires equal arrays with at least two subjects"
        )
    centered = estimate - estimate.mean()
    denominator = float(np.square(centered).sum())
    if denominator <= 1e-12:
        raise ConfirmatoryAnalysisError(
            "calibration is undefined for constant predictions"
        )
    slope = float(np.dot(centered, truth - truth.mean()) / denominator)
    intercept = float(truth.mean() - slope * estimate.mean())
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ConfirmatoryAnalysisError("calibration parameters are non-finite")
    return {"intercept": intercept, "slope": slope}


def _validate_paired_series(
    candidate: Mapping[int, PredictionSeries],
    baseline: Mapping[int, PredictionSeries],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    candidate_seeds = tuple(sorted(candidate))
    baseline_seeds = tuple(sorted(baseline))
    if candidate_seeds != EXPECTED_SEEDS or baseline_seeds != EXPECTED_SEEDS:
        raise ConfirmatoryAnalysisError(
            "paired analysis requires the exact seed inventory 33 through 42"
        )
    reference_subjects: tuple[str, ...] | None = None
    reference_targets: tuple[float, ...] | None = None
    reference_provenance: dict[str, Any] | None = None
    for seed in EXPECTED_SEEDS:
        candidate_series = candidate[seed]
        baseline_series = baseline[seed]
        if (
            not isinstance(candidate_series, PredictionSeries)
            or not isinstance(baseline_series, PredictionSeries)
            or candidate_series.seed != seed
            or baseline_series.seed != seed
        ):
            raise ConfirmatoryAnalysisError("prediction series seed identity is invalid")
        for series in (candidate_series, baseline_series):
            if (
                len(series.subject_ids) != len(series.targets)
                or len(series.targets) != len(series.predictions)
                or len(series.subject_ids) < 2
                or len(set(series.subject_ids)) != len(series.subject_ids)
            ):
                raise ConfirmatoryAnalysisError("prediction series shape is invalid")
            _finite_vector(series.targets, name="targets")
            _finite_vector(series.predictions, name="predictions")
        if candidate_series.subject_ids != baseline_series.subject_ids:
            raise ConfirmatoryAnalysisError(
                f"candidate and baseline subject order differs for seed {seed}"
            )
        if candidate_series.targets != baseline_series.targets:
            raise ConfirmatoryAnalysisError(
                f"candidate and baseline targets differ for seed {seed}"
            )
        if dict(candidate_series.provenance) != dict(baseline_series.provenance):
            raise ConfirmatoryAnalysisError(
                f"candidate and baseline provenance differs for seed {seed}"
            )
        if reference_subjects is None:
            reference_subjects = candidate_series.subject_ids
            reference_targets = candidate_series.targets
            reference_provenance = dict(candidate_series.provenance)
        elif candidate_series.subject_ids != reference_subjects:
            raise ConfirmatoryAnalysisError("subject order differs across seeds")
        elif candidate_series.targets != reference_targets:
            raise ConfirmatoryAnalysisError("targets differ across seeds")
        elif dict(candidate_series.provenance) != reference_provenance:
            raise ConfirmatoryAnalysisError("provenance differs across seeds")
    assert reference_subjects is not None
    return EXPECTED_SEEDS, reference_subjects


def paired_seed_statistics(
    candidate: Mapping[int, PredictionSeries],
    baseline: Mapping[int, PredictionSeries],
) -> dict[str, Any]:
    """Compute the paired seed-level primary estimand and descriptive metrics."""

    seeds, subject_ids = _validate_paired_series(candidate, baseline)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        candidate_metrics = regression_metrics(
            candidate[seed].targets, candidate[seed].predictions
        )
        baseline_metrics = regression_metrics(
            baseline[seed].targets, baseline[seed].predictions
        )
        candidate_calibration = calibration_parameters(
            candidate[seed].targets, candidate[seed].predictions
        )
        baseline_calibration = calibration_parameters(
            baseline[seed].targets, baseline[seed].predictions
        )
        delta = candidate_metrics["pearson"] - baseline_metrics["pearson"]
        rows.append(
            {
                "seed": seed,
                "candidate": {**candidate_metrics, "calibration": candidate_calibration},
                "baseline": {**baseline_metrics, "calibration": baseline_calibration},
                "pearson_delta": float(delta),
            }
        )
    deltas = np.asarray([row["pearson_delta"] for row in rows], dtype=np.float64)
    return {
        "seeds": list(seeds),
        "subject_count": len(subject_ids),
        "per_seed": rows,
        "mean_pearson_delta": float(deltas.mean()),
        "seed_delta_sample_sd": float(deltas.std(ddof=1)),
        "wins": int(np.count_nonzero(deltas > 0.0)),
        "ties": int(np.count_nonzero(deltas == 0.0)),
        "losses": int(np.count_nonzero(deltas < 0.0)),
        "worst_seed_delta": float(deltas.min()),
    }


def hierarchical_paired_bootstrap(
    candidate: Mapping[int, PredictionSeries],
    baseline: Mapping[int, PredictionSeries],
    *,
    iterations: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    """Resample paired seeds and subjects and return a percentile interval."""

    seeds, subject_ids = _validate_paired_series(candidate, baseline)
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0.0 < confidence < 1.0
    ):
        raise ConfirmatoryAnalysisError("bootstrap specification is invalid")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    failed = 0
    for _ in range(iterations):
        sampled_seed_indices = rng.integers(0, len(seeds), size=len(seeds))
        sampled_subject_indices = rng.integers(
            0, len(subject_ids), size=len(subject_ids)
        )
        seed_deltas: list[float] = []
        try:
            for seed_index in sampled_seed_indices:
                sampled_seed = seeds[int(seed_index)]
                candidate_series = candidate[sampled_seed]
                baseline_series = baseline[sampled_seed]
                sampled_targets = tuple(
                    candidate_series.targets[int(index)]
                    for index in sampled_subject_indices
                )
                candidate_predictions = tuple(
                    candidate_series.predictions[int(index)]
                    for index in sampled_subject_indices
                )
                baseline_predictions = tuple(
                    baseline_series.predictions[int(index)]
                    for index in sampled_subject_indices
                )
                seed_deltas.append(
                    regression_metrics(sampled_targets, candidate_predictions)[
                        "pearson"
                    ]
                    - regression_metrics(sampled_targets, baseline_predictions)[
                        "pearson"
                    ]
                )
        except ConfirmatoryAnalysisError:
            failed += 1
            continue
        value = float(np.mean(seed_deltas))
        if math.isfinite(value):
            values.append(value)
        else:
            failed += 1
    if not values:
        raise ConfirmatoryAnalysisError("all hierarchical bootstrap iterations failed")
    alpha = 1.0 - confidence
    distribution = np.asarray(values, dtype=np.float64)
    return {
        "ci_low": float(np.quantile(distribution, alpha / 2.0)),
        "ci_high": float(np.quantile(distribution, 1.0 - alpha / 2.0)),
        "confidence": float(confidence),
        "iterations": iterations,
        "valid_iterations": len(values),
        "failed_iterations": failed,
        "seed": seed,
        "seed_count": len(seeds),
        "subject_count": len(subject_ids),
        "resampling": "paired_seeds_and_subjects_with_replacement",
    }


def exact_seed_randomization(deltas: Sequence[float]) -> dict[str, Any]:
    """Enumerate the exact one-sided paired sign-flip distribution."""

    values = _finite_vector(deltas, name="seed deltas")
    if values.size != len(EXPECTED_SEEDS):
        raise ConfirmatoryAnalysisError("randomization requires exactly ten seed deltas")
    observed = float(values.mean())
    exceedances = 0
    permutations = 0
    for signs in product((-1.0, 1.0), repeat=len(EXPECTED_SEEDS)):
        permuted = float(np.mean(values * np.asarray(signs, dtype=np.float64)))
        exceedances += int(permuted >= observed - 1e-15)
        permutations += 1
    return {
        "observed_mean_delta": observed,
        "p_value": float(exceedances / permutations),
        "permutations": permutations,
        "tail": "greater",
        "zero_deltas": int(np.count_nonzero(values == 0.0)),
        "comparison": "permuted_mean_greater_than_or_equal_to_observed",
    }


def holm_step_down(
    p_values: Mapping[str, float], *, order: Sequence[str]
) -> dict[str, float]:
    """Apply Holm correction, using the sealed order to break equal p-values."""

    declared = tuple(order)
    if len(declared) == 0 or len(set(declared)) != len(declared) or set(p_values) != set(declared):
        raise ConfirmatoryAnalysisError("Holm inputs do not match the declared order")
    normalized: dict[str, float] = {}
    for name in declared:
        value = p_values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfirmatoryAnalysisError("Holm p-values must be numeric")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ConfirmatoryAnalysisError("Holm p-values must be within [0, 1]")
        normalized[name] = value
    rank = {name: index for index, name in enumerate(declared)}
    sorted_names = sorted(declared, key=lambda name: (normalized[name], rank[name]))
    adjusted_sorted: dict[str, float] = {}
    running = 0.0
    count = len(sorted_names)
    for index, name in enumerate(sorted_names):
        running = max(running, (count - index) * normalized[name])
        adjusted_sorted[name] = min(1.0, running)
    return {name: adjusted_sorted[name] for name in declared}


def stable_improvement_decision(
    *,
    adjusted_p_value: float,
    bootstrap_ci_low: float,
    wins: int,
    worst_seed_delta: float,
    alpha: float,
    minimum_seed_wins: int,
    minimum_worst_seed_delta: float,
    require_ci_above_zero: bool,
    adequately_powered: bool,
) -> dict[str, Any]:
    """Evaluate all conjunctive conditions from the sealed decision rule."""

    conditions = {
        "adequate_primary_cohort": adequately_powered,
        "holm_adjusted_p_below_alpha": adjusted_p_value < alpha,
        "bootstrap_lower_bound_above_zero": (
            bootstrap_ci_low > 0.0 if require_ci_above_zero else True
        ),
        "minimum_seed_wins": wins >= minimum_seed_wins,
        "worst_seed_delta_at_least_minimum": (
            worst_seed_delta >= minimum_worst_seed_delta
        ),
    }
    return {
        "established_stable_improvement": all(conditions.values()),
        "conditions": conditions,
    }


def confirmatory_conclusion(
    established_heads: Sequence[str], *, adequately_powered: bool = True
) -> str:
    """Return only the bounded conclusion permitted by the protocol."""

    established = tuple(established_heads)
    unknown = set(established) - set(APPROVED_HEADS[1:])
    if unknown or len(established) != len(set(established)):
        raise ConfirmatoryAnalysisError("conclusion contains an invalid head inventory")
    if not adequately_powered:
        if established:
            raise ConfirmatoryAnalysisError(
                "an underpowered study cannot establish a stable improvement"
            )
        return (
            "The primary external cohort was underpowered under the predeclared "
            "minimum, so no confirmatory superiority claim is made; all estimates "
            "are descriptive."
        )
    if established:
        return (
            "The following tested heads established a stable external gain under "
            f"the predeclared protocol: {', '.join(established)}."
        )
    return (
        "No tested complex head established a stable external gain under the "
        "predeclared protocol; this does not establish equivalence."
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfirmatoryAnalysisError(f"could not read {description}: {path}") from error
    if not isinstance(payload, dict):
        raise ConfirmatoryAnalysisError(f"{description} must contain a JSON object")
    return payload


def _validate_hashed_object(
    payload: Mapping[str, Any], hash_field: str, description: str
) -> None:
    claimed = payload.get(hash_field)
    body = {key: value for key, value in payload.items() if key != hash_field}
    if not _is_sha256(claimed) or claimed != canonical_sha256(body):
        raise ConfirmatoryAnalysisError(f"{description} hash does not match")


def _statistics_payload(protocol: StudyProtocol) -> dict[str, Any]:
    payload = asdict(protocol.statistics)
    payload["holm_order"] = list(protocol.statistics.holm_order)
    return payload


def _load_completed_state(lock_path: Path, lock: Mapping[str, Any]) -> None:
    state = _load_json(lock_path.parent / "study_state.json", "study state")
    if (
        state.get("state") != "completed"
        or state.get("lock_sha256") != lock["lock_sha256"]
    ):
        raise ConfirmatoryAnalysisError(
            "confirmatory analysis requires the exact completed study"
        )


def _load_manifest(
    manifest_path: Path,
    *,
    lock: Mapping[str, Any],
    protocol: StudyProtocol,
) -> tuple[tuple[str, ...], tuple[float, ...], dict[str, Any]]:
    if _sha256_file(manifest_path) != lock["mipdb_manifest_sha256"]:
        raise ConfirmatoryAnalysisError("MIPDB manifest provenance differs from lock")
    manifest = _load_json(manifest_path, "MIPDB manifest")
    if manifest.get("schema_version") != 2 or manifest.get("status") != "finalized":
        raise ConfirmatoryAnalysisError("MIPDB manifest is not finalized by cohort QC")
    if not _is_sha256(manifest.get("cohort_qc_sha256")):
        raise ConfirmatoryAnalysisError("MIPDB cohort QC identity is invalid")
    if manifest.get("protocol_sha256") != protocol.sha256:
        raise ConfirmatoryAnalysisError("MIPDB manifest protocol provenance differs")
    cohorts = manifest.get("cohorts")
    subject_hashes = manifest.get("subject_list_sha256")
    if not isinstance(cohorts, Mapping) or not isinstance(subject_hashes, Mapping):
        raise ConfirmatoryAnalysisError("MIPDB cohort metadata is invalid")
    pilot = cohorts.get("pilot")
    primary = cohorts.get("primary")
    extrapolation = cohorts.get("extrapolation")
    if not all(isinstance(value, list) for value in (pilot, primary, extrapolation)):
        raise ConfirmatoryAnalysisError("MIPDB cohorts must be ordered arrays")
    if len(pilot) != protocol.datasets.pilot_size:
        raise ConfirmatoryAnalysisError("MIPDB pilot cohort size differs from protocol")
    if (
        set(pilot) & set(primary)
        or set(pilot) & set(extrapolation)
        or set(primary) & set(extrapolation)
        or len(primary) != len(set(primary))
        or not primary
    ):
        raise ConfirmatoryAnalysisError("MIPDB cohort identity is invalid")
    expected_hashes = {
        "pilot": lock["subject_list_sha256"]["mipdb_pilot"],
        "primary": lock["subject_list_sha256"]["mipdb_primary"],
        "extrapolation": lock["subject_list_sha256"]["mipdb_extrapolation"],
    }
    for name, subjects in (
        ("pilot", pilot),
        ("primary", primary),
        ("extrapolation", extrapolation),
    ):
        actual = canonical_sha256(subjects)
        if subject_hashes.get(name) != actual or expected_hashes[name] != actual:
            raise ConfirmatoryAnalysisError(
                f"MIPDB {name} subject provenance differs from lock"
            )
    raw_subjects = manifest.get("subjects")
    if not isinstance(raw_subjects, list):
        raise ConfirmatoryAnalysisError("MIPDB subjects must be an array")
    ages: dict[str, float] = {}
    for item in raw_subjects:
        if not isinstance(item, Mapping):
            raise ConfirmatoryAnalysisError("MIPDB subject metadata is invalid")
        subject_id = item.get("subject_id")
        age = item.get("age")
        if (
            not isinstance(subject_id, str)
            or subject_id in ages
            or isinstance(age, bool)
            or not isinstance(age, (int, float))
            or not math.isfinite(float(age))
        ):
            raise ConfirmatoryAnalysisError("MIPDB subject metadata is invalid")
        ages[subject_id] = float(age)
    if any(subject_id not in ages for subject_id in primary):
        raise ConfirmatoryAnalysisError("MIPDB primary subject age is missing")
    minimum = protocol.datasets.minimum_primary_subjects
    expected_underpowered = len(primary) < minimum
    if (
        manifest.get("minimum_primary_subjects") != minimum
        or manifest.get("underpowered") is not expected_underpowered
    ):
        raise ConfirmatoryAnalysisError("MIPDB power metadata differs from protocol")
    exclusions = manifest.get("exclusions")
    if not isinstance(exclusions, list) or any(
        not isinstance(item, Mapping) for item in exclusions
    ):
        raise ConfirmatoryAnalysisError("MIPDB exclusions are invalid")
    cohort = {
        "pilot_subjects": len(pilot),
        "primary_subjects": len(primary),
        "extrapolation_subjects": len(extrapolation),
        "minimum_primary_subjects": minimum,
        "underpowered": expected_underpowered,
        "power_warning": (
            f"Primary cohort has {len(primary)} subjects, below the predeclared minimum of {minimum}."
            if expected_underpowered
            else None
        ),
    }
    return (
        tuple(primary),
        tuple(ages[subject_id] for subject_id in primary),
        {"cohort": cohort, "exclusions": [dict(item) for item in exclusions]},
    )


def _load_resources(
    *,
    checkpoint_root: Path,
    checkpoint_inventory_path: Path,
    lock: Mapping[str, Any],
) -> tuple[dict[tuple[str, int], str], dict[str, Any]]:
    try:
        inventory = load_checkpoint_inventory(checkpoint_inventory_path)
    except StudyLockError as error:
        raise ConfirmatoryAnalysisError(str(error)) from error
    if inventory["checkpoint_inventory_sha256"] != lock[
        "checkpoint_inventory_sha256"
    ]:
        raise ConfirmatoryAnalysisError("checkpoint inventory provenance differs from lock")
    checkpoint_hashes: dict[tuple[str, int], str] = {}
    resources_by_head: dict[str, list[dict[str, Any]]] = {
        head_name: [] for head_name in APPROVED_HEADS
    }
    for record in inventory["runs"]:
        head_name = record["head_name"]
        seed = record["seed"]
        checkpoint_hashes[(head_name, seed)] = record["checkpoint_sha256"]
        manifest_path = (
            checkpoint_root / head_name / f"seed-{seed}" / "run_manifest.json"
        )
        manifest = _load_json(manifest_path, "frozen-probe run manifest")
        _validate_hashed_object(
            manifest, "run_manifest_sha256", "frozen-probe run manifest"
        )
        parameters = manifest.get("head_parameters")
        values = {
            "runtime_seconds": manifest.get("runtime_seconds"),
            "peak_process_rss_bytes": manifest.get("peak_process_rss_bytes"),
            "peak_accelerator_memory_bytes": manifest.get(
                "peak_accelerator_memory_bytes"
            ),
        }
        if (
            manifest.get("head_name") != head_name
            or manifest.get("seed") != seed
            or manifest.get("run_identity_sha256")
            != record["run_identity_sha256"]
            or manifest.get("checkpoint_sha256")
            != record["checkpoint_sha256"]
            or manifest.get("selected_epoch") != record["selected_epoch"]
            or manifest.get("run_manifest_sha256")
            != record["run_manifest_sha256"]
            or not isinstance(parameters, Mapping)
            or parameters.get("total") != record["head_parameter_count"]
            or parameters.get("trainable") != record["head_parameter_count"]
        ):
            raise ConfirmatoryAnalysisError("frozen-probe resource provenance differs")
        normalized_values: dict[str, float | int] = {}
        for name, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ConfirmatoryAnalysisError(
                    f"frozen-probe resource {name} is invalid"
                )
            normalized_values[name] = value
        resources_by_head[head_name].append(
            {"seed": seed, **normalized_values}
        )
    resources: dict[str, Any] = {}
    for head_name, rows in resources_by_head.items():
        rows.sort(key=lambda row: row["seed"])
        parameter_count = next(
            record["head_parameter_count"]
            for record in inventory["runs"]
            if record["head_name"] == head_name
        )
        resources[head_name] = {
            "head_parameter_count": parameter_count,
            "training_runtime_seconds_total": float(
                sum(float(row["runtime_seconds"]) for row in rows)
            ),
            "training_runtime_seconds_mean": float(
                np.mean([float(row["runtime_seconds"]) for row in rows])
            ),
            "peak_process_rss_bytes_max": max(
                int(row["peak_process_rss_bytes"]) for row in rows
            ),
            "peak_accelerator_memory_bytes_max": max(
                int(row["peak_accelerator_memory_bytes"]) for row in rows
            ),
            "per_seed": rows,
        }
    return checkpoint_hashes, resources


_PREDICTION_FIELDS = {
    "schema_version",
    "study_id",
    "lock_sha256",
    "protocol_sha256",
    "training_source_sha256",
    "environment_sha256",
    "mipdb_manifest_sha256",
    "preprocessing_sha256",
    "encoder_checkpoint_sha256",
    "head_checkpoint_sha256",
    "head_name",
    "seed",
    "subject_id",
    "target_age",
    "representation_cache_key",
    "qc_status",
    "qc_sha256",
    "prediction",
    "prediction_sha256",
}


def _load_prediction_series(
    *,
    prediction_root: Path,
    lock: Mapping[str, Any],
    subject_ids: tuple[str, ...],
    targets: tuple[float, ...],
    checkpoint_hashes: Mapping[tuple[str, int], str],
) -> dict[str, dict[int, PredictionSeries]]:
    inventory_path = prediction_root / "prediction_inventory.json"
    inventory = _load_json(inventory_path, "prediction inventory")
    _validate_hashed_object(
        inventory, "prediction_inventory_sha256", "prediction inventory"
    )
    if (
        inventory.get("status") != "complete"
        or inventory.get("study_id") != lock["study_id"]
        or inventory.get("lock_sha256") != lock["lock_sha256"]
        or tuple(inventory.get("heads", ())) != APPROVED_HEADS
        or tuple(inventory.get("seeds", ())) != EXPECTED_SEEDS
        or tuple(inventory.get("subjects", ())) != subject_ids
    ):
        raise ConfirmatoryAnalysisError("prediction inventory provenance differs")
    expected_count = len(APPROVED_HEADS) * len(EXPECTED_SEEDS) * len(subject_ids)
    entries = inventory.get("predictions")
    if (
        inventory.get("prediction_count") != expected_count
        or not isinstance(entries, list)
        or len(entries) != expected_count
    ):
        raise ConfirmatoryAnalysisError("prediction inventory is incomplete")
    completion = _load_json(
        prediction_root / "evaluation_completed.json", "evaluation completion marker"
    )
    _validate_hashed_object(completion, "marker_sha256", "evaluation completion marker")
    if (
        completion.get("status") != "complete"
        or completion.get("lock_sha256") != lock["lock_sha256"]
        or completion.get("prediction_inventory_sha256")
        != inventory["prediction_inventory_sha256"]
    ):
        raise ConfirmatoryAnalysisError("evaluation completion provenance differs")

    expected_keys = {
        (head_name, seed, subject_id)
        for head_name in APPROVED_HEADS
        for seed in EXPECTED_SEEDS
        for subject_id in subject_ids
    }
    entry_by_key: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "head_name",
            "seed",
            "subject_id",
            "path",
            "prediction_sha256",
        }:
            raise ConfirmatoryAnalysisError("prediction inventory entry is invalid")
        key = (entry["head_name"], entry["seed"], entry["subject_id"])
        if key in entry_by_key:
            raise ConfirmatoryAnalysisError("prediction inventory contains duplicates")
        entry_by_key[key] = entry
    if set(entry_by_key) != expected_keys:
        raise ConfirmatoryAnalysisError("prediction inventory is incomplete or has extras")
    expected_paths = {
        prediction_root
        / "predictions"
        / head_name
        / f"seed-{seed}"
        / f"{subject_id}.json"
        for head_name, seed, subject_id in expected_keys
    }
    actual_paths = set((prediction_root / "predictions").rglob("*.json"))
    if actual_paths != expected_paths:
        raise ConfirmatoryAnalysisError("prediction file inventory is incomplete or has extras")

    target_by_subject = dict(zip(subject_ids, targets))
    records: dict[tuple[str, int, str], dict[str, Any]] = {}
    common_expected = {
        "study_id": lock["study_id"],
        "lock_sha256": lock["lock_sha256"],
        "protocol_sha256": lock["protocol_sha256"],
        "training_source_sha256": lock["training_source_sha256"],
        "environment_sha256": lock["environment_sha256"],
        "mipdb_manifest_sha256": lock["mipdb_manifest_sha256"],
        "preprocessing_sha256": lock["preprocessing_sha256"],
        "encoder_checkpoint_sha256": lock["encoder_checkpoint_sha256"],
    }
    for key in sorted(expected_keys):
        head_name, seed, subject_id = key
        entry = entry_by_key[key]
        expected_relative = (
            Path("predictions")
            / head_name
            / f"seed-{seed}"
            / f"{subject_id}.json"
        ).as_posix()
        if entry.get("path") != expected_relative:
            raise ConfirmatoryAnalysisError("prediction path provenance differs")
        path = prediction_root / expected_relative
        record = _load_json(path, "subject prediction")
        if set(record) != _PREDICTION_FIELDS:
            raise ConfirmatoryAnalysisError("subject prediction schema differs")
        _validate_hashed_object(record, "prediction_sha256", "subject prediction")
        if record["prediction_sha256"] != entry["prediction_sha256"]:
            raise ConfirmatoryAnalysisError("prediction inventory hash differs from record")
        expected = {
            **common_expected,
            "schema_version": 2,
            "head_name": head_name,
            "seed": seed,
            "subject_id": subject_id,
            "target_age": target_by_subject[subject_id],
            "head_checkpoint_sha256": checkpoint_hashes[(head_name, seed)],
            "qc_status": "passed",
        }
        if any(record.get(field) != value for field, value in expected.items()):
            raise ConfirmatoryAnalysisError("subject prediction provenance differs from lock")
        if (
            not _is_sha256(record.get("representation_cache_key"))
            or not _is_sha256(record.get("qc_sha256"))
            or isinstance(record.get("prediction"), bool)
            or not isinstance(record.get("prediction"), (int, float))
            or not math.isfinite(float(record["prediction"]))
        ):
            raise ConfirmatoryAnalysisError("subject prediction content is invalid")
        records[key] = record

    result: dict[str, dict[int, PredictionSeries]] = {
        head_name: {} for head_name in APPROVED_HEADS
    }
    for head_name in APPROVED_HEADS:
        for seed in EXPECTED_SEEDS:
            ordered = [records[(head_name, seed, subject_id)] for subject_id in subject_ids]
            subject_evidence = tuple(
                (
                    row["subject_id"],
                    row["representation_cache_key"],
                    row["qc_sha256"],
                )
                for row in ordered
            )
            result[head_name][seed] = PredictionSeries(
                seed=seed,
                subject_ids=subject_ids,
                targets=targets,
                predictions=tuple(float(row["prediction"]) for row in ordered),
                provenance={
                    **common_expected,
                    "subject_evidence": subject_evidence,
                },
            )
    return result


def _head_summary(series: Mapping[int, PredictionSeries]) -> dict[str, Any]:
    rows = []
    for seed in EXPECTED_SEEDS:
        metrics = regression_metrics(series[seed].targets, series[seed].predictions)
        rows.append(
            {
                "seed": seed,
                **metrics,
                "calibration": calibration_parameters(
                    series[seed].targets, series[seed].predictions
                ),
            }
        )
    pearsons = np.asarray([row["pearson"] for row in rows], dtype=np.float64)
    return {
        "per_seed": rows,
        "mean_pearson": float(pearsons.mean()),
        "pearson_sample_sd": float(pearsons.std(ddof=1)),
        "mean_mae": float(np.mean([row["mae"] for row in rows])),
        "mean_rmse": float(np.mean([row["rmse"] for row in rows])),
        "mean_r2": float(np.mean([row["r2"] for row in rows])),
        "mean_calibration_intercept": float(
            np.mean([row["calibration"]["intercept"] for row in rows])
        ),
        "mean_calibration_slope": float(
            np.mean([row["calibration"]["slope"] for row in rows])
        ),
    }


def _publish_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise ConfirmatoryAnalysisError(
            f"immutable analysis artifact already exists: {path}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def analyze_confirmatory_study(
    *,
    protocol_path: Path,
    lock_path: Path,
    checkpoint_root: Path,
    checkpoint_inventory_path: Path,
    mipdb_manifest_path: Path,
    prediction_root: Path,
    analysis_output_root: Path,
) -> dict[str, Any]:
    """Audit completed predictions and run only the sealed confirmatory analysis."""

    try:
        protocol = load_study_protocol(protocol_path)
        lock = load_study_lock(lock_path)
    except (ProtocolError, StudyLockError) as error:
        raise ConfirmatoryAnalysisError(str(error)) from error
    if (
        lock["protocol_sha256"] != protocol.sha256
        or lock["study_id"] != protocol.study_id
        or tuple(lock["heads"]) != APPROVED_HEADS
        or tuple(lock["seeds"]) != EXPECTED_SEEDS
        or lock["statistics_sha256"] != protocol.statistics_sha256
    ):
        raise ConfirmatoryAnalysisError("sealed analysis provenance differs from protocol")
    prediction_root = Path(prediction_root).resolve()
    if prediction_root != Path(lock["output_root"]).resolve():
        raise ConfirmatoryAnalysisError("prediction root provenance differs from lock")
    _load_completed_state(Path(lock_path), lock)
    subject_ids, targets, manifest_evidence = _load_manifest(
        Path(mipdb_manifest_path), lock=lock, protocol=protocol
    )
    checkpoint_hashes, resources = _load_resources(
        checkpoint_root=Path(checkpoint_root),
        checkpoint_inventory_path=Path(checkpoint_inventory_path),
        lock=lock,
    )
    series = _load_prediction_series(
        prediction_root=prediction_root,
        lock=lock,
        subject_ids=subject_ids,
        targets=targets,
        checkpoint_hashes=checkpoint_hashes,
    )

    comparisons: dict[str, dict[str, Any]] = {}
    raw_p_values: dict[str, float] = {}
    for head_name in protocol.statistics.holm_order:
        paired = paired_seed_statistics(series[head_name], series["mean_linear"])
        bootstrap = hierarchical_paired_bootstrap(
            series[head_name],
            series["mean_linear"],
            iterations=protocol.statistics.bootstrap_iterations,
            seed=protocol.statistics.bootstrap_seed,
            confidence=protocol.statistics.confidence,
        )
        randomization = exact_seed_randomization(
            [row["pearson_delta"] for row in paired["per_seed"]]
        )
        raw_p_values[head_name] = randomization["p_value"]
        comparisons[head_name] = {
            "paired": paired,
            "bootstrap": bootstrap,
            "randomization": randomization,
            "candidate_metrics": _head_summary(series[head_name]),
            "resources": resources[head_name],
        }
    adjusted = holm_step_down(
        raw_p_values, order=protocol.statistics.holm_order
    )
    established: list[str] = []
    for head_name in protocol.statistics.holm_order:
        comparison = comparisons[head_name]
        comparison["holm_adjusted_p_value"] = adjusted[head_name]
        comparison["decision"] = stable_improvement_decision(
            adjusted_p_value=adjusted[head_name],
            bootstrap_ci_low=comparison["bootstrap"]["ci_low"],
            wins=comparison["paired"]["wins"],
            worst_seed_delta=comparison["paired"]["worst_seed_delta"],
            alpha=protocol.statistics.alpha,
            minimum_seed_wins=protocol.statistics.minimum_seed_wins,
            minimum_worst_seed_delta=protocol.statistics.minimum_worst_seed_delta,
            require_ci_above_zero=protocol.statistics.require_ci_above_zero,
            adequately_powered=not manifest_evidence["cohort"]["underpowered"],
        )
        if comparison["decision"]["established_stable_improvement"]:
            established.append(head_name)
    conclusion = confirmatory_conclusion(
        established,
        adequately_powered=not manifest_evidence["cohort"]["underpowered"],
    )
    report_body: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "study_id": lock["study_id"],
        "lock_sha256": lock["lock_sha256"],
        "protocol_sha256": protocol.sha256,
        "statistics_sha256": lock["statistics_sha256"],
        "prediction_inventory_sha256": _load_json(
            prediction_root / "prediction_inventory.json", "prediction inventory"
        )["prediction_inventory_sha256"],
        "estimand": "mean_across_seeds_candidate_minus_mean_linear_external_pearson",
        "baseline": {
            "head_name": "mean_linear",
            "metrics": _head_summary(series["mean_linear"]),
        },
        "comparisons": comparisons,
        "resources": resources,
        "cohort": manifest_evidence["cohort"],
        "exclusions": manifest_evidence["exclusions"],
        "established_heads": established,
        "conclusion": conclusion,
        "protocol_statistics": _statistics_payload(protocol),
    }
    report = {**report_body, "analysis_sha256": canonical_sha256(report_body)}
    output_root = Path(analysis_output_root)
    output_path = output_root / "confirmatory_analysis.json"
    if output_root.exists():
        extras = [path for path in output_root.iterdir() if path != output_path]
        if extras:
            raise ConfirmatoryAnalysisError(
                "analysis output directory is non-empty and does not exactly resume"
            )
    if output_path.exists():
        existing = _load_json(output_path, "confirmatory analysis")
        _validate_hashed_object(existing, "analysis_sha256", "confirmatory analysis")
        if existing != report:
            raise ConfirmatoryAnalysisError(
                "existing confirmatory analysis does not match exact resume"
            )
        return existing
    _publish_json_create_only(output_path, report)
    return report
