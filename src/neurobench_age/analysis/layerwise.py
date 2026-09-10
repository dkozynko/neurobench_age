"""Exploratory paired analysis for the completed layer-wise probe study."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from neurobench_age.analysis.confirmatory import (
    PredictionSeries,
    exact_seed_randomization,
    hierarchical_paired_bootstrap,
    holm_step_down,
    paired_seed_statistics,
    regression_metrics,
)


class LayerwiseAnalysisError(ValueError):
    """Raised when layer-wise analysis inputs or statistics are invalid."""


EXPECTED_HEAD_LAYERS = {
    "mean_linear_layer_m4": -4,
    "mean_linear_layer_m3": -3,
    "mean_linear_layer_m2": -2,
    "mean_linear_layer_m1": -1,
}
EXPECTED_SEEDS = tuple(range(33, 43))


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LayerwiseAnalysisError(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise LayerwiseAnalysisError(f"{field} must be finite")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LayerwiseAnalysisError(f"could not read {description}: {path}") from error
    if not isinstance(value, dict):
        raise LayerwiseAnalysisError(f"{description} must be a JSON object")
    return value


@dataclass(frozen=True)
class LayerwiseAnalysisConfig:
    analysis_id: str
    scope: str
    baseline_head: str
    comparison_heads: tuple[str, ...]
    seeds: tuple[int, ...]
    expected_subject_count: int
    bootstrap_iterations: int
    bootstrap_seed: int
    confidence: float
    randomization_tail: str
    holm_order: tuple[str, ...]
    alpha: float
    sha256: str


def load_layerwise_analysis_config(path: Path) -> LayerwiseAnalysisConfig:
    """Load and validate the immutable exploratory analysis specification."""

    payload = _load_json(Path(path), "layer-wise analysis config")
    expected_fields = {
        "schema_version",
        "analysis_id",
        "scope",
        "baseline_head",
        "comparison_heads",
        "seeds",
        "expected_subject_count",
        "bootstrap_iterations",
        "bootstrap_seed",
        "confidence",
        "randomization_tail",
        "holm_order",
        "alpha",
    }
    if set(payload) != expected_fields or payload.get("schema_version") != 1:
        raise LayerwiseAnalysisError("layer-wise analysis config schema is invalid")
    comparisons = payload.get("comparison_heads")
    seeds = payload.get("seeds")
    holm_order = payload.get("holm_order")
    if (
        not isinstance(payload.get("analysis_id"), str)
        or payload.get("scope") != "exploratory_secondary"
        or not isinstance(payload.get("baseline_head"), str)
        or not isinstance(comparisons, list)
        or not isinstance(seeds, list)
        or not isinstance(holm_order, list)
        or tuple(comparisons) != tuple(holm_order)
        or tuple(comparisons)
        != (
            "mean_linear_layer_m4",
            "mean_linear_layer_m3",
            "mean_linear_layer_m2",
        )
        or payload.get("baseline_head") != "mean_linear_layer_m1"
        or tuple(seeds) != tuple(range(33, 43))
    ):
        raise LayerwiseAnalysisError("layer-wise analysis comparison or seed inventory is invalid")
    scalar_fields = {
        "expected_subject_count": int,
        "bootstrap_iterations": int,
        "bootstrap_seed": int,
    }
    for field, expected_type in scalar_fields.items():
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, expected_type) or value <= 0:
            raise LayerwiseAnalysisError(f"layer-wise analysis {field} is invalid")
    confidence = payload.get("confidence")
    alpha = payload.get("alpha")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 < float(confidence) < 1.0
        or isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not 0.0 < float(alpha) < 1.0
        or payload.get("randomization_tail") != "greater"
    ):
        raise LayerwiseAnalysisError("layer-wise analysis statistical settings are invalid")
    return LayerwiseAnalysisConfig(
        analysis_id=payload["analysis_id"],
        scope=payload["scope"],
        baseline_head=payload["baseline_head"],
        comparison_heads=tuple(comparisons),
        seeds=tuple(seeds),
        expected_subject_count=payload["expected_subject_count"],
        bootstrap_iterations=payload["bootstrap_iterations"],
        bootstrap_seed=payload["bootstrap_seed"],
        confidence=float(confidence),
        randomization_tail=payload["randomization_tail"],
        holm_order=tuple(holm_order),
        alpha=float(alpha),
        sha256=_sha256_file(Path(path)),
    )


def _validate_self_hash(
    payload: Mapping[str, Any], *, field: str, description: str
) -> None:
    claimed = payload.get(field)
    body = {key: value for key, value in payload.items() if key != field}
    if not _is_sha256(claimed) or claimed != _canonical_sha256(body):
        raise LayerwiseAnalysisError(f"{description} {field} does not match content")


def _load_artifact_json(root: Path, filename: str, description: str) -> dict[str, Any]:
    return _load_json(Path(root) / filename, description)


def _validate_lock(lock: Mapping[str, Any]) -> None:
    _validate_self_hash(lock, field="lock_sha256", description="layer-wise lock")
    if lock.get("schema_version") != 1 or lock.get("status") != "sealed":
        raise LayerwiseAnalysisError("layer-wise lock is not sealed")
    if not isinstance(lock.get("study_id"), str):
        raise LayerwiseAnalysisError("layer-wise lock study identity is invalid")
    if not _is_sha256(lock.get("protocol_sha256")) or not _is_sha256(
        lock.get("training_protocol_sha256")
    ):
        raise LayerwiseAnalysisError("layer-wise lock protocol identity is invalid")
    if tuple(lock.get("seeds", ())) != EXPECTED_SEEDS:
        raise LayerwiseAnalysisError("layer-wise lock seed inventory is invalid")
    if lock.get("head_layers") != EXPECTED_HEAD_LAYERS:
        raise LayerwiseAnalysisError("layer-wise lock head inventory is invalid")
    if tuple(lock.get("layer_indices", ())) != (-4, -3, -2, -1):
        raise LayerwiseAnalysisError("layer-wise lock layer inventory is invalid")


def _validate_metrics(
    metrics: Mapping[str, Any], *, lock: Mapping[str, Any], config: LayerwiseAnalysisConfig
) -> dict[tuple[str, int], Mapping[str, Any]]:
    _validate_self_hash(
        metrics, field="metrics_sha256", description="layer-wise metrics"
    )
    if (
        metrics.get("schema_version") != 1
        or metrics.get("status") != "complete"
        or metrics.get("lock_sha256") != lock["lock_sha256"]
    ):
        raise LayerwiseAnalysisError("layer-wise metrics provenance is invalid")
    runs = metrics.get("runs")
    if not isinstance(runs, list):
        raise LayerwiseAnalysisError("layer-wise metrics runs are invalid")
    expected = {
        (head, seed)
        for head in EXPECTED_HEAD_LAYERS
        for seed in config.seeds
    }
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for run in runs:
        if not isinstance(run, Mapping):
            raise LayerwiseAnalysisError("layer-wise metric run is invalid")
        head = run.get("head_name")
        seed = run.get("seed")
        layer = run.get("layer_index")
        if (
            not isinstance(head, str)
            or head not in EXPECTED_HEAD_LAYERS
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in config.seeds
            or layer != EXPECTED_HEAD_LAYERS[head]
            or run.get("subject_count") != config.expected_subject_count
            or (head, seed) in by_key
        ):
            raise LayerwiseAnalysisError("layer-wise metric run inventory is invalid")
        for field in ("pearson", "mae", "rmse"):
            _finite_float(run.get(field), f"layer-wise metric {field}")
        by_key[(head, seed)] = run
    if set(by_key) != expected:
        raise LayerwiseAnalysisError("layer-wise metric matrix is incomplete")
    return by_key


def _validate_prediction_inventory(
    inventory: Mapping[str, Any], *, lock: Mapping[str, Any]
) -> None:
    _validate_self_hash(
        inventory,
        field="prediction_inventory_sha256",
        description="layer-wise prediction inventory",
    )
    if (
        inventory.get("schema_version") != 1
        or inventory.get("status") != "complete"
        or inventory.get("lock_sha256") != lock["lock_sha256"]
        or inventory.get("prediction_count") != 3_000
        or inventory.get("prediction_file_count") != 3_000
        or not _is_sha256(inventory.get("prediction_files_sha256"))
    ):
        raise LayerwiseAnalysisError("layer-wise prediction inventory is invalid")


def _load_prediction_rows(
    root: Path,
    *,
    lock: Mapping[str, Any],
    inventory: Mapping[str, Any],
    config: LayerwiseAnalysisConfig,
) -> list[dict[str, Any]]:
    paths = sorted((Path(root) / "predictions").rglob("*.json"))
    if len(paths) != inventory["prediction_file_count"]:
        raise LayerwiseAnalysisError("prediction matrix has an unexpected file count")
    rows: list[dict[str, Any]] = []
    expected_fields = {
        "schema_version",
        "study_id",
        "lock_sha256",
        "protocol_sha256",
        "training_protocol_sha256",
        "training_source_sha256",
        "head_name",
        "layer_index",
        "seed",
        "subject_id",
        "true_age",
        "prediction",
        "body_sha256",
    }
    for path in paths:
        row = _load_json(path, "layer-wise prediction")
        if set(row) != expected_fields:
            raise LayerwiseAnalysisError("layer-wise prediction fields are invalid")
        _validate_self_hash(row, field="body_sha256", description="prediction")
        head = row.get("head_name")
        seed = row.get("seed")
        if (
            row.get("schema_version") != 1
            or row.get("study_id") != lock["study_id"]
            or row.get("lock_sha256") != lock["lock_sha256"]
            or row.get("protocol_sha256") != lock["protocol_sha256"]
            or row.get("training_protocol_sha256") != lock["training_protocol_sha256"]
            or row.get("training_source_sha256") != lock.get("training_source_sha256")
            or head not in EXPECTED_HEAD_LAYERS
            or row.get("layer_index") != EXPECTED_HEAD_LAYERS.get(head)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in config.seeds
            or not isinstance(row.get("subject_id"), str)
            or not row["subject_id"]
        ):
            raise LayerwiseAnalysisError("layer-wise prediction provenance is invalid")
        _finite_float(row.get("true_age"), "layer-wise true age")
        _finite_float(row.get("prediction"), "layer-wise prediction")
        rows.append(row)
    actual_files_digest = _canonical_sha256(
        sorted(_canonical_sha256(row) for row in rows)
    )
    if actual_files_digest != inventory["prediction_files_sha256"]:
        raise LayerwiseAnalysisError("prediction matrix digest does not match inventory")
    return rows


def _build_series(
    rows: list[dict[str, Any]],
    *,
    config: LayerwiseAnalysisConfig,
    lock: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, dict[int, PredictionSeries]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["head_name"], row["seed"]), []).append(row)
    expected_keys = {
        (head, seed)
        for head in EXPECTED_HEAD_LAYERS
        for seed in config.seeds
    }
    if set(grouped) != expected_keys or any(
        len(items) != config.expected_subject_count for items in grouped.values()
    ):
        raise LayerwiseAnalysisError("prediction matrix is incomplete")
    provenance = {
        "lock_sha256": lock["lock_sha256"],
        "protocol_sha256": lock["protocol_sha256"],
        "training_protocol_sha256": lock["training_protocol_sha256"],
        "prediction_inventory_sha256": inventory["prediction_inventory_sha256"],
    }
    series: dict[str, dict[int, PredictionSeries]] = {}
    reference_subjects: tuple[str, ...] | None = None
    reference_targets: tuple[float, ...] | None = None
    for (head, seed), items in grouped.items():
        ordered = sorted(items, key=lambda item: item["subject_id"].encode("utf-8"))
        subject_ids = tuple(item["subject_id"] for item in ordered)
        targets = tuple(float(item["true_age"]) for item in ordered)
        if len(set(subject_ids)) != config.expected_subject_count:
            raise LayerwiseAnalysisError("prediction matrix contains duplicate subjects")
        if reference_subjects is None:
            reference_subjects = subject_ids
            reference_targets = targets
        elif subject_ids != reference_subjects or targets != reference_targets:
            raise LayerwiseAnalysisError("prediction matrix subject or target order differs")
        series.setdefault(head, {})[seed] = PredictionSeries(
            seed=seed,
            subject_ids=subject_ids,
            targets=targets,
            predictions=tuple(float(item["prediction"]) for item in ordered),
            provenance=provenance,
        )
    return series


def _validate_metrics_against_predictions(
    metrics: Mapping[tuple[str, int], Mapping[str, Any]],
    series: Mapping[str, Mapping[int, PredictionSeries]],
) -> None:
    for (head, seed), stored in metrics.items():
        candidate = regression_metrics(
            series[head][seed].targets,
            series[head][seed].predictions,
        )
        for field in ("pearson", "mae", "rmse"):
            if not math.isclose(
                float(stored[field]),
                float(candidate[field]),
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise LayerwiseAnalysisError(
                    f"external metrics differ from predictions for {head}, seed {seed}"
                )


def analyze_layerwise_artifacts(
    *,
    artifact_root: Path,
    config: LayerwiseAnalysisConfig,
    expected_protocol_sha256: str | None = None,
    expected_training_protocol_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate immutable predictions and calculate the exploratory report."""

    root = Path(artifact_root)
    lock = _load_artifact_json(root, "layerwise_external_lock.json", "layer-wise lock")
    _validate_lock(lock)
    if expected_protocol_sha256 is not None and lock["protocol_sha256"] != expected_protocol_sha256:
        raise LayerwiseAnalysisError("layer-wise protocol hash differs from lock")
    if (
        expected_training_protocol_sha256 is not None
        and lock["training_protocol_sha256"] != expected_training_protocol_sha256
    ):
        raise LayerwiseAnalysisError("layer-wise training protocol hash differs from lock")
    metrics = _load_artifact_json(root, "external_metrics.json", "layer-wise metrics")
    metric_runs = _validate_metrics(metrics, lock=lock, config=config)
    inventory = _load_artifact_json(
        root, "prediction_inventory.json", "layer-wise prediction inventory"
    )
    _validate_prediction_inventory(inventory, lock=lock)
    rows = _load_prediction_rows(root, lock=lock, inventory=inventory, config=config)
    series = _build_series(rows, config=config, lock=lock, inventory=inventory)
    _validate_metrics_against_predictions(metric_runs, series)

    baseline = series[config.baseline_head]
    comparisons: dict[str, dict[str, Any]] = {}
    p_values: dict[str, float] = {}
    for head in config.comparison_heads:
        paired = paired_seed_statistics(series[head], baseline)
        bootstrap = hierarchical_paired_bootstrap(
            series[head],
            baseline,
            iterations=config.bootstrap_iterations,
            seed=config.bootstrap_seed,
            confidence=config.confidence,
        )
        randomization = exact_seed_randomization(
            [row["pearson_delta"] for row in paired["per_seed"]]
        )
        comparisons[head] = {
            "layer_index": EXPECTED_HEAD_LAYERS[head],
            "paired": paired,
            "bootstrap": bootstrap,
            "randomization": randomization,
        }
        p_values[head] = float(randomization["p_value"])
    adjusted = holm_step_down(p_values, order=config.holm_order)
    for head, comparison in comparisons.items():
        comparison["holm_adjusted_p_value"] = adjusted[head]

    body = {
        "schema_version": 1,
        "status": "complete",
        "scope": config.scope,
        "analysis_id": config.analysis_id,
        "analysis_config_sha256": config.sha256,
        "lock_sha256": lock["lock_sha256"],
        "metrics_sha256": metrics["metrics_sha256"],
        "prediction_inventory_sha256": inventory["prediction_inventory_sha256"],
        "protocol_sha256": lock["protocol_sha256"],
        "training_protocol_sha256": lock["training_protocol_sha256"],
        "subject_count": config.expected_subject_count,
        "seeds": list(config.seeds),
        "head_layers": dict(EXPECTED_HEAD_LAYERS),
        "baseline_head": config.baseline_head,
        "comparison_order": list(config.comparison_heads),
        "comparisons": comparisons,
        "interpretation": (
            "Exploratory paired layer-wise comparison; p-values and intervals do not "
            "replace the primary confirmatory decision rule."
        ),
    }
    return {**body, "analysis_sha256": _canonical_sha256(body)}
