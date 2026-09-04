"""Audit and summarize NeuroBench evidence directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..core.evidence import (
    SCHEMA_VERSION,
    add_declared_head_bucket,
    sha256_json,
    validate_parameter_buckets,
    write_json_atomic,
)
from ..core.predictions import PredictionEvidenceError, compute_regression_metrics, read_prediction_jsonl


BOOTSTRAP_SEED = 20260903
BOOTSTRAP_ITERATIONS = 10_000
SELECTION_SCORE_TOLERANCE = 1e-8


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _prediction_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[float, float]]:
    return {
        str(row["subject_id"]): (float(row["true_age"]), float(row["predicted_age"]))
        for row in rows
    }


def audit_run(run_dir: Path) -> dict[str, Any]:
    """Validate one run and return normalized data for downstream analysis."""

    run_dir = Path(run_dir)
    errors: list[str] = []
    missing: list[str] = []
    warnings: list[str] = []
    manifest: dict[str, Any] = {}
    complexity: dict[str, Any] = {}
    try:
        manifest = _read_json(run_dir / "run_manifest.json")
    except ValueError as error:
        errors.append(str(error))
    try:
        complexity = _read_json(run_dir / "complexity.json")
    except ValueError as error:
        errors.append(str(error))

    for name, payload in (("run_manifest.json", manifest), ("complexity.json", complexity)):
        if payload and payload.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"{name} has schema_version {payload.get('schema_version')!r}")
    if manifest and manifest.get("status") != "completed":
        errors.append(
            f"run_manifest.json has status {manifest.get('status')!r}, expected 'completed'"
        )
    reported_missing = manifest.get("missing") if manifest else None
    if reported_missing:
        if isinstance(reported_missing, list):
            errors.append(
                "run_manifest.json reports missing evidence: "
                + "; ".join(str(item) for item in reported_missing)
            )
        else:
            errors.append("run_manifest.json has an invalid missing-evidence list")

    git_metadata = manifest.get("git") if manifest else None
    source_tree_digest = manifest.get("source_tree_sha256") if manifest else None
    if source_tree_digest is None and isinstance(git_metadata, Mapping):
        source_tree_digest = git_metadata.get("source_tree_sha256")
    if not isinstance(source_tree_digest, str) or not source_tree_digest:
        errors.append("run_manifest.json is missing source tree SHA-256 provenance")
    protocol_digest = manifest.get("protocol_digest") if manifest else None
    if not isinstance(protocol_digest, str) or not protocol_digest:
        errors.append("run_manifest.json is missing protocol digest")
    comparison_config_digest = manifest.get("comparison_config_hash") if manifest else None
    if not isinstance(comparison_config_digest, str) or not comparison_config_digest:
        errors.append("run_manifest.json is missing comparison config hash")
    comparison_factor_keys = manifest.get("comparison_factor_keys") if manifest else None
    if not isinstance(comparison_factor_keys, list) or any(
        not isinstance(key, str) or not key for key in comparison_factor_keys
    ):
        errors.append("run_manifest.json has an invalid comparison factor key list")
    deterministic_policy = manifest.get("deterministic_policy") if manifest else None
    if deterministic_policy not in {"strict", "best_effort"}:
        errors.append("run_manifest.json has an invalid deterministic policy")
    elif deterministic_policy == "strict":
        if manifest.get("deterministic_policy_satisfied") is not True:
            errors.append("strict deterministic policy is not satisfied")
    else:
        warnings.append("run uses best-effort, not strict bitwise determinism")

    buckets = complexity.get("parameter_buckets") if complexity else None
    head_complexity = complexity.get("head_complexity") if complexity else None
    declared_head_count = (
        head_complexity.get("parameter_count")
        if isinstance(head_complexity, Mapping)
        else None
    )
    raw_head_bucket = buckets.get("head") if isinstance(buckets, Mapping) else None
    raw_head_total = raw_head_bucket.get("total") if isinstance(raw_head_bucket, Mapping) else None
    if (
        isinstance(buckets, Mapping)
        and isinstance(declared_head_count, int)
        and not isinstance(declared_head_count, bool)
        and declared_head_count > 0
        and raw_head_total == 0
    ):
        try:
            buckets = add_declared_head_bucket(
                buckets,
                parameter_count=declared_head_count,
            )
        except (KeyError, ValueError) as error:
            errors.append(f"external head parameter accounting: {error}")
        else:
            warnings.append(
                "parameter buckets normalized using the external head complexity contract"
            )
    if isinstance(buckets, Mapping):
        try:
            validate_parameter_buckets(buckets)
        except (KeyError, ValueError) as error:
            errors.append(f"parameter accounting: {error}")
    else:
        missing.append("complexity.parameter_buckets")
    if isinstance(head_complexity, Mapping) and "parameter_count" in head_complexity:
        declared_head_count = head_complexity.get("parameter_count")
        measured_head_count = buckets.get("head", {}).get("total") if isinstance(buckets, Mapping) else None
        if (
            isinstance(declared_head_count, bool)
            or not isinstance(declared_head_count, int)
            or declared_head_count < 0
        ):
            errors.append("head complexity has an invalid parameter_count")
        elif measured_head_count != declared_head_count:
            errors.append(
                "head complexity parameter count does not match parameter bucket: "
                f"declared={declared_head_count} measured={measured_head_count}"
            )

    prediction_path = run_dir / "predictions" / "validation.jsonl"
    rows: list[dict[str, Any]] = []
    if not prediction_path.is_file():
        missing.append("predictions/validation.jsonl")
    else:
        try:
            rows = read_prediction_jsonl(prediction_path, expected_split="validation")
            if len(rows) < 2:
                errors.append("validation predictions require at least two subjects")
        except PredictionEvidenceError as error:
            errors.append(f"validation predictions: {error}")

    metrics: dict[str, Any] = {}
    if rows and not errors:
        try:
            metrics = compute_regression_metrics(
                [row["true_age"] for row in rows],
                [row["predicted_age"] for row in rows],
            )
        except PredictionEvidenceError as error:
            errors.append(f"validation metrics: {error}")

    selection: dict[str, Any] = {}
    selection_path = run_dir / "selection.json"
    if selection_path.is_file():
        try:
            selection = _read_json(selection_path)
        except ValueError as error:
            errors.append(str(error))
    elif manifest.get("evaluation_mode") in {"validation_only", "final_test"}:
        missing.append("selection.json")

    if selection:
        selected_epoch = selection.get("selected_epoch")
        if isinstance(selected_epoch, bool) or not isinstance(selected_epoch, int) or selected_epoch < 1:
            errors.append("selection.json has an invalid selected_epoch")
        selected_score = selection.get("selected_val_pearsonr")
        selected_score_valid = (
            isinstance(selected_score, (int, float))
            and not isinstance(selected_score, bool)
            and math.isfinite(float(selected_score))
        )
        if isinstance(selected_score, bool) or not isinstance(selected_score, (int, float)):
            errors.append("selection.json has an invalid selected_val_pearsonr")
        elif not selected_score_valid:
            errors.append("selection.json has a non-finite selected_val_pearsonr")
        validation_history_path = run_dir / "epoch_validation_metrics.jsonl"
        if not validation_history_path.is_file():
            missing.append("epoch_validation_metrics.jsonl")
        elif isinstance(selected_epoch, int) and not isinstance(selected_epoch, bool):
            try:
                history_rows = jsonl_rows(validation_history_path)
            except (OSError, json.JSONDecodeError, ValueError) as error:
                errors.append(f"validation history: {error}")
            else:
                matching_rows = [row for row in history_rows if row.get("epoch") == selected_epoch]
                if len(matching_rows) != 1:
                    errors.append("selection epoch does not match exactly one validation history record")
                else:
                    history_score = matching_rows[0].get("val/pearsonr")
                    if selected_score_valid and isinstance(history_score, (int, float)) and not isinstance(history_score, bool):
                        if not math.isclose(
                            float(history_score), float(selected_score),
                            rel_tol=SELECTION_SCORE_TOLERANCE,
                            abs_tol=SELECTION_SCORE_TOLERANCE,
                        ):
                            errors.append("selection Pearson does not match validation history")
                    else:
                        errors.append("selected validation history record has an invalid Pearson score")

    test_marker: dict[str, Any] = {}
    test_rows: list[dict[str, Any]] = []
    test_metrics: dict[str, Any] = {}
    official_test_pearsonr: float | None = None
    test_completed_path = run_dir / "test_completed.json"
    test_prediction_path = run_dir / "predictions" / "test.jsonl"
    evaluation_mode = manifest.get("evaluation_mode")
    if evaluation_mode == "validation_only":
        if manifest.get("test_access") != "sealed":
            errors.append("validation_only run must declare test_access=sealed")
        for test_artifact in (
            run_dir / "test_started.json",
            test_completed_path,
            test_prediction_path,
            run_dir / "epoch_test_metrics.jsonl",
        ):
            if test_artifact.is_file():
                errors.append(
                    f"validation_only run contains test artifact: {test_artifact.name}"
                )
    elif evaluation_mode == "final_test":
        if manifest.get("test_access") != "single_use_predeclared":
            errors.append("final_test run must declare test_access=single_use_predeclared")
        if not (run_dir / "test_started.json").is_file():
            missing.append("test_started.json")
    if test_completed_path.is_file():
        try:
            test_marker = _read_json(test_completed_path)
            if test_marker.get("schema_version") != SCHEMA_VERSION:
                errors.append("test_completed.json has the wrong schema_version")
            if test_marker.get("evaluation_mode") != "final_test":
                errors.append("test_completed.json is not marked as final_test")
            marker_score = test_marker.get("test_pearsonr")
            if isinstance(marker_score, bool) or not isinstance(marker_score, (int, float)):
                errors.append("test_completed.json has an invalid test_pearsonr")
            elif not math.isfinite(float(marker_score)):
                errors.append("test_completed.json has a non-finite test_pearsonr")
            else:
                official_test_pearsonr = float(marker_score)
        except ValueError as error:
            errors.append(str(error))
        if not test_prediction_path.is_file():
            missing.append("predictions/test.jsonl")
        else:
            try:
                test_rows = read_prediction_jsonl(test_prediction_path, expected_split="test")
                test_metrics = compute_regression_metrics(
                    [row["true_age"] for row in test_rows],
                    [row["predicted_age"] for row in test_rows],
                )
                export_payload = test_marker.get("prediction_export")
                export_metrics = (
                    export_payload.get("metrics")
                    if isinstance(export_payload, Mapping)
                    else None
                )
                export_score = (
                    export_metrics.get("pearsonr")
                    if isinstance(export_metrics, Mapping)
                    else None
                )
                if export_score is not None:
                    if isinstance(export_score, bool) or not isinstance(export_score, (int, float)):
                        errors.append("prediction export has an invalid Pearson")
                    elif not math.isclose(
                        float(export_score),
                        float(test_metrics["pearsonr"]),
                        rel_tol=SELECTION_SCORE_TOLERANCE,
                        abs_tol=SELECTION_SCORE_TOLERANCE,
                    ):
                        errors.append("prediction export Pearson does not match test predictions")
                elif (
                    official_test_pearsonr is not None
                    and not math.isclose(
                        official_test_pearsonr,
                        float(test_metrics["pearsonr"]),
                        rel_tol=SELECTION_SCORE_TOLERANCE,
                        abs_tol=SELECTION_SCORE_TOLERANCE,
                    )
                ):
                    # Legacy artifacts did not declare a separate export metric;
                    # retain the strict consistency check for that format.
                    errors.append("test marker Pearson does not match test predictions")
            except PredictionEvidenceError as error:
                errors.append(f"test predictions: {error}")
    elif evaluation_mode == "final_test":
        missing.append("test_completed.json")

    status = "invalid" if errors else "partial" if missing else "complete"
    hardware = manifest.get("hardware") or complexity.get("hardware") or {}
    return {
        "run_dir": str(run_dir.resolve()),
        "run_id": manifest.get("run_id"),
        "status": status,
        "errors": errors,
        "missing": missing,
        "warnings": warnings,
        "seed": manifest.get("seed"),
        "task": manifest.get("task"),
        "head_variant": manifest.get("resolved_config", {}).get("head_variant")
        if isinstance(manifest.get("resolved_config"), Mapping)
        else None,
        "dataset_manifest": manifest.get("dataset_manifest"),
        "split_fingerprint": manifest.get("split_fingerprint"),
        "source_tree_sha256": source_tree_digest,
        "protocol_digest": protocol_digest,
        "comparison_config_hash": comparison_config_digest,
        "comparison_factor_keys": comparison_factor_keys,
        "deterministic_policy": deterministic_policy,
        "deterministic_policy_satisfied": manifest.get("deterministic_policy_satisfied"),
        "evaluation_mode": manifest.get("evaluation_mode"),
        "test_access": manifest.get("test_access"),
        "selected_epoch": selection.get("selected_epoch"),
        "selected_val_pearsonr": selection.get("selected_val_pearsonr"),
        "validation_metric_source": "epoch_validation_metrics",
        "validation_prediction_pearsonr": metrics.get("pearsonr"),
        "test_status": "completed" if test_marker else (
            "withheld" if manifest.get("evaluation_mode") == "validation_only" else "missing"
        ),
        "test_metrics": test_metrics,
        "official_test_pearsonr": official_test_pearsonr,
        "test_metric_source": "official_test_marker",
        "test_predictions": test_rows,
        "hardware_class": hardware.get("hardware_class"),
        "hardware_mixed": False,
        "metrics": metrics,
        "predictions": rows,
        "parameter_buckets": buckets,
        "phases": complexity.get("phases", {}),
        "throughput": complexity.get("throughput", {}),
        "memory": complexity.get("memory", {}),
        "cost_usd": complexity.get("cost_usd"),
    }


def _metric_from_pairs(pairs: Sequence[tuple[float, float]], metric: str) -> float:
    if metric not in {"pearsonr", "mae", "rmse", "r2"}:
        raise ValueError(f"unsupported metric {metric!r}")
    values = compute_regression_metrics(
        [truth for truth, _ in pairs], [prediction for _, prediction in pairs]
    )
    return float(values[metric])


def paired_bootstrap_delta(
    candidate: Mapping[str, tuple[float, float]],
    baseline: Mapping[str, tuple[float, float]],
    *,
    metric: str = "pearsonr",
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute candidate-minus-baseline paired subject bootstrap evidence."""

    candidate_subject_ids = set(candidate)
    baseline_subject_ids = set(baseline)
    if candidate_subject_ids != baseline_subject_ids:
        raise ValueError("paired bootstrap requires exactly the same subject IDs")
    subject_ids = sorted(candidate_subject_ids)
    if len(subject_ids) < 2:
        raise ValueError("paired bootstrap requires at least two matched subjects")
    if any(candidate[subject][0] != baseline[subject][0] for subject in subject_ids):
        raise ValueError("paired bootstrap requires identical true ages for matched subjects")
    candidate_pairs = [candidate[subject] for subject in subject_ids]
    baseline_pairs = [baseline[subject] for subject in subject_ids]
    observed = _metric_from_pairs(candidate_pairs, metric) - _metric_from_pairs(baseline_pairs, metric)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(int(iterations)):
        indices = rng.integers(0, len(subject_ids), size=len(subject_ids))
        candidate_sample = [candidate_pairs[int(index)] for index in indices]
        baseline_sample = [baseline_pairs[int(index)] for index in indices]
        try:
            value = _metric_from_pairs(candidate_sample, metric) - _metric_from_pairs(baseline_sample, metric)
        except PredictionEvidenceError:
            continue
        if math.isfinite(value):
            samples.append(float(value))
    if not samples:
        raise ValueError("paired bootstrap produced no finite resamples")
    values = np.asarray(samples, dtype=np.float64)
    return {
        "metric": metric,
        "delta": float(observed),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "iterations": int(iterations),
        "valid_iterations": int(len(values)),
        "seed": int(seed),
        "subject_count": len(subject_ids),
        "pairing": "same_subject_ids",
    }


def _age_group_edges(train_age_reference: Mapping[str, float]) -> tuple[float, ...]:
    ages = np.asarray(sorted(float(age) for age in train_age_reference.values()), dtype=np.float64)
    if ages.size == 0:
        raise ValueError("train age reference is empty")
    raw = np.quantile(ages, [0.25, 0.5, 0.75], method="linear")
    return tuple(float(value) for value in raw)


def compute_age_group_metrics(
    predictions: Mapping[str, tuple[float, float]],
    *,
    train_age_reference: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Compute metrics by quartile bins derived only from train subjects."""

    edges = _age_group_edges(train_age_reference)
    groups: dict[str, list[tuple[float, float]]] = {f"q{index}": [] for index in range(1, 5)}
    for subject_id, pair in predictions.items():
        if subject_id not in train_age_reference:
            # Validation/test subjects are assigned using the train-derived
            # thresholds, never used to recompute the thresholds.
            age = float(pair[0])
        else:
            age = float(train_age_reference[subject_id])
        group = "q4" if age > edges[2] else "q3" if age > edges[1] else "q2" if age > edges[0] else "q1"
        groups[group].append(pair)
    rows: list[dict[str, Any]] = []
    for group, pairs in groups.items():
        row: dict[str, Any] = {
            "age_group": group,
            "n_subjects": len(pairs),
            "lower_edge": None if group == "q1" else edges[int(group[1]) - 2],
            "upper_edge": None if group == "q4" else edges[int(group[1]) - 1],
        }
        if len(pairs) >= 2:
            row.update(_metric_from_pairs_dict(pairs))
        else:
            row.update({"pearsonr": None, "mae": None, "rmse": None, "r2": None})
        rows.append(row)
    return rows


def _metric_from_pairs_dict(pairs: Sequence[tuple[float, float]]) -> dict[str, float | int]:
    return dict(compute_regression_metrics([truth for truth, _ in pairs], [prediction for _, prediction in pairs]))


def summarize_seed_stability(
    candidate_dirs: Sequence[Path], baseline_dirs: Sequence[Path]
) -> dict[str, Any]:
    """Summarize per-seed candidate-versus-baseline validation stability."""

    matched = _matched_audits(candidate_dirs, baseline_dirs)
    shared_seeds = [seed for seed, _, _ in matched]
    per_seed: list[dict[str, Any]] = []
    for seed, candidate, baseline in matched:
        candidate_score = float(candidate["selected_val_pearsonr"])
        baseline_score = float(baseline["selected_val_pearsonr"])
        candidate_test = candidate.get("official_test_pearsonr")
        baseline_test = baseline.get("official_test_pearsonr")
        per_seed.append(
            {
                "seed": seed,
                "candidate_pearsonr": candidate_score,
                "baseline_pearsonr": baseline_score,
                "candidate_selected_val_pearsonr": candidate.get("selected_val_pearsonr"),
                "baseline_selected_val_pearsonr": baseline.get("selected_val_pearsonr"),
                "candidate_validation_prediction_pearsonr": candidate.get("validation_prediction_pearsonr"),
                "baseline_validation_prediction_pearsonr": baseline.get("validation_prediction_pearsonr"),
                "candidate_test_pearsonr": candidate_test,
                "baseline_test_pearsonr": baseline_test,
                "candidate_test_prediction_pearsonr": candidate.get("test_metrics", {}).get("pearsonr"),
                "baseline_test_prediction_pearsonr": baseline.get("test_metrics", {}).get("pearsonr"),
                "pearsonr_delta": candidate_score - baseline_score,
                "candidate_status": candidate["status"],
                "baseline_status": baseline["status"],
            }
        )
    deltas = np.asarray([row["pearsonr_delta"] for row in per_seed], dtype=np.float64)
    scores = np.asarray([row["candidate_pearsonr"] for row in per_seed], dtype=np.float64)
    hardware_classes = {
        audit["hardware_class"]
        for _, candidate, baseline in matched
        for audit in (candidate, baseline)
        if audit.get("hardware_class") is not None
    }
    return {
        "seed_count": len(per_seed),
        "seeds": shared_seeds,
        "per_seed": per_seed,
        "pearsonr_mean": float(scores.mean()),
        "pearsonr_sample_sd": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "pearsonr_wins": int((deltas > 0).sum()),
        "worst_seed_delta": float(deltas.min()),
        "hardware_mixed": len(hardware_classes) > 1,
        "hardware_classes": sorted(hardware_classes),
    }


def _audits_by_seed(run_dirs: Sequence[Path], group_name: str) -> dict[int, dict[str, Any]]:
    audits_by_seed: dict[int, dict[str, Any]] = {}
    for path in run_dirs:
        audit = audit_run(Path(path))
        if audit["status"] != "complete":
            raise ValueError(
                f"{group_name} run {audit['run_dir']} is not complete: "
                f"errors={audit['errors']} missing={audit['missing']}"
            )
        seed = audit.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{group_name} run {audit['run_dir']} has an invalid seed")
        if seed in audits_by_seed:
            raise ValueError(f"{group_name} group contains duplicate seed {seed}")
        audits_by_seed[seed] = audit
    return audits_by_seed


def _validate_matched_pair(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any], *, seed: int
) -> None:
    for field in (
        "task",
        "dataset_manifest",
        "split_fingerprint",
        "source_tree_sha256",
        "protocol_digest",
        "comparison_config_hash",
        "comparison_factor_keys",
        "evaluation_mode",
        "test_access",
    ):
        if candidate.get(field) != baseline.get(field):
            label = {
                "comparison_config_hash": "comparison config",
                "comparison_factor_keys": "comparison factor declaration",
                "protocol_digest": "protocol digest",
                "source_tree_sha256": "source tree",
            }.get(field, field)
            raise ValueError(
                f"matched comparison has different {label} for seed {seed}: "
                f"candidate={candidate.get(field)!r} baseline={baseline.get(field)!r}"
            )
    candidate_map = _prediction_map(candidate.get("predictions", []))
    baseline_map = _prediction_map(baseline.get("predictions", []))
    if set(candidate_map) != set(baseline_map):
        raise ValueError(
            f"matched comparison requires exactly the same validation subject IDs for seed {seed}"
        )
    for subject_id in sorted(candidate_map):
        if candidate_map[subject_id][0] != baseline_map[subject_id][0]:
            raise ValueError(
                f"matched comparison requires identical true ages for subject {subject_id!r}"
            )


def _matched_audits(
    candidate_dirs: Sequence[Path], baseline_dirs: Sequence[Path]
) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    candidates = _audits_by_seed(candidate_dirs, "candidate")
    baselines = _audits_by_seed(baseline_dirs, "baseline")
    shared_seeds = sorted(set(candidates) & set(baselines))
    if not shared_seeds:
        raise ValueError("candidate and baseline have no shared seeds")
    matched: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for seed in shared_seeds:
        candidate = candidates[seed]
        baseline = baselines[seed]
        _validate_matched_pair(candidate, baseline, seed=seed)
        matched.append((seed, candidate, baseline))
    return matched


def complexity_adjusted_comparison(
    candidate_dirs: Sequence[Path], baseline_dirs: Sequence[Path]
) -> list[dict[str, Any]]:
    """Compare matched seeds while exposing the raw parameter denominator."""

    rows: list[dict[str, Any]] = []
    for seed, candidate, baseline in _matched_audits(candidate_dirs, baseline_dirs):
        candidate_head = (candidate.get("parameter_buckets") or {}).get("head", {})
        baseline_head = (baseline.get("parameter_buckets") or {}).get("head", {})
        candidate_parameters = candidate_head.get("total")
        baseline_parameters = baseline_head.get("total")
        denominator = None
        ratio = None
        if isinstance(candidate_parameters, int) and isinstance(baseline_parameters, int):
            denominator = candidate_parameters - baseline_parameters
            if denominator > 0:
                ratio = (
                    float(candidate["selected_val_pearsonr"])
                    - float(baseline["selected_val_pearsonr"])
                ) / denominator
        rows.append(
            {
                "seed": seed,
                "candidate_head_parameters": candidate_parameters,
                "baseline_head_parameters": baseline_parameters,
                "head_parameter_delta": denominator,
                "pearsonr_delta": (
                    float(candidate["selected_val_pearsonr"])
                    - float(baseline["selected_val_pearsonr"])
                ),
                "delta_per_extra_head_parameter": ratio,
                "hardware_mixed": candidate.get("hardware_class") != baseline.get("hardware_class"),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def analyze_runs(
    run_dirs: Sequence[Path],
    *,
    output_dir: Path,
    train_age_reference: Path | None = None,
    baseline_variant: str = "mean_linear",
    baseline_run_dirs: Sequence[Path] = (),
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Write deterministic analysis tables, figures, and an analysis manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    audits = [audit_run(Path(path)) for path in run_dirs]
    for audit in audits:
        if audit["status"] != "complete":
            raise ValueError(
                f"run {audit['run_dir']} is not complete: "
                f"errors={audit['errors']} missing={audit['missing']}"
            )

    reference: dict[str, float] = {}
    reference_source_sha256: str | None = None
    if train_age_reference is not None:
        reference_path = Path(train_age_reference)
        reference_source_sha256 = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        for row in jsonl_rows(reference_path):
            subject = row.get("subject_id")
            reference[str(subject)] = float(row["true_age"])
    elif audits:
        for audit in audits:
            candidate_path = Path(audit["run_dir"]) / "analysis" / "train_age_reference.jsonl"
            if candidate_path.is_file():
                reference_source_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
                for row in jsonl_rows(candidate_path):
                    reference[str(row["subject_id"])] = float(row["true_age"])
                break
    analysis_spec = {
        "schema_version": SCHEMA_VERSION,
        "bootstrap": {"iterations": bootstrap_iterations, "seed": seed, "confidence": 0.95},
        "age_groups": {"method": "train_only_quartiles", "edges": _age_group_edges(reference) if reference else None},
        "train_age_reference_sha256": reference_source_sha256,
        "aggregation": "arithmetic_mean_per_subject",
    }
    analysis_spec["analysis_spec_hash"] = sha256_json(analysis_spec)
    write_json_atomic(output_dir / "analysis_spec.json", analysis_spec)

    normalized_rows: list[dict[str, Any]] = []
    age_rows: list[dict[str, Any]] = []
    for audit in audits:
        metrics = dict(audit.get("metrics", {}))
        normalized_rows.append(
            {
                "run_id": audit.get("run_id"),
                "run_dir": audit["run_dir"],
                "seed": audit.get("seed"),
                "head_variant": audit.get("head_variant"),
                "hardware_class": audit.get("hardware_class"),
                "hardware_mixed": audit.get("hardware_mixed"),
                "selected_val_pearsonr": audit.get("selected_val_pearsonr"),
                "selection_metric_source": audit.get("validation_metric_source"),
                "test_status": audit.get("test_status"),
                "test_pearsonr": audit.get("official_test_pearsonr"),
                "test_metric_source": audit.get("test_metric_source"),
                "test_prediction_pearsonr": (audit.get("test_metrics") or {}).get("pearsonr"),
                "audit_warnings": "; ".join(audit.get("warnings", [])),
                **metrics,
            }
        )
        if reference:
            for split_name, split_rows in (
                ("validation", audit["predictions"]),
                ("test", audit.get("test_predictions", [])),
            ):
                if not split_rows:
                    continue
                age_rows.extend(
                    {
                        "run_id": audit.get("run_id"),
                        "seed": audit.get("seed"),
                        "split": split_name,
                        **row,
                    }
                    for row in compute_age_group_metrics(
                        _prediction_map(split_rows),
                        train_age_reference=reference,
                    )
                )
    _write_csv(output_dir / "per_run_metrics.csv", normalized_rows)
    _write_csv(output_dir / "age_group_metrics.csv", age_rows)

    complexity_rows = []
    for audit in audits:
        buckets = audit.get("parameter_buckets") or {}
        head = buckets.get("head", {}) if isinstance(buckets, Mapping) else {}
        throughput = audit.get("throughput", {})
        if not isinstance(throughput, Mapping):
            throughput = {}
        phases = audit.get("phases", {})
        fit_phase = (
            phases.get("fit_and_evaluation", phases.get("train", {}))
            if isinstance(phases, Mapping)
            else {}
        )
        complexity_rows.append(
            {
                "run_id": audit.get("run_id"),
                "seed": audit.get("seed"),
                "head_variant": audit.get("head_variant"),
                "head_parameters": head.get("total"),
                "head_trainable_parameters": head.get("trainable"),
                "train_seconds": throughput.get("elapsed_seconds") or fit_phase.get("elapsed_seconds"),
                "train_batches": throughput.get("batches"),
                "train_samples": throughput.get("samples"),
                "samples_per_second": throughput.get("samples_per_second"),
                "peak_allocated_mb": (audit.get("memory") or {}).get("peak_allocated_mb"),
                "cost_usd": audit.get("cost_usd"),
                "hardware_class": audit.get("hardware_class"),
            }
        )
    _write_csv(output_dir / "complexity_metrics.csv", complexity_rows)

    comparison: dict[str, Any] | None = None
    if baseline_run_dirs:
        comparison = summarize_seed_stability(run_dirs, baseline_run_dirs)
        comparison["baseline_variant"] = baseline_variant
        comparison["complexity"] = complexity_adjusted_comparison(run_dirs, baseline_run_dirs)
        matched = _matched_audits(run_dirs, baseline_run_dirs)
        bootstrap_rows: dict[str, list[dict[str, Any]]] = {}
        for split_name, candidate_key, baseline_key in (
            ("validation", "predictions", "predictions"),
            ("test", "test_predictions", "test_predictions"),
        ):
            rows_for_split: list[dict[str, Any]] = []
            for seed_value, candidate_audit, baseline_audit in matched:
                if not candidate_audit.get(candidate_key) or not baseline_audit.get(baseline_key):
                    continue
                candidate_map = _prediction_map(candidate_audit[candidate_key])
                baseline_map = _prediction_map(baseline_audit[baseline_key])
                metric_results: dict[str, Any] = {}
                for metric_name in ("pearsonr", "mae", "rmse", "r2"):
                    try:
                        metric_results[metric_name] = paired_bootstrap_delta(
                            candidate_map,
                            baseline_map,
                            metric=metric_name,
                            iterations=bootstrap_iterations,
                            seed=seed,
                        )
                    except ValueError as error:
                        metric_results[metric_name] = {
                            "status": "unavailable",
                            "reason": str(error),
                        }
                rows_for_split.append(
                    {
                        "seed": seed_value,
                        "candidate_run_id": candidate_audit.get("run_id"),
                        "baseline_run_id": baseline_audit.get("run_id"),
                        "metrics": metric_results,
                    }
                )
            if rows_for_split:
                bootstrap_rows[split_name] = rows_for_split
        comparison["paired_bootstrap"] = bootstrap_rows
        write_json_atomic(output_dir / "candidate_vs_baseline.json", comparison)
        _write_csv(output_dir / "complexity_comparison.csv", comparison["complexity"])

    figure_paths: list[Path] = []
    try:
        import matplotlib.pyplot as plt

        if audits and all(audit["predictions"] for audit in audits):
            figure, axis = plt.subplots(figsize=(6, 5))
            for audit in audits:
                rows = audit["predictions"]
                axis.scatter(
                    [row["true_age"] for row in rows],
                    [row["predicted_age"] for row in rows],
                    label=f"{audit.get('head_variant')} seed{audit.get('seed')}",
                    alpha=0.7,
                )
            axis.set_xlabel("True age")
            axis.set_ylabel("Predicted age")
            axis.legend(fontsize="small")
            figure.tight_layout()
            path = output_dir / "predicted_vs_true.png"
            figure.savefig(path, dpi=160)
            plt.close(figure)
            figure_paths.append(path)

            figure, axis = plt.subplots(figsize=(6, 5))
            for audit in audits:
                rows = audit["predictions"]
                axis.scatter(
                    [row["true_age"] for row in rows],
                    [row["predicted_age"] - row["true_age"] for row in rows],
                    label=f"{audit.get('head_variant')} seed{audit.get('seed')}",
                    alpha=0.7,
                )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_xlabel("True age")
            axis.set_ylabel("Residual (predicted − true)")
            axis.legend(fontsize="small")
            figure.tight_layout()
            path = output_dir / "residuals.png"
            figure.savefig(path, dpi=160)
            plt.close(figure)
            figure_paths.append(path)
    except Exception:
        pass

    output_files = [path for path in output_dir.iterdir() if path.is_file() and path.name != "analysis_manifest.json"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input_run_ids": [audit.get("run_id") for audit in audits],
        "input_run_dirs": [audit["run_dir"] for audit in audits],
        "analysis_spec_hash": analysis_spec["analysis_spec_hash"],
        "command_line": shlex.join([sys.executable, *sys.argv]),
        "rng_seed": seed,
        "bootstrap_iterations": bootstrap_iterations,
        "outputs": {
            str(path.relative_to(output_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output_files)
        },
    }
    write_json_atomic(output_dir / "analysis_manifest.json", manifest)
    return {"analysis_manifest": manifest, "audits": audits, "analysis_spec": analysis_spec}


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(dict(payload))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-age-reference", type=Path)
    parser.add_argument("--baseline-variant", default="mean_linear")
    parser.add_argument("--baseline-run", type=Path, action="append", default=[])
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args(argv)
    result = analyze_runs(
        args.run_dirs,
        output_dir=args.output_dir,
        train_age_reference=args.train_age_reference,
        baseline_variant=args.baseline_variant,
        baseline_run_dirs=args.baseline_run,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    print(json.dumps(result["analysis_manifest"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
