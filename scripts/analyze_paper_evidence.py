"""Audit and summarize article-ready NeuroBench evidence directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_evidence import SCHEMA_VERSION, sha256_json, validate_parameter_buckets, write_json_atomic
from prediction_evidence import PredictionEvidenceError, compute_regression_metrics, read_prediction_jsonl


BOOTSTRAP_SEED = 20260903
BOOTSTRAP_ITERATIONS = 10_000


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

    buckets = complexity.get("parameter_buckets") if complexity else None
    if isinstance(buckets, Mapping):
        try:
            validate_parameter_buckets(buckets)
        except (KeyError, ValueError) as error:
            errors.append(f"parameter accounting: {error}")
    else:
        missing.append("complexity.parameter_buckets")

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
    elif manifest.get("evaluation_mode") == "validation_only":
        missing.append("selection.json")

    test_marker: dict[str, Any] = {}
    test_rows: list[dict[str, Any]] = []
    test_metrics: dict[str, Any] = {}
    test_completed_path = run_dir / "test_completed.json"
    test_prediction_path = run_dir / "predictions" / "test.jsonl"
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
                marker_score = test_marker.get("test_pearsonr")
                if isinstance(marker_score, (int, float)) and abs(
                    float(marker_score) - float(test_metrics["pearsonr"])
                ) > 1e-10:
                    errors.append("test marker Pearson does not match test predictions")
            except PredictionEvidenceError as error:
                errors.append(f"test predictions: {error}")
    elif manifest.get("evaluation_mode") == "final_test":
        missing.append("test_completed.json")

    status = "invalid" if errors else "partial" if missing else "complete"
    hardware = manifest.get("hardware") or complexity.get("hardware") or {}
    return {
        "run_dir": str(run_dir.resolve()),
        "run_id": manifest.get("run_id"),
        "status": status,
        "errors": errors,
        "missing": missing,
        "seed": manifest.get("seed"),
        "head_variant": manifest.get("resolved_config", {}).get("head_variant")
        if isinstance(manifest.get("resolved_config"), Mapping)
        else None,
        "dataset_manifest": manifest.get("dataset_manifest"),
        "split_fingerprint": manifest.get("split_fingerprint"),
        "evaluation_mode": manifest.get("evaluation_mode"),
        "test_access": manifest.get("test_access"),
        "selected_epoch": selection.get("selected_epoch"),
        "selected_val_pearsonr": selection.get("selected_val_pearsonr"),
        "test_status": "completed" if test_marker else (
            "withheld" if manifest.get("evaluation_mode") == "validation_only" else "missing"
        ),
        "test_metrics": test_metrics,
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

    subject_ids = sorted(set(candidate) & set(baseline))
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

    candidate_audits = [audit_run(Path(path)) for path in candidate_dirs]
    baseline_audits = [audit_run(Path(path)) for path in baseline_dirs]
    candidate_by_seed = {audit["seed"]: audit for audit in candidate_audits}
    baseline_by_seed = {audit["seed"]: audit for audit in baseline_audits}
    shared_seeds = sorted(set(candidate_by_seed) & set(baseline_by_seed))
    if not shared_seeds:
        raise ValueError("candidate and baseline have no shared seeds")
    per_seed: list[dict[str, Any]] = []
    for seed in shared_seeds:
        candidate = candidate_by_seed[seed]
        baseline = baseline_by_seed[seed]
        candidate_score = float(candidate["metrics"]["pearsonr"])
        baseline_score = float(baseline["metrics"]["pearsonr"])
        candidate_test = candidate.get("test_metrics", {}).get("pearsonr")
        baseline_test = baseline.get("test_metrics", {}).get("pearsonr")
        per_seed.append(
            {
                "seed": seed,
                "candidate_pearsonr": candidate_score,
                "baseline_pearsonr": baseline_score,
                "candidate_selected_val_pearsonr": candidate.get("selected_val_pearsonr"),
                "baseline_selected_val_pearsonr": baseline.get("selected_val_pearsonr"),
                "candidate_test_pearsonr": candidate_test,
                "baseline_test_pearsonr": baseline_test,
                "pearsonr_delta": candidate_score - baseline_score,
                "candidate_status": candidate["status"],
                "baseline_status": baseline["status"],
            }
        )
    deltas = np.asarray([row["pearsonr_delta"] for row in per_seed], dtype=np.float64)
    scores = np.asarray([row["candidate_pearsonr"] for row in per_seed], dtype=np.float64)
    hardware_classes = {
        audit["hardware_class"]
        for audit in candidate_audits + baseline_audits
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


def complexity_adjusted_comparison(
    candidate_dirs: Sequence[Path], baseline_dirs: Sequence[Path]
) -> list[dict[str, Any]]:
    """Compare matched seeds while exposing the raw parameter denominator."""

    candidates = {audit["seed"]: audit for audit in (audit_run(Path(path)) for path in candidate_dirs)}
    baselines = {audit["seed"]: audit for audit in (audit_run(Path(path)) for path in baseline_dirs)}
    rows: list[dict[str, Any]] = []
    for seed in sorted(set(candidates) & set(baselines)):
        candidate = candidates[seed]
        baseline = baselines[seed]
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
                    float(candidate["metrics"]["pearsonr"])
                    - float(baseline["metrics"]["pearsonr"])
                ) / denominator
        rows.append(
            {
                "seed": seed,
                "candidate_head_parameters": candidate_parameters,
                "baseline_head_parameters": baseline_parameters,
                "head_parameter_delta": denominator,
                "pearsonr_delta": (
                    float(candidate["metrics"]["pearsonr"])
                    - float(baseline["metrics"]["pearsonr"])
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
        if audit["status"] == "invalid":
            raise ValueError(f"invalid run {audit['run_dir']}: {audit['errors']}")

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
                "test_status": audit.get("test_status"),
                "test_pearsonr": (audit.get("test_metrics") or {}).get("pearsonr"),
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
        baseline_audits = [audit_run(Path(path)) for path in baseline_run_dirs]
        candidate_by_seed = {audit.get("seed"): audit for audit in audits}
        baseline_by_seed = {audit.get("seed"): audit for audit in baseline_audits}
        bootstrap_rows: dict[str, list[dict[str, Any]]] = {}
        for split_name, candidate_key, baseline_key in (
            ("validation", "predictions", "predictions"),
            ("test", "test_predictions", "test_predictions"),
        ):
            rows_for_split: list[dict[str, Any]] = []
            for seed_value in sorted(set(candidate_by_seed) & set(baseline_by_seed)):
                candidate_audit = candidate_by_seed[seed_value]
                baseline_audit = baseline_by_seed[seed_value]
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
