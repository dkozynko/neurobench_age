"""Validated, deterministic assets for the retained REVE age-probing evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from neurobench_age.analysis.confirmatory import (
    ConfirmatoryAnalysisError,
    confirmatory_conclusion,
    holm_step_down,
    stable_improvement_decision,
)
from neurobench_age.research.study_lock import (
    StudyLockError,
    canonical_sha256,
    load_checkpoint_inventory,
    load_study_lock,
)


EXPECTED_HEADS = (
    "mean_linear",
    "mean_layer_linear",
    "mean_rich_stats_residual",
    "multi_query_rich_stats",
)
EXPECTED_SEEDS = tuple(range(33, 43))
HEAD_LABELS = {
    "mean_linear": "Mean-pooled linear",
    "mean_layer_linear": "Layer-mixed linear",
    "mean_rich_stats_residual": "Rich-statistics residual",
    "multi_query_rich_stats": "Multi-query rich-statistics",
}
HEAD_MACRO_PREFIXES = {
    "mean_linear": "Baseline",
    "mean_layer_linear": "LayerLinear",
    "mean_rich_stats_residual": "RichResidual",
    "multi_query_rich_stats": "MultiQuery",
}


class ManuscriptEvidenceError(RuntimeError):
    """Raised when retained evidence is incomplete, altered, or inconsistent."""


@dataclass(frozen=True)
class ValidatedSourceBundle:
    lock: Mapping[str, Any]
    checkpoint_inventory: Mapping[str, Any]
    prediction_inventory: Mapping[str, Any]
    analysis: Mapping[str, Any]
    run_pairs: tuple[tuple[str, int], ...]
    subject_ids: tuple[str, ...]
    prediction_count: int


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManuscriptEvidenceError(f"could not read {description}: {path}") from error
    if not isinstance(value, dict):
        raise ManuscriptEvidenceError(f"{description} must contain a JSON object")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_self_hash(
    payload: Mapping[str, Any], *, field: str, description: str
) -> None:
    claimed = payload.get(field)
    body = {key: value for key, value in payload.items() if key != field}
    if not _is_sha256(claimed) or claimed != canonical_sha256(body):
        raise ManuscriptEvidenceError(f"{description} {field} does not match content")


def _require_finite_numbers(value: object, *, path: str = "analysis") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManuscriptEvidenceError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_numbers(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_numbers(item, path=f"{path}[{index}]")


def _validate_hash_chain_and_decisions(
    *,
    lock: Mapping[str, Any],
    checkpoint_inventory: Mapping[str, Any],
    prediction_inventory: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> None:
    _validate_self_hash(
        analysis, field="analysis_sha256", description="confirmatory analysis"
    )
    _require_finite_numbers(analysis)
    expected_common = {
        "study_id": lock["study_id"],
        "lock_sha256": lock["lock_sha256"],
        "protocol_sha256": lock["protocol_sha256"],
        "training_protocol_sha256": lock["training_protocol_sha256"],
    }
    if any(prediction_inventory.get(key) != value for key, value in expected_common.items()):
        raise ManuscriptEvidenceError("prediction inventory provenance differs from lock")
    if any(analysis.get(key) != value for key, value in expected_common.items()):
        raise ManuscriptEvidenceError("confirmatory analysis provenance differs from lock")
    if (
        lock["checkpoint_inventory_sha256"]
        != checkpoint_inventory["checkpoint_inventory_sha256"]
        or checkpoint_inventory["representation_protocol_sha256"]
        != lock["protocol_sha256"]
        or checkpoint_inventory["training_protocol_sha256"]
        != lock["training_protocol_sha256"]
        or checkpoint_inventory["training_source_sha256"]
        != lock["training_source_sha256"]
    ):
        raise ManuscriptEvidenceError("checkpoint inventory provenance differs from lock")
    if (
        analysis.get("prediction_inventory_sha256")
        != prediction_inventory["prediction_inventory_sha256"]
    ):
        raise ManuscriptEvidenceError("analysis prediction-inventory hash chain differs")
    statistics = analysis.get("protocol_statistics")
    if not isinstance(statistics, Mapping):
        raise ManuscriptEvidenceError("analysis protocol statistics are missing")
    if (
        canonical_sha256(statistics) != lock["statistics_sha256"]
        or analysis.get("statistics_sha256") != lock["statistics_sha256"]
    ):
        raise ManuscriptEvidenceError("analysis statistics hash chain differs")
    comparisons = analysis.get("comparisons")
    candidates = EXPECTED_HEADS[1:]
    if not isinstance(comparisons, Mapping) or tuple(comparisons) != candidates:
        raise ManuscriptEvidenceError("analysis comparison inventory differs")
    if tuple(statistics.get("holm_order", ())) != candidates:
        raise ManuscriptEvidenceError("analysis Holm order differs from the declared heads")
    try:
        adjusted = holm_step_down(
            {
                head: float(comparisons[head]["randomization"]["p_value"])
                for head in candidates
            },
            order=candidates,
        )
    except (KeyError, TypeError, ValueError, ConfirmatoryAnalysisError) as error:
        raise ManuscriptEvidenceError("analysis randomization primitives are invalid") from error
    cohort = analysis.get("cohort")
    if not isinstance(cohort, Mapping) or not isinstance(cohort.get("underpowered"), bool):
        raise ManuscriptEvidenceError("analysis cohort power status is invalid")
    adequately_powered = not cohort["underpowered"]
    established: list[str] = []
    for head in candidates:
        comparison = comparisons[head]
        try:
            stored_adjusted = float(comparison["holm_adjusted_p_value"])
            if not math.isclose(stored_adjusted, adjusted[head], rel_tol=0.0, abs_tol=1e-15):
                raise ManuscriptEvidenceError("analysis Holm-adjusted p-value differs")
            expected_decision = stable_improvement_decision(
                adjusted_p_value=adjusted[head],
                bootstrap_ci_low=float(comparison["bootstrap"]["ci_low"]),
                wins=int(comparison["paired"]["wins"]),
                worst_seed_delta=float(comparison["paired"]["worst_seed_delta"]),
                alpha=float(statistics["alpha"]),
                minimum_seed_wins=int(statistics["minimum_seed_wins"]),
                minimum_worst_seed_delta=float(
                    statistics["minimum_worst_seed_delta"]
                ),
                require_ci_above_zero=bool(statistics["require_ci_above_zero"]),
                adequately_powered=adequately_powered,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ManuscriptEvidenceError("analysis decision primitives are invalid") from error
        if comparison.get("decision") != expected_decision:
            raise ManuscriptEvidenceError(
                f"analysis decision differs from recomputation for {head}"
            )
        if expected_decision["established_stable_improvement"]:
            established.append(head)
    try:
        expected_conclusion = confirmatory_conclusion(
            established, adequately_powered=adequately_powered
        )
    except ConfirmatoryAnalysisError as error:
        raise ManuscriptEvidenceError("analysis conclusion inventory is invalid") from error
    if analysis.get("established_heads") != established:
        raise ManuscriptEvidenceError("analysis established-head decision differs")
    if analysis.get("conclusion") != expected_conclusion:
        raise ManuscriptEvidenceError("analysis conclusion differs from recomputation")


def validate_source_bundle(
    *,
    lock_path: Path,
    checkpoint_inventory_path: Path,
    prediction_inventory_path: Path,
    analysis_path: Path,
) -> ValidatedSourceBundle:
    """Validate the exact sealed source inventory before retaining any evidence."""

    try:
        lock = load_study_lock(Path(lock_path))
        checkpoint_inventory = load_checkpoint_inventory(
            Path(checkpoint_inventory_path)
        )
    except StudyLockError as error:
        raise ManuscriptEvidenceError(str(error)) from error
    prediction_inventory = _load_object(
        Path(prediction_inventory_path), "prediction inventory"
    )
    analysis = _load_object(Path(analysis_path), "confirmatory analysis")
    _validate_self_hash(
        prediction_inventory,
        field="prediction_inventory_sha256",
        description="prediction inventory",
    )
    _validate_hash_chain_and_decisions(
        lock=lock,
        checkpoint_inventory=checkpoint_inventory,
        prediction_inventory=prediction_inventory,
        analysis=analysis,
    )

    if tuple(lock.get("heads", ())) != EXPECTED_HEADS or tuple(
        lock.get("seeds", ())
    ) != EXPECTED_SEEDS:
        raise ManuscriptEvidenceError("study lock does not declare the exact head/seed matrix")
    if (
        tuple(checkpoint_inventory.get("heads", ())) != EXPECTED_HEADS
        or tuple(checkpoint_inventory.get("seeds", ())) != EXPECTED_SEEDS
    ):
        raise ManuscriptEvidenceError(
            "checkpoint inventory does not declare the exact head/seed matrix"
        )
    expected_pairs = tuple(
        (head, seed) for head in EXPECTED_HEADS for seed in EXPECTED_SEEDS
    )
    actual_pairs = tuple(
        (record.get("head_name"), record.get("seed"))
        for record in checkpoint_inventory.get("runs", ())
        if isinstance(record, Mapping)
    )
    if len(actual_pairs) != 40 or set(actual_pairs) != set(expected_pairs):
        raise ManuscriptEvidenceError("checkpoint inventory is not the exact 40-run matrix")

    if (
        prediction_inventory.get("schema_version") != 3
        or prediction_inventory.get("status") != "complete"
        or tuple(prediction_inventory.get("heads", ())) != EXPECTED_HEADS
        or tuple(prediction_inventory.get("seeds", ())) != EXPECTED_SEEDS
    ):
        raise ManuscriptEvidenceError("prediction inventory header is invalid")
    raw_subjects = prediction_inventory.get("subjects")
    if (
        not isinstance(raw_subjects, list)
        or len(raw_subjects) != 75
        or any(not isinstance(subject, str) or not subject for subject in raw_subjects)
        or len(set(raw_subjects)) != 75
    ):
        raise ManuscriptEvidenceError("prediction inventory must declare 75 unique subjects")
    subject_ids = tuple(raw_subjects)
    if canonical_sha256(raw_subjects) != lock["subject_list_sha256"]["mipdb_primary"]:
        raise ManuscriptEvidenceError("primary subject digest differs from the study lock")

    entries = prediction_inventory.get("predictions")
    expected_count = len(EXPECTED_HEADS) * len(EXPECTED_SEEDS) * len(subject_ids)
    if (
        not isinstance(entries, list)
        or prediction_inventory.get("prediction_count") != expected_count
        or len(entries) != expected_count
    ):
        raise ManuscriptEvidenceError("prediction inventory does not contain 3,000 records")
    entry_fields = {
        "head_name",
        "seed",
        "subject_id",
        "path",
        "prediction_sha256",
    }
    keys: set[tuple[str, int, str]] = set()
    ordered_by_pair: dict[tuple[str, int], list[str]] = {
        pair: [] for pair in expected_pairs
    }
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != entry_fields:
            raise ManuscriptEvidenceError("prediction inventory contains an invalid record")
        head = entry.get("head_name")
        seed = entry.get("seed")
        subject = entry.get("subject_id")
        if not isinstance(head, str) or not isinstance(seed, int) or isinstance(seed, bool) or not isinstance(subject, str):
            raise ManuscriptEvidenceError("prediction inventory record identity is invalid")
        pair = (head, seed)
        key = (head, seed, subject)
        if pair not in ordered_by_pair or key in keys:
            raise ManuscriptEvidenceError("prediction inventory has an extra or duplicate identity")
        expected_path = f"predictions/{head}/seed-{seed}/{subject}.json"
        if entry.get("path") != expected_path or not _is_sha256(
            entry.get("prediction_sha256")
        ):
            raise ManuscriptEvidenceError("prediction inventory record provenance is invalid")
        keys.add(key)
        ordered_by_pair[pair].append(subject)
    expected_keys = {
        (head, seed, subject)
        for head in EXPECTED_HEADS
        for seed in EXPECTED_SEEDS
        for subject in subject_ids
    }
    if keys != expected_keys or any(
        tuple(pair_subjects) != subject_ids
        for pair_subjects in ordered_by_pair.values()
    ):
        raise ManuscriptEvidenceError(
            "each head/seed pair must contain the same ordered 75-subject cohort"
        )
    return ValidatedSourceBundle(
        lock=lock,
        checkpoint_inventory=checkpoint_inventory,
        prediction_inventory=prediction_inventory,
        analysis=analysis,
        run_pairs=expected_pairs,
        subject_ids=subject_ids,
        prediction_count=expected_count,
    )


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ManuscriptEvidenceError("compact evidence is not canonical JSON") from error


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _compact_analysis(bundle: ValidatedSourceBundle) -> dict[str, Any]:
    source = bundle.analysis
    exclusions_by_reason: dict[str, int] = {}
    for record in source.get("exclusions", ()):
        if not isinstance(record, Mapping) or not isinstance(record.get("reason"), str):
            raise ManuscriptEvidenceError("analysis exclusion summary is invalid")
        reason = record["reason"]
        exclusions_by_reason[reason] = exclusions_by_reason.get(reason, 0) + 1
    body = {
        "schema_version": 1,
        "status": "complete",
        "study_id": source["study_id"],
        "source_analysis_sha256": source["analysis_sha256"],
        "lock_sha256": source["lock_sha256"],
        "protocol_sha256": source["protocol_sha256"],
        "training_protocol_sha256": source["training_protocol_sha256"],
        "statistics_sha256": source["statistics_sha256"],
        "prediction_inventory_sha256": source["prediction_inventory_sha256"],
        "estimand": source["estimand"],
        "baseline": source["baseline"],
        "comparisons": source["comparisons"],
        "resources": source["resources"],
        "cohort": source["cohort"],
        "exclusion_count": len(source.get("exclusions", ())),
        "exclusions_by_reason": dict(sorted(exclusions_by_reason.items())),
        "established_heads": source["established_heads"],
        "conclusion": source["conclusion"],
        "protocol_statistics": source["protocol_statistics"],
    }
    return {**body, "compact_analysis_sha256": canonical_sha256(body)}


def build_study_summary(bundle: ValidatedSourceBundle) -> dict[str, Any]:
    """Return a compact identity and inventory summary without subject records."""

    lock = bundle.lock
    analysis = bundle.analysis
    cohort = analysis["cohort"]
    body = {
        "schema_version": 1,
        "status": "complete",
        "study_id": lock["study_id"],
        "identities": {
            "lock_sha256": lock["lock_sha256"],
            "protocol_sha256": lock["protocol_sha256"],
            "training_protocol_sha256": lock["training_protocol_sha256"],
            "statistics_sha256": lock["statistics_sha256"],
            "checkpoint_inventory_sha256": lock["checkpoint_inventory_sha256"],
            "prediction_inventory_sha256": bundle.prediction_inventory[
                "prediction_inventory_sha256"
            ],
            "analysis_sha256": analysis["analysis_sha256"],
            "mipdb_primary_subject_list_sha256": lock["subject_list_sha256"][
                "mipdb_primary"
            ],
        },
        "inventory": {
            "heads": list(EXPECTED_HEADS),
            "seeds": list(EXPECTED_SEEDS),
            "run_count": len(bundle.run_pairs),
            "prediction_count": bundle.prediction_count,
        },
        "cohorts": {
            "mipdb_pilot_subjects": cohort["pilot_subjects"],
            "mipdb_primary_subjects": cohort["primary_subjects"],
            "mipdb_extrapolation_subjects": cohort["extrapolation_subjects"],
            "minimum_primary_subjects": cohort["minimum_primary_subjects"],
            "primary_underpowered": cohort["underpowered"],
            "hbn_counts_retained": False,
        },
        "provenance_evidence": {
            "hbn_subject_manifest": lock["hbn_manifest_sha256"],
            "hbn_training_manifest": lock["hbn_training_manifest_sha256"],
            "mipdb_manifest": lock["mipdb_manifest_sha256"],
            "mipdb_pilot_qc": lock["mipdb_pilot_qc_sha256"],
            "mipdb_cohort_qc": lock["mipdb_cohort_qc_sha256"],
        },
        "resource_summary": analysis["resources"],
        "established_heads": analysis["established_heads"],
        "conclusion": analysis["conclusion"],
    }
    return {**body, "summary_sha256": canonical_sha256(body)}


def _directory_bytes(path: Path) -> dict[str, bytes]:
    return {
        item.name: item.read_bytes()
        for item in sorted(path.iterdir())
        if item.is_file()
    }


def export_compact_evidence(
    bundle: ValidatedSourceBundle, destination: Path
) -> Mapping[str, Any]:
    """Publish a compact evidence bundle atomically and idempotently."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    compact = _compact_analysis(bundle)
    summary = build_study_summary(bundle)
    payloads = {
        "confirmatory_analysis.json": _json_bytes(compact),
        "study_summary.json": _json_bytes(summary),
    }
    manifest = {
        "schema_version": 1,
        "files": {
            name: {"sha256": _sha256_bytes(content), "bytes": len(content)}
            for name, content in sorted(payloads.items())
        },
    }
    manifest_bytes = _json_bytes(manifest)
    intended = {
        **payloads,
        "artifact_manifest.json": manifest_bytes,
        "artifact_manifest.sha256": (
            f"{_sha256_bytes(manifest_bytes)}  artifact_manifest.json\n"
        ).encode("ascii"),
    }
    if destination.exists():
        if not destination.is_dir() or _directory_bytes(destination) != intended:
            raise ManuscriptEvidenceError(
                "evidence destination exists with conflicting or partial content"
            )
        return manifest
    temporary: Path | None = None
    try:
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        for name, content in intended.items():
            path = temporary / name
            with path.open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        temporary.replace(destination)
        temporary = None
    except FileExistsError as error:
        raise ManuscriptEvidenceError("evidence destination appeared during publication") from error
    finally:
        if temporary is not None and temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
    return manifest


def latex_escape(value: str) -> str:
    """Escape plain text for use in LaTeX table cells."""

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def format_decimal(value: object, *, digits: int = 3) -> str:
    """Format a finite number with stable rounding and no negative zero."""

    number = _finite_float(value, name="display value")
    if not isinstance(digits, int) or isinstance(digits, bool) or digits < 0:
        raise ValueError("digits must be a non-negative integer")
    if round(number, digits) == 0:
        number = 0.0
    return f"{number:.{digits}f}"


def format_p_value(value: object) -> str:
    number = _finite_float(value, name="p-value")
    if not 0.0 <= number <= 1.0:
        raise ValueError("p-value must be within [0, 1]")
    return r"<0.001" if number < 0.001 else format_decimal(number, digits=3)


def format_interval(low: object, high: object, *, digits: int = 3) -> str:
    lower = _finite_float(low, name="interval lower bound")
    upper = _finite_float(high, name="interval upper bound")
    if lower > upper:
        raise ValueError("interval lower bound exceeds upper bound")
    return rf"[{format_decimal(lower, digits=digits)},\ {format_decimal(upper, digits=digits)}]"


def _metrics_for_head(analysis: Mapping[str, Any], head: str) -> Mapping[str, Any]:
    if head == "mean_linear":
        return analysis["baseline"]["metrics"]
    return analysis["comparisons"][head]["candidate_metrics"]


def _macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def render_text_assets(analysis: Mapping[str, Any]) -> dict[str, str]:
    """Render generated LaTeX macros and table rows from compact analysis."""

    _require_finite_numbers(analysis)
    macros = ["% Generated deterministically; do not edit."]
    cohort = analysis["cohort"]
    macros.extend(
        (
            _macro("PrimarySubjects", str(cohort["primary_subjects"])),
            _macro("PilotSubjects", str(cohort["pilot_subjects"])),
            _macro("ExtrapolationSubjects", str(cohort["extrapolation_subjects"])),
            _macro("ProbeSeeds", str(len(EXPECTED_SEEDS))),
            _macro("ProbeRuns", str(len(EXPECTED_HEADS) * len(EXPECTED_SEEDS))),
            _macro("ExternalPredictions", str(len(EXPECTED_HEADS) * len(EXPECTED_SEEDS) * int(cohort["primary_subjects"]))),
            _macro("SourceAnalysisHash", latex_escape(str(analysis["source_analysis_sha256"]))),
            _macro("PredictionInventoryHash", latex_escape(str(analysis["prediction_inventory_sha256"]))),
        )
    )
    metric_macro_names = {
        "mean_pearson": "Pearson",
        "mean_mae": "MAE",
        "mean_rmse": "RMSE",
        "mean_r2": "RSquared",
        "mean_calibration_intercept": "CalibrationIntercept",
        "mean_calibration_slope": "CalibrationSlope",
    }
    for head in EXPECTED_HEADS:
        metrics = _metrics_for_head(analysis, head)
        prefix = HEAD_MACRO_PREFIXES[head]
        for field, suffix in metric_macro_names.items():
            macros.append(_macro(prefix + suffix, format_decimal(metrics[field])))
        if head != "mean_linear":
            comparison = analysis["comparisons"][head]
            paired = comparison["paired"]
            bootstrap = comparison["bootstrap"]
            macros.extend(
                (
                    _macro(prefix + "PearsonDelta", format_decimal(paired["mean_pearson_delta"])),
                    _macro(prefix + "CILow", format_decimal(bootstrap["ci_low"])),
                    _macro(prefix + "CIHigh", format_decimal(bootstrap["ci_high"])),
                    _macro(prefix + "AdjustedP", format_p_value(comparison["holm_adjusted_p_value"])),
                    _macro(prefix + "Wins", str(paired["wins"])),
                    _macro(prefix + "Ties", str(paired["ties"])),
                    _macro(prefix + "Losses", str(paired["losses"])),
                    _macro(prefix + "WorstSeedDelta", format_decimal(paired["worst_seed_delta"])),
                )
            )

    metric_rows = [
        "% Generated table; do not edit.",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Head & Pearson & MAE & RMSE & $R^2$ & Cal. slope \\",
        r"\midrule",
    ]
    for head in EXPECTED_HEADS:
        metrics = _metrics_for_head(analysis, head)
        metric_rows.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(HEAD_LABELS[head]),
                format_decimal(metrics["mean_pearson"]),
                format_decimal(metrics["mean_mae"]),
                format_decimal(metrics["mean_rmse"]),
                format_decimal(metrics["mean_r2"]),
                format_decimal(metrics["mean_calibration_slope"]),
            )
        )
    metric_rows.extend((r"\bottomrule", r"\end{tabular}"))

    comparison_rows = [
        "% Generated table; do not edit.",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Candidate & Mean $\Delta r$ & 95\% CI & Holm $p$ & W/T/L & Worst $\Delta r$ & Stable \\",
        r"\midrule",
    ]
    for head in EXPECTED_HEADS[1:]:
        comparison = analysis["comparisons"][head]
        paired = comparison["paired"]
        bootstrap = comparison["bootstrap"]
        comparison_rows.append(
            "{} & {} & {} & {} & {}/{}/{} & {} & {} \\\\".format(
                latex_escape(HEAD_LABELS[head]),
                format_decimal(paired["mean_pearson_delta"]),
                format_interval(bootstrap["ci_low"], bootstrap["ci_high"]),
                format_p_value(comparison["holm_adjusted_p_value"]),
                paired["wins"],
                paired["ties"],
                paired["losses"],
                format_decimal(paired["worst_seed_delta"]),
                "yes" if comparison["decision"]["established_stable_improvement"] else "no",
            )
        )
    comparison_rows.extend((r"\bottomrule", r"\end{tabular}"))

    cohort_rows = [
        "% Generated table; do not edit.",
        r"\begin{tabular}{lrl}",
        r"\toprule",
        r"Cohort & Subjects & Role \\",
        r"\midrule",
        f"MIPDB pilot & {cohort['pilot_subjects']} & Protocol/QC pilot \\\\",
        f"MIPDB primary & {cohort['primary_subjects']} & Sealed confirmatory cohort \\\\",
        f"MIPDB extrapolation & {cohort['extrapolation_subjects']} & Descriptive boundary cohort \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return {
        "results_macros.tex": "\n".join(macros) + "\n",
        "main_metrics.tex": "\n".join(metric_rows) + "\n",
        "confirmatory_comparisons.tex": "\n".join(comparison_rows) + "\n",
        "cohort_summary.tex": "\n".join(cohort_rows) + "\n",
    }


def _fixed_pdf_metadata(title: str) -> dict[str, object]:
    from datetime import datetime, timezone

    timestamp = datetime(2000, 1, 1, tzinfo=timezone.utc)
    return {
        "Title": title,
        "Author": "Anonymous",
        "Subject": "REVE age-probing evidence; No difference reference",
        "Keywords": "EEG, REVE, age probing",
        "Creator": "neurobench-age",
        "Producer": "Matplotlib",
        "CreationDate": timestamp,
        "ModDate": timestamp,
    }


def _figure_bytes(figure, *, title: str) -> bytes:
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="pdf",
        bbox_inches="tight",
        metadata=_fixed_pdf_metadata(title),
    )
    return buffer.getvalue()


def _render_figure_payloads(analysis: Mapping[str, Any]) -> dict[str, bytes]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "pdf.compression": 0,
            "pdf.fonttype": 42,
            "savefig.transparent": False,
        }
    ):
        colors = ("#0072B2", "#D55E00", "#009E73")
        seed_figure, seed_axis = plt.subplots(figsize=(6.2, 3.2), constrained_layout=True)
        for color, head in zip(colors, EXPECTED_HEADS[1:]):
            rows = analysis["comparisons"][head]["paired"]["per_seed"]
            seed_axis.plot(
                [row["seed"] for row in rows],
                [row["pearson_delta"] for row in rows],
                marker="o",
                markersize=3.5,
                color=color,
                label=HEAD_LABELS[head],
            )
        seed_axis.axhline(0.0, color="#333333", linestyle="--", label="No difference")
        seed_axis.set_xlabel("Training seed")
        seed_axis.set_ylabel("Paired Pearson delta")
        seed_axis.set_xticks(EXPECTED_SEEDS)
        seed_axis.legend(frameon=False, fontsize=7, ncol=2)
        seed_axis.grid(axis="y", alpha=0.2)
        seed_payload = _figure_bytes(seed_figure, title="Paired seed deltas")
        plt.close(seed_figure)

        interval_figure, interval_axis = plt.subplots(
            figsize=(6.2, 2.8), constrained_layout=True
        )
        candidates = EXPECTED_HEADS[1:]
        means = [analysis["comparisons"][head]["paired"]["mean_pearson_delta"] for head in candidates]
        lows = [analysis["comparisons"][head]["bootstrap"]["ci_low"] for head in candidates]
        highs = [analysis["comparisons"][head]["bootstrap"]["ci_high"] for head in candidates]
        positions = list(range(len(candidates)))
        interval_axis.errorbar(
            means,
            positions,
            xerr=[
                [mean - low for mean, low in zip(means, lows)],
                [high - mean for mean, high in zip(means, highs)],
            ],
            fmt="o",
            color="#0072B2",
            capsize=3,
        )
        interval_axis.axvline(0.0, color="#333333", linestyle="--", label="No difference")
        interval_axis.set_yticks(positions, [HEAD_LABELS[head] for head in candidates])
        interval_axis.set_xlabel("Mean paired Pearson delta (95% bootstrap interval)")
        interval_axis.grid(axis="x", alpha=0.2)
        interval_payload = _figure_bytes(
            interval_figure, title="Hierarchical bootstrap intervals"
        )
        plt.close(interval_figure)

        calibration_figure, axes = plt.subplots(
            1, 2, figsize=(7.0, 2.8), constrained_layout=True
        )
        positions = list(range(len(EXPECTED_HEADS)))
        slopes = [
            _metrics_for_head(analysis, head)["mean_calibration_slope"]
            for head in EXPECTED_HEADS
        ]
        intercepts = [
            _metrics_for_head(analysis, head)["mean_calibration_intercept"]
            for head in EXPECTED_HEADS
        ]
        short_labels = ("Mean", "Layer", "Residual", "Multi-query")
        axes[0].scatter(positions, slopes, color="#0072B2")
        axes[0].axhline(1.0, color="#333333", linestyle="--", label="Ideal")
        axes[0].set_ylabel("Calibration slope")
        axes[1].scatter(positions, intercepts, color="#D55E00")
        axes[1].axhline(0.0, color="#333333", linestyle="--", label="Ideal")
        axes[1].set_ylabel("Calibration intercept (years)")
        for axis in axes:
            axis.set_xticks(positions, short_labels, rotation=20, ha="right")
            axis.grid(axis="y", alpha=0.2)
        calibration_payload = _figure_bytes(
            calibration_figure, title="Calibration summary"
        )
        plt.close(calibration_figure)
    return {
        "seed_deltas.pdf": seed_payload,
        "bootstrap_intervals.pdf": interval_payload,
        "calibration_summary.pdf": calibration_payload,
    }


def _load_compact_evidence(evidence: Path) -> dict[str, Any]:
    evidence = Path(evidence)
    expected = {
        "confirmatory_analysis.json",
        "study_summary.json",
        "artifact_manifest.json",
        "artifact_manifest.sha256",
    }
    if not evidence.is_dir() or {path.name for path in evidence.iterdir()} != expected:
        raise ManuscriptEvidenceError("compact evidence inventory is not exact")
    manifest_bytes = (evidence / "artifact_manifest.json").read_bytes()
    expected_sidecar = f"{_sha256_bytes(manifest_bytes)}  artifact_manifest.json\n"
    if (evidence / "artifact_manifest.sha256").read_text(encoding="ascii") != expected_sidecar:
        raise ManuscriptEvidenceError("compact evidence manifest sidecar differs")
    manifest = _load_object(evidence / "artifact_manifest.json", "artifact manifest")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "confirmatory_analysis.json",
        "study_summary.json",
    }:
        raise ManuscriptEvidenceError("compact evidence manifest payload is invalid")
    for name, record in files.items():
        content = (evidence / name).read_bytes()
        if (
            not isinstance(record, Mapping)
            or record.get("sha256") != _sha256_bytes(content)
            or record.get("bytes") != len(content)
        ):
            raise ManuscriptEvidenceError(f"compact evidence hash differs for {name}")
    analysis = _load_object(
        evidence / "confirmatory_analysis.json", "compact confirmatory analysis"
    )
    analysis_body = {
        key: value
        for key, value in analysis.items()
        if key != "compact_analysis_sha256"
    }
    if analysis.get("compact_analysis_sha256") != canonical_sha256(analysis_body):
        raise ManuscriptEvidenceError("compact analysis digest differs")
    summary = _load_object(evidence / "study_summary.json", "study summary")
    summary_body = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if summary.get("summary_sha256") != canonical_sha256(summary_body):
        raise ManuscriptEvidenceError("study summary digest differs")
    if summary.get("identities", {}).get("analysis_sha256") != analysis.get(
        "source_analysis_sha256"
    ):
        raise ManuscriptEvidenceError("compact analysis and summary provenance differs")
    return analysis


def render_assets(evidence: Path, destination: Path) -> Mapping[str, Any]:
    """Render and atomically publish every deterministic derived asset."""

    analysis = _load_compact_evidence(Path(evidence))
    text_payloads = {
        name: value.encode("utf-8")
        for name, value in render_text_assets(analysis).items()
    }
    figure_payloads = _render_figure_payloads(analysis)
    payloads = {**text_payloads, **figure_payloads}
    manifest = {
        "schema_version": 1,
        "source_analysis_sha256": analysis["source_analysis_sha256"],
        "files": {
            name: {"sha256": _sha256_bytes(content), "bytes": len(content)}
            for name, content in sorted(payloads.items())
        },
    }
    manifest_bytes = _json_bytes(manifest)
    intended = {
        **payloads,
        "assets_manifest.json": manifest_bytes,
        "assets_manifest.sha256": (
            f"{_sha256_bytes(manifest_bytes)}  assets_manifest.json\n"
        ).encode("ascii"),
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_dir() or _directory_bytes(destination) != intended:
            raise ManuscriptEvidenceError(
                "asset destination exists with conflicting or partial content"
            )
        return manifest
    temporary: Path | None = None
    try:
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        for name, content in intended.items():
            path = temporary / name
            with path.open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        temporary.replace(destination)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
    return manifest
