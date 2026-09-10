"""Deterministic aggregate assets for the ds006780 external-transfer extension."""

from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neurobench_age.research.strict_json import canonical_sha256


class Ds006780AssetError(ValueError):
    """Raised when the retained ds006780 evidence is incomplete or altered."""


_CELL_ORDER = (
    (200, "mean_linear"),
    (200, "mean_rich_stats_residual"),
    (800, "mean_linear"),
    (800, "mean_rich_stats_residual"),
)
_HEAD_LABELS = {
    "mean_linear": "Mean-pooled linear",
    "mean_rich_stats_residual": "Rich-statistics residual",
}
_MACRO_SUFFIX = {
    "mean_linear": "Baseline",
    "mean_rich_stats_residual": "RichResidual",
}


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Ds006780AssetError(f"could not read {description}: {path}") from error
    if not isinstance(value, dict):
        raise Ds006780AssetError(f"{description} must contain a JSON object")
    return value


def _validate_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise Ds006780AssetError(f"{field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise Ds006780AssetError(f"{field} must be a SHA-256 digest") from error
    return value


def _validate_self_hash(payload: Mapping[str, Any], field: str, description: str) -> None:
    claimed = _validate_sha256(payload.get(field), field)
    body = {key: value for key, value in payload.items() if key != field}
    if claimed != canonical_sha256(body):
        raise Ds006780AssetError(f"{description} {field} does not match content")


def _validate_evidence(
    analysis: Mapping[str, Any], precision_gate: Mapping[str, Any]
) -> None:
    _validate_self_hash(analysis, "analysis_sha256", "analysis")
    _validate_self_hash(precision_gate, "precision_gate_sha256", "precision gate")
    if analysis.get("status") != "complete":
        raise Ds006780AssetError("analysis is not complete")
    if analysis.get("analysis") != "ds006780_external_head_complexity":
        raise Ds006780AssetError("analysis identity is not the ds006780 head-complexity report")
    if tuple((row.get("training_size"), row.get("head")) for row in analysis.get("cell_summaries", ())) != _CELL_ORDER:
        raise Ds006780AssetError("analysis cell order is invalid")
    if analysis.get("subject_count") != 126:
        raise Ds006780AssetError("analysis subject count is not the sealed 126-subject cohort")
    primary = analysis.get("primary_contrast")
    secondary = analysis.get("secondary_contrasts")
    if not isinstance(primary, Mapping) or not isinstance(secondary, list) or len(secondary) != 1:
        raise Ds006780AssetError("analysis contrast inventory is invalid")
    if primary.get("training_size") != 800 or secondary[0].get("training_size") != 200:
        raise Ds006780AssetError("analysis primary/secondary training sizes are invalid")
    common_provenance = analysis.get("common_provenance")
    if not isinstance(common_provenance, Mapping):
        raise Ds006780AssetError("analysis common provenance is missing")
    if (
        precision_gate.get("analysis_sha256") != analysis.get("analysis_sha256")
        or precision_gate.get("lock_sha256") != analysis.get("lock_sha256")
        or precision_gate.get("study_id") != common_provenance.get("study_id")
    ):
        raise Ds006780AssetError("precision gate provenance differs from analysis")
    if precision_gate.get("status") != "failed":
        raise Ds006780AssetError("precision gate status is not the observed failed gate")
    bootstrap = primary.get("bootstrap")
    paired = primary.get("paired")
    if not isinstance(bootstrap, Mapping) or not isinstance(paired, Mapping):
        raise Ds006780AssetError("primary contrast statistics are incomplete")
    if bootstrap.get("iterations") != 10_000 or bootstrap.get("subject_count") != 126:
        raise Ds006780AssetError("primary bootstrap contract is invalid")
    if paired.get("wins") != 10 or paired.get("losses") != 0:
        raise Ds006780AssetError("primary seed-pair accounting is invalid")
    expected_width = float(bootstrap["ci_high"]) - float(bootstrap["ci_low"])
    if abs(expected_width - float(precision_gate["primary_observed_ci_width"])) > 1e-12:
        raise Ds006780AssetError("precision-gate interval width differs from analysis")


def _number(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _render_table(analysis: Mapping[str, Any], precision_gate: Mapping[str, Any]) -> str:
    cells = {
        (int(cell["training_size"]), str(cell["head"])): cell
        for cell in analysis["cell_summaries"]
    }
    primary = analysis["primary_contrast"]
    secondary = analysis["secondary_contrasts"][0]
    contrasts = {200: secondary, 800: primary}
    rows = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Training $n$ & Head & Pearson & Seed SD & $\Delta r$ & 95\% interval \\",
        r"\midrule",
    ]
    for size, head in _CELL_ORDER:
        cell = cells[(size, head)]
        if head == "mean_linear":
            delta = "--"
            interval = "--"
        else:
            contrast = contrasts[size]
            delta = _number(contrast["paired"]["mean_pearson_delta"])
            interval = (
                f"[{_number(contrast['bootstrap']['ci_low'])}, "
                f"{_number(contrast['bootstrap']['ci_high'])}]"
            )
        rows.append(
            f"{size} & {_HEAD_LABELS[head]} & {_number(cell['mean_pearson'])} & "
            f"{_number(cell['seed_pearson_sd'])} & {delta} & {interval} "
            r"\\"
        )
    rows.extend(
        [
            r"\midrule",
            r"\multicolumn{6}{l}{Precision gate: "
            f"{precision_gate['status']}; observed width "
            f"{_number(precision_gate['primary_observed_ci_width'])}, "
            f"threshold {_number(precision_gate['max_primary_confidence_interval_width'])}.}} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    return "\n".join(rows)


def _render_macros(analysis: Mapping[str, Any], precision_gate: Mapping[str, Any]) -> str:
    primary = analysis["primary_contrast"]
    secondary = analysis["secondary_contrasts"][0]
    lines = ["% Generated ds006780 aggregate macros; do not edit."]
    lines.extend(
        [
            f"\\newcommand{{\\DsSubjectCount}}{{{analysis['subject_count']}}}",
            r"\newcommand{\DsRunCount}{40}",
            r"\newcommand{\DsPredictionCount}{5040}",
            f"\\newcommand{{\\DsPrimaryDelta}}{{{_number(primary['paired']['mean_pearson_delta'])}}}",
            f"\\newcommand{{\\DsPrimaryCILow}}{{{_number(primary['bootstrap']['ci_low'])}}}",
            f"\\newcommand{{\\DsPrimaryCIHigh}}{{{_number(primary['bootstrap']['ci_high'])}}}",
            f"\\newcommand{{\\DsPrimaryCIWidth}}{{{_number(precision_gate['primary_observed_ci_width'])}}}",
            f"\\newcommand{{\\DsPrecisionMaxWidth}}{{{_number(precision_gate['max_primary_confidence_interval_width'])}}}",
            f"\\newcommand{{\\DsPrecisionGateStatus}}{{{precision_gate['status']}}}",
            f"\\newcommand{{\\DsPrimaryWins}}{{{primary['paired']['wins']}}}",
            f"\\newcommand{{\\DsSecondaryDelta}}{{{_number(secondary['paired']['mean_pearson_delta'])}}}",
            f"\\newcommand{{\\DsSecondaryCILow}}{{{_number(secondary['bootstrap']['ci_low'])}}}",
            f"\\newcommand{{\\DsSecondaryCIHigh}}{{{_number(secondary['bootstrap']['ci_high'])}}}",
        ]
    )
    for size, head in _CELL_ORDER:
        cell = next(
            item
            for item in analysis["cell_summaries"]
            if item["training_size"] == size and item["head"] == head
        )
        suffix = f"{'Small' if size == 200 else 'Large'}{_MACRO_SUFFIX[head]}"
        lines.append(
            f"\\newcommand{{\\DsPearson{suffix}}}"
            f"{{{_number(cell['mean_pearson'])}}}"
        )
    return "\n".join(lines) + "\n"


def _fixed_pdf_metadata(title: str) -> dict[str, object]:
    from datetime import datetime, timezone

    timestamp = datetime(2000, 1, 1, tzinfo=timezone.utc)
    return {
        "Title": title,
        "Author": "Anonymous",
        "Subject": "ds006780 external age-probing transfer",
        "Keywords": "EEG, REVE, age probing, external transfer",
        "Creator": "neurobench-age",
        "Producer": "Matplotlib",
        "CreationDate": timestamp,
        "ModDate": timestamp,
    }


def _render_figure(analysis: Mapping[str, Any]) -> bytes:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise Ds006780AssetError("Matplotlib is required for ds006780 figures") from error

    cells = {
        (int(cell["training_size"]), str(cell["head"])): cell
        for cell in analysis["cell_summaries"]
    }
    primary = analysis["primary_contrast"]
    secondary = analysis["secondary_contrasts"][0]
    contrasts = {200: secondary, 800: primary}
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
        figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), constrained_layout=True)
        sizes = [200, 800]
        for head, color, marker in (
            ("mean_linear", "#555555", "o"),
            ("mean_rich_stats_residual", "#0072B2", "s"),
        ):
            means = [cells[(size, head)]["mean_pearson"] for size in sizes]
            sds = [cells[(size, head)]["seed_pearson_sd"] for size in sizes]
            axes[0].errorbar(
                sizes,
                means,
                yerr=sds,
                marker=marker,
                color=color,
                capsize=3,
                label=_HEAD_LABELS[head],
            )
        axes[0].set_xlabel("HBN training subjects")
        axes[0].set_ylabel("Mean external Pearson")
        axes[0].set_xticks(sizes)
        axes[0].legend(frameon=False, fontsize=8)
        axes[0].grid(axis="y", alpha=0.25)

        deltas = [contrasts[size]["paired"]["mean_pearson_delta"] for size in sizes]
        lows = [contrasts[size]["bootstrap"]["ci_low"] for size in sizes]
        highs = [contrasts[size]["bootstrap"]["ci_high"] for size in sizes]
        axes[1].errorbar(
            sizes,
            deltas,
            yerr=[[mean - low for mean, low in zip(deltas, lows)], [high - mean for high, mean in zip(highs, deltas)]],
            fmt="o-",
            color="#D55E00",
            capsize=3,
        )
        axes[1].axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        axes[1].set_xlabel("HBN training subjects")
        axes[1].set_ylabel("Rich head $-$ linear Pearson")
        axes[1].set_xticks(sizes)
        axes[1].grid(axis="y", alpha=0.25)
        buffer = BytesIO()
        figure.savefig(
            buffer,
            format="pdf",
            bbox_inches="tight",
            metadata=_fixed_pdf_metadata("ds006780 external head transfer"),
        )
        plt.close(figure)
    return buffer.getvalue()


def _write_exact(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise Ds006780AssetError(f"existing asset differs: {path}")
    if not path.exists():
        path.write_bytes(content)


def build_assets(
    *, analysis_path: Path, precision_gate_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Build exact aggregate-only LaTeX and figure assets from v5 evidence."""

    analysis = _load_object(Path(analysis_path), "analysis")
    precision_gate = _load_object(Path(precision_gate_path), "precision gate")
    _validate_evidence(analysis, precision_gate)
    output = Path(output_dir).resolve()
    contents = {
        "ds006780_results_macros.tex": _render_macros(analysis, precision_gate).encode("utf-8"),
        "ds006780_summary.tex": _render_table(analysis, precision_gate).encode("utf-8"),
        "ds006780_external.pdf": _render_figure(analysis),
    }
    for name, content in contents.items():
        _write_exact(output / name, content)
    files = {
        name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        for name, content in sorted(contents.items())
    }
    manifest_body = {
        "schema_version": 1,
        "status": "complete",
        "analysis_sha256": analysis["analysis_sha256"],
        "precision_gate_sha256": precision_gate["precision_gate_sha256"],
        "files": sorted((*contents, "assets_manifest.json")),
        "file_hashes": files,
    }
    manifest = {**manifest_body, "asset_manifest_sha256": canonical_sha256(manifest_body)}
    _write_exact(
        output / "assets_manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return {
        "status": "complete",
        "primary_training_size": analysis["primary_contrast"]["training_size"],
        "primary_mean_delta": analysis["primary_contrast"]["paired"]["mean_pearson_delta"],
        "primary_ci_low": analysis["primary_contrast"]["bootstrap"]["ci_low"],
        "primary_ci_high": analysis["primary_contrast"]["bootstrap"]["ci_high"],
        "precision_gate_status": precision_gate["status"],
        "precision_gate_width": precision_gate["primary_observed_ci_width"],
        "asset_manifest_sha256": manifest["asset_manifest_sha256"],
    }
