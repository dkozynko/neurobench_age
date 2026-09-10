"""Deterministic aggregate tables and figures for layer-wise evidence."""

from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neurobench_age.analysis.layerwise import (
    LayerwiseAnalysisError,
    _canonical_sha256,
)


class LayerwiseAssetError(ValueError):
    """Raised when aggregate layer-wise assets cannot be created safely."""


_COMPARISON_ORDER = (
    "mean_linear_layer_m4",
    "mean_linear_layer_m3",
    "mean_linear_layer_m2",
)
_LABELS = {
    "mean_linear_layer_m4": "Layer -4",
    "mean_linear_layer_m3": "Layer -3",
    "mean_linear_layer_m2": "Layer -2",
}
_MACRO_SUFFIX = {
    "mean_linear_layer_m4": "MFour",
    "mean_linear_layer_m3": "MThree",
    "mean_linear_layer_m2": "MTwo",
}
_ASSET_NAMES = (
    "layerwise_summary.tex",
    "layerwise_results_macros.tex",
    "layerwise_depth.pdf",
    "layerwise_seed_deltas.pdf",
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LayerwiseAssetError(f"could not read analysis: {path}") from error
    if not isinstance(value, dict):
        raise LayerwiseAssetError("analysis must be a JSON object")
    return value


def _validate_analysis(analysis: Mapping[str, Any]) -> None:
    claimed = analysis.get("analysis_sha256")
    body = {key: value for key, value in analysis.items() if key != "analysis_sha256"}
    if not isinstance(claimed, str) or claimed != _canonical_sha256(body):
        raise LayerwiseAssetError("analysis self-hash does not match content")
    if analysis.get("status") != "complete" or analysis.get("scope") != "exploratory_secondary":
        raise LayerwiseAssetError("analysis is not a complete exploratory report")
    if analysis.get("baseline_head") != "mean_linear_layer_m1":
        raise LayerwiseAssetError("analysis baseline is not the final layer")
    if tuple(analysis.get("comparison_order", ())) != _COMPARISON_ORDER:
        raise LayerwiseAssetError("analysis comparison order is invalid")
    comparisons = analysis.get("comparisons")
    if not isinstance(comparisons, Mapping) or set(comparisons) != set(_COMPARISON_ORDER):
        raise LayerwiseAssetError("analysis comparison inventory is invalid")


def _latex_number(value: float) -> str:
    return f"{float(value):.3f}"


def _render_table(analysis: Mapping[str, Any]) -> str:
    comparisons = analysis["comparisons"]
    rows = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Layer & Mean $\Delta r$ & 95\% interval & Holm $p$ & W/T/L & Worst $\Delta r$ \\",
        r"\midrule",
    ]
    for head in _COMPARISON_ORDER:
        comparison = comparisons[head]
        paired = comparison["paired"]
        bootstrap = comparison["bootstrap"]
        rows.append(
            f"{_LABELS[head]} & "
            f"{_latex_number(paired['mean_pearson_delta'])} & "
            f"[{_latex_number(bootstrap['ci_low'])}, {_latex_number(bootstrap['ci_high'])}] & "
            f"{_latex_number(comparison['holm_adjusted_p_value'])} & "
            f"{paired['wins']}/{paired['ties']}/{paired['losses']} & "
            f"{_latex_number(paired['worst_seed_delta'])} "
            r"\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(rows)


def _render_macros(analysis: Mapping[str, Any]) -> str:
    lines = ["% Generated layer-wise exploratory macros; do not edit."]
    comparisons = analysis["comparisons"]
    for head in _COMPARISON_ORDER:
        suffix = _MACRO_SUFFIX[head]
        comparison = comparisons[head]
        paired = comparison["paired"]
        bootstrap = comparison["bootstrap"]
        lines.extend(
            [
                f"\\newcommand{{\\Layerwise{suffix}Delta}}{{{_latex_number(paired['mean_pearson_delta'])}}}",
                f"\\newcommand{{\\Layerwise{suffix}CILow}}{{{_latex_number(bootstrap['ci_low'])}}}",
                f"\\newcommand{{\\Layerwise{suffix}CIHigh}}{{{_latex_number(bootstrap['ci_high'])}}}",
                f"\\newcommand{{\\Layerwise{suffix}HolmP}}{{{_latex_number(comparison['holm_adjusted_p_value'])}}}",
            ]
        )
    lines.append(f"\\newcommand{{\\LayerwiseSubjectCount}}{{{analysis['subject_count']}}}")
    return "\n".join(lines) + "\n"


def _fixed_pdf_metadata(title: str) -> dict[str, object]:
    from datetime import datetime, timezone

    timestamp = datetime(2000, 1, 1, tzinfo=timezone.utc)
    return {
        "Title": title,
        "Author": "Anonymous",
        "Subject": "Layer-wise exploratory age probing",
        "Keywords": "EEG, REVE, age probing, representation depth",
        "Creator": "neurobench-age",
        "Producer": "Matplotlib",
        "CreationDate": timestamp,
        "ModDate": timestamp,
    }


def _render_figures(analysis: Mapping[str, Any]) -> dict[str, bytes]:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise LayerwiseAssetError("Matplotlib is required for layer-wise figures") from error

    comparisons = analysis["comparisons"]
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
        depth, axis = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
        x = [-4, -3, -2]
        means = [comparisons[head]["paired"]["mean_pearson_delta"] for head in _COMPARISON_ORDER]
        lows = [comparisons[head]["bootstrap"]["ci_low"] for head in _COMPARISON_ORDER]
        highs = [comparisons[head]["bootstrap"]["ci_high"] for head in _COMPARISON_ORDER]
        axis.errorbar(
            x,
            means,
            yerr=[[mean - low for mean, low in zip(means, lows)], [high - mean for high, mean in zip(highs, means)]],
            fmt="o-",
            color="#0072B2",
            capsize=3,
        )
        axis.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        axis.set_xlabel("Representation layer")
        axis.set_ylabel("Paired external Pearson difference vs. layer -1")
        axis.set_xticks(x)
        axis.grid(axis="y", alpha=0.25)
        depth_buffer = BytesIO()
        depth.savefig(
            depth_buffer,
            format="pdf",
            bbox_inches="tight",
            metadata=_fixed_pdf_metadata("Layer-wise paired external performance"),
        )
        plt.close(depth)

        seed, seed_axis = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)
        for head in _COMPARISON_ORDER:
            rows = comparisons[head]["paired"]["per_seed"]
            seed_axis.plot(
                [row["seed"] for row in rows],
                [row["pearson_delta"] for row in rows],
                marker="o",
                label=_LABELS[head],
            )
        seed_axis.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        seed_axis.set_xlabel("Optimization seed")
        seed_axis.set_ylabel("Paired external Pearson difference")
        seed_axis.legend(frameon=False)
        seed_axis.grid(axis="y", alpha=0.25)
        seed_buffer = BytesIO()
        seed.savefig(
            seed_buffer,
            format="pdf",
            bbox_inches="tight",
            metadata=_fixed_pdf_metadata("Layer-wise seed-level paired differences"),
        )
        plt.close(seed)
    return {
        "layerwise_depth.pdf": depth_buffer.getvalue(),
        "layerwise_seed_deltas.pdf": seed_buffer.getvalue(),
    }


def _write_exact(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise LayerwiseAssetError(f"existing asset differs: {path}")
        return
    path.write_bytes(content)


def build_layerwise_assets(
    *, analysis_path: Path, output_root: Path, repository_root: Path
) -> dict[str, Any]:
    """Create deterministic aggregate-only assets from a validated report."""

    output = Path(output_root).resolve()
    repository = Path(repository_root).resolve()
    forbidden = (repository / "manuscript/generated", repository / "results/canonical")
    if any(output == root or output.is_relative_to(root) for root in forbidden):
        raise LayerwiseAssetError("layer-wise assets cannot overwrite primary evidence")
    analysis = _load_json(Path(analysis_path))
    _validate_analysis(analysis)
    text_assets = {
        "layerwise_summary.tex": _render_table(analysis).encode("utf-8"),
        "layerwise_results_macros.tex": _render_macros(analysis).encode("utf-8"),
    }
    binary_assets = _render_figures(analysis)
    contents = {**text_assets, **binary_assets}
    for name, content in contents.items():
        _write_exact(output / name, content)
    files = {
        name: {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for name, content in sorted(contents.items())
    }
    manifest_body = {
        "schema_version": 1,
        "status": "complete",
        "analysis_sha256": analysis["analysis_sha256"],
        "files": sorted((*contents, "assets_manifest.json")),
        "file_hashes": files,
    }
    manifest = {**manifest_body, "asset_manifest_sha256": _canonical_sha256(manifest_body)}
    _write_exact(
        output / "assets_manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest
