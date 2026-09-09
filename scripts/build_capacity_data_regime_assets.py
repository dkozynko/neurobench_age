#!/usr/bin/env python3
"""Render isolated tables and figures for the secondary extension evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from neurobench_age.research.capacity_data_regime_lock import (
    CapacityDataRegimeLockError,
    load_final_lock,
)


class CapacityDataRegimeAssetError(ValueError):
    """Raised when extension assets are not bound to finalized evidence."""


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CapacityDataRegimeAssetError(f"cannot read {description}: {path}") from error
    if not isinstance(value, dict):
        raise CapacityDataRegimeAssetError(f"{description} must be an object")
    return value


def _write_create_only(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise CapacityDataRegimeAssetError(f"asset already exists with different content: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _validate_output_root(output_root: Path, repository_root: Path) -> Path:
    output = Path(output_root).resolve()
    forbidden = (
        repository_root / "manuscript/generated",
        repository_root / "results/canonical",
    )
    if any(output == root.resolve() or output.is_relative_to(root.resolve()) for root in forbidden):
        raise CapacityDataRegimeAssetError(
            "extension assets cannot overwrite primary generated assets"
        )
    return output


def _svg_figure(cells: Sequence[Mapping[str, Any]]) -> str:
    colors = {
        "mean_rich_stats_residual": "#1f77b4",
        "mean_mlp_residual_matched(hidden_dim=4)": "#d62728",
    }
    width, height = 760, 430
    left, bottom, plot_width, plot_height = 90, 70, 590, 290
    values = [float(cell["mean_delta"]) for cell in cells]
    ymin = min(values + [0.0])
    ymax = max(values + [0.0])
    span = max(ymax - ymin, 1e-9)

    def x(size: int) -> float:
        return left + ((size - 200) / 600.0) * plot_width

    def y(value: float) -> float:
        return bottom + plot_height - ((value - ymin) / span) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="380" y="28" text-anchor="middle" font-family="sans-serif" font-size="16">Capacity–data regime: mean paired Pearson delta</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{left}" y2="{bottom + plot_height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{bottom + plot_height}" x2="{left + plot_width}" y2="{bottom + plot_height}" stroke="#222"/>',
    ]
    if ymin <= 0.0 <= ymax:
        lines.append(
            f'<line x1="{left}" y1="{y(0.0):.2f}" x2="{left + plot_width}" y2="{y(0.0):.2f}" stroke="#aaa" stroke-dasharray="4 4"/>'
        )
    for size in (200, 400, 800):
        lines.append(f'<line x1="{x(size):.2f}" y1="{bottom + plot_height}" x2="{x(size):.2f}" y2="{bottom + plot_height + 6}" stroke="#222"/>')
        lines.append(f'<text x="{x(size):.2f}" y="{bottom + plot_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{size}</text>')
    for head, color in colors.items():
        points = [cell for cell in cells if cell["head"] == head]
        points.sort(key=lambda cell: int(cell["training_size"]))
        path = " ".join(f"{x(int(point['training_size'])):.2f},{y(float(point['mean_delta'])):.2f}" for point in points)
        lines.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for point in points:
            lines.append(f'<circle cx="{x(int(point["training_size"])):.2f}" cy="{y(float(point["mean_delta"])):.2f}" r="4" fill="{color}"/>')
    lines.extend(
        [
            '<text x="380" y="410" text-anchor="middle" font-family="sans-serif" font-size="13">HBN training subjects</text>',
            '<text x="18" y="220" transform="rotate(-90 18 220)" text-anchor="middle" font-family="sans-serif" font-size="13">Candidate − mean_linear Pearson</text>',
            '<text x="500" y="55" font-family="sans-serif" font-size="12" fill="#1f77b4">rich statistics</text>',
            '<text x="500" y="72" font-family="sans-serif" font-size="12" fill="#d62728">matched MLP (hidden_dim=4)</text>',
            '</svg>',
        ]
    )
    return "\n".join(lines) + "\n"


def _seed_delta_svg(cells: Sequence[Mapping[str, Any]]) -> str:
    width, height = 800, 430
    left, bottom, plot_width, plot_height = 90, 70, 630, 290
    values = [float(row["pearson_delta"]) for cell in cells for row in cell["per_seed"]]
    ymin, ymax = min(values + [0.0]), max(values + [0.0])
    span = max(ymax - ymin, 1e-9)

    def x(index: int) -> float:
        return left + index * (plot_width / 9.0)

    def y(value: float) -> float:
        return bottom + plot_height - ((value - ymin) / span) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="400" y="28" text-anchor="middle" font-family="sans-serif" font-size="16">Per-seed paired external Pearson deltas</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{left}" y2="{bottom + plot_height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{bottom + plot_height}" x2="{left + plot_width}" y2="{bottom + plot_height}" stroke="#222"/>',
    ]
    if ymin <= 0.0 <= ymax:
        lines.append(f'<line x1="{left}" y1="{y(0.0):.2f}" x2="{left + plot_width}" y2="{y(0.0):.2f}" stroke="#aaa" stroke-dasharray="4 4"/>')
    colors = {"mean_rich_stats_residual": "#1f77b4", "mean_mlp_residual_matched(hidden_dim=4)": "#d62728"}
    for cell_index, cell in enumerate(cells):
        color = colors[str(cell["head"])]
        offset = (cell_index % 3 - 1) * 3.0
        for seed_index, row in enumerate(cell["per_seed"]):
            lines.append(f'<circle cx="{x(seed_index) + offset:.2f}" cy="{y(float(row["pearson_delta"])):.2f}" r="2.5" fill="{color}" opacity="0.55"/>')
    lines.extend(
        [
            '<text x="400" y="410" text-anchor="middle" font-family="sans-serif" font-size="13">Seed index (33–42; points are grouped by training size)</text>',
            '<text x="18" y="220" transform="rotate(-90 18 220)" text-anchor="middle" font-family="sans-serif" font-size="13">Pearson delta</text>',
            '</svg>',
        ]
    )
    return "\n".join(lines) + "\n"


def _parameter_count(head: str) -> int:
    counts = {
        "mean_linear": 513,
        "mean_rich_stats_residual": 2561,
        "mean_mlp_residual_matched(hidden_dim=4)": 2569,
    }
    try:
        return counts[head]
    except KeyError as error:
        raise CapacityDataRegimeAssetError(f"unexpected extension head: {head}") from error


def build_capacity_data_regime_assets(
    *, analysis_path: Path, final_lock_path: Path, output_root: Path, repository_root: Path
) -> Mapping[str, Any]:
    analysis = _load_json(analysis_path, "capacity analysis")
    try:
        final_lock = load_final_lock(_load_json(final_lock_path, "final extension lock"))
    except CapacityDataRegimeLockError as error:
        raise CapacityDataRegimeAssetError(str(error)) from error
    body = {key: value for key, value in analysis.items() if key != "analysis_sha256"}
    if analysis.get("analysis_sha256") != _canonical_sha256(body):
        raise CapacityDataRegimeAssetError("analysis digest does not match its content")
    if analysis.get("final_lock_sha256") != final_lock["lock_sha256"]:
        raise CapacityDataRegimeAssetError("analysis is not bound to the supplied final lock")
    cells = analysis.get("cells")
    if not isinstance(cells, list) or len(cells) != 6:
        raise CapacityDataRegimeAssetError("analysis must contain all six extension cells")
    training_evidence = analysis.get("training_evidence")
    if not isinstance(training_evidence, Mapping) or training_evidence.get("run_count") != 90:
        raise CapacityDataRegimeAssetError(
            "analysis must retain the complete 90-run training evidence"
        )
    output = _validate_output_root(output_root, repository_root)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "capacity_data_regime_cells.csv"
    csv_lines: list[list[object]] = [["training_size", "head", "parameter_count", "mean_delta", "wins", "ties", "losses", "worst_seed_delta"]]
    for cell in cells:
        csv_lines.append([cell["training_size"], cell["head"], _parameter_count(str(cell["head"])), cell["mean_delta"], cell["wins"], cell["ties"], cell["losses"], cell["worst_seed_delta"]])
    import io

    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerows(csv_lines)
    _write_create_only(csv_path, csv_buffer.getvalue())
    table_rows = [
        "\\begin{tabular}{llrrrrr}",
        "Training size & Head & Parameters & Mean $\\Delta$ & Wins & Ties & Losses \\\\",
        "\\hline",
    ]
    for cell in cells:
        table_rows.append(
            f"{cell['training_size']} & {str(cell['head']).replace('_', r'\\_')} & {_parameter_count(str(cell['head']))} & {float(cell['mean_delta']):.4f} & {cell['wins']} & {cell['ties']} & {cell['losses']} \\\\")
    table_rows.append("\\end{tabular}")
    _write_create_only(output / "capacity_data_regime_cells.tex", "\n".join(table_rows) + "\n")
    _write_create_only(output / "capacity_data_regime_delta.svg", _svg_figure(cells))
    seed_rows: list[list[object]] = [["training_size", "head", "seed", "pearson_delta"]]
    for cell in cells:
        for row in cell["per_seed"]:
            seed_rows.append([cell["training_size"], cell["head"], row["seed"], row["pearson_delta"]])
    seed_buffer = io.StringIO(newline="")
    csv.writer(seed_buffer).writerows(seed_rows)
    _write_create_only(output / "capacity_data_regime_seed_deltas.csv", seed_buffer.getvalue())
    _write_create_only(output / "capacity_data_regime_seed_deltas.svg", _seed_delta_svg(cells))
    validation_lookup: dict[tuple[int, str, int], float] = {}
    for run in training_evidence["runs"]:
        history = run["validation_history"]
        selected_epoch = run["selected_epoch"]
        selected = [row for row in history if row.get("epoch") == selected_epoch]
        if len(selected) != 1 or not isinstance(selected[0].get("validation_subject_pearson"), (int, float)):
            raise CapacityDataRegimeAssetError("training validation history cannot resolve selected Pearson")
        validation_lookup[(run["training_size"], run["head"], run["seed"])] = float(selected[0]["validation_subject_pearson"])
    transfer_rows = [
        "\\begin{tabular}{llrr}",
        "Training size & Head & HBN validation Pearson & MIPDB external Pearson \\\\",
        "\\hline",
    ]
    for cell in cells:
        validation_values = [validation_lookup[(cell["training_size"], cell["head"], row["seed"])] for row in cell["per_seed"]]
        external_values = [float(row["candidate"]["pearson"]) for row in cell["per_seed"]]
        transfer_rows.append(
            f"{cell['training_size']} & {str(cell['head']).replace('_', r'\\_')} & {sum(validation_values) / len(validation_values):.4f} & {sum(external_values) / len(external_values):.4f} \\\\")
    transfer_rows.append("\\end{tabular}")
    _write_create_only(output / "capacity_data_regime_validation_external_transfer.tex", "\n".join(transfer_rows) + "\n")
    manifest_body = {
        "schema_version": 1,
        "status": "complete",
        "analysis_sha256": analysis["analysis_sha256"],
        "final_lock_sha256": final_lock["lock_sha256"],
        "output_files": [
            "capacity_data_regime_cells.csv",
            "capacity_data_regime_cells.tex",
            "capacity_data_regime_delta.svg",
            "capacity_data_regime_seed_deltas.csv",
            "capacity_data_regime_seed_deltas.svg",
            "capacity_data_regime_validation_external_transfer.tex",
        ],
    }
    manifest = {**manifest_body, "asset_manifest_sha256": _canonical_sha256(manifest_body)}
    _write_create_only(
        output / "capacity_data_regime_asset_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--final-lock", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_capacity_data_regime_assets(
            analysis_path=args.analysis,
            final_lock_path=args.final_lock,
            output_root=args.output_root,
            repository_root=Path(__file__).resolve().parents[1],
        )
        print(json.dumps({"status": manifest["status"], "asset_manifest_sha256": manifest["asset_manifest_sha256"]}, sort_keys=True))
        return 0
    except Exception as error:
        build_parser().error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
