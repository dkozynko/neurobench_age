#!/usr/bin/env python3
"""Create editorial presentation assets without changing retained evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import rfc8785

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neurobench_age.analysis.capacity_data_regime_assets import _render_figures


def verify_files(directory: Path, manifest: dict) -> None:
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("source manifest path escapes its directory")
        data = path.read_bytes()
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"source asset hash mismatch: {name}")


def build(repository: Path, output: Path) -> dict:
    repository = repository.resolve()
    if output.resolve().is_relative_to(repository / "results"):
        raise ValueError("presentation output must not overwrite retained results")
    capacity = repository / "results/extensions/capacity_data_regime_v3"
    primary = repository / "results/canonical/prospective"
    manifest_path = capacity / "capacity_data_regime_asset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    body = {key: value for key, value in manifest.items() if key != "asset_manifest_sha256"}
    if hashlib.sha256(rfc8785.dumps(body)).hexdigest() != manifest["asset_manifest_sha256"]:
        raise ValueError("capacity manifest hash mismatch")
    verify_files(capacity, manifest)
    primary_manifest = json.loads((primary / "artifact_manifest.json").read_text())
    verify_files(primary, primary_manifest)

    replacements = {
        r"Matched MLP (hidden\_dim=4)": "MLP (4 units)",
        "Rich-statistics residual": "Rich statistics",
        r"endpoint\_800\_minus\_200": "800 minus 200",
        r"adjacent\_400\_minus\_200": "400 minus 200",
        r"adjacent\_800\_minus\_400": "800 minus 400",
        "Training size": "Train $n$",
        "HBN validation Pearson": "HBN validation $r$",
        "MIPDB external Pearson": "MIPDB $r$",
    }
    generated = {}
    for path in sorted(capacity.glob("*.tex")):
        text = path.read_text()
        for old, new in replacements.items():
            text = text.replace(old, new)
        generated[path.name] = text.encode()

    with (capacity / "capacity_data_regime_cells.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (capacity / "capacity_data_regime_seed_deltas.csv").open(newline="") as handle:
        seeds = list(csv.DictReader(handle))
    cells = []
    for row in rows:
        cells.append({
            "head": row["head"], "training_size": int(row["training_size"]),
            "mean_candidate_pearson": float(row["mean_candidate_pearson"]),
            "mean_baseline_pearson": float(row["mean_baseline_pearson"]),
            "mean_delta": float(row["mean_delta"]),
            "bootstrap": {"ci_low": float(row["bootstrap_ci_low"]), "ci_high": float(row["bootstrap_ci_high"])},
            "per_seed": [{"seed": int(item["seed"]), "pearson_delta": float(item["pearson_delta"])}
                         for item in seeds if item["head"] == row["head"] and item["training_size"] == row["training_size"]],
        })
    generated.update(_render_figures(cells))
    summary = json.loads((primary / "study_summary.json").read_text())
    head_commands = {
        "mean_linear": "BaselineParameters", "mean_layer_linear": "LayerParameters",
        "mean_rich_stats_residual": "RichParameters", "multi_query_rich_stats": "QueryParameters",
    }
    generated["head_parameters.tex"] = ("% Generated from verified aggregate evidence.\n" + "".join(
        rf"\newcommand{{\{command}}}{{{summary['resource_summary'][head]['head_parameter_count']}}}" + "\n"
        for head, command in head_commands.items()
    )).encode()
    sources = [manifest_path, primary / "artifact_manifest.json", primary / "study_summary.json"]
    provenance = {
        "description": "Editorial labels and figures derived from retained aggregate evidence; no new inference.",
        "sources": {str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in sorted(generated.items())},
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, data in generated.items():
        (output / name).write_bytes(data)
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "manuscript/presentation")
    args = parser.parse_args()
    try:
        result = build(args.repository_root, args.output)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(f"Validated source evidence; generated {len(result['files'])} presentation assets")


if __name__ == "__main__":
    main()
