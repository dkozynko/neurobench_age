from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobench_age.analysis.layerwise import _canonical_sha256
from neurobench_age.analysis.layerwise_assets import build_layerwise_assets


def _write_analysis(path: Path) -> None:
    comparisons = {}
    for index, (head, layer, delta) in enumerate(
        (
            ("mean_linear_layer_m4", -4, 0.04),
            ("mean_linear_layer_m3", -3, 0.03),
            ("mean_linear_layer_m2", -2, 0.01),
        )
    ):
        per_seed = [
            {
                "seed": seed,
                "pearson_delta": delta + 0.001 * (seed - 37),
                "candidate": {"pearson": 0.68 + delta},
                "baseline": {"pearson": 0.64},
            }
            for seed in range(33, 43)
        ]
        comparisons[head] = {
            "layer_index": layer,
            "paired": {
                "mean_pearson_delta": delta,
                "wins": 10,
                "ties": 0,
                "losses": 0,
                "worst_seed_delta": min(row["pearson_delta"] for row in per_seed),
                "per_seed": per_seed,
            },
            "bootstrap": {"ci_low": delta - 0.01, "ci_high": delta + 0.01},
            "randomization": {"p_value": 0.02 + index * 0.01},
            "holm_adjusted_p_value": 0.06 + index * 0.01,
        }
    body = {
        "schema_version": 1,
        "status": "complete",
        "scope": "exploratory_secondary",
        "analysis_id": "test",
        "lock_sha256": "a" * 64,
        "metrics_sha256": "b" * 64,
        "prediction_inventory_sha256": "c" * 64,
        "analysis_config_sha256": "d" * 64,
        "baseline_head": "mean_linear_layer_m1",
        "comparison_order": list(comparisons),
        "comparisons": comparisons,
        "subject_count": 75,
        "seeds": list(range(33, 43)),
    }
    path.write_text(
        json.dumps({**body, "analysis_sha256": _canonical_sha256(body)}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def test_layerwise_assets_are_deterministic_and_aggregate_only(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.json"
    _write_analysis(analysis)
    first = build_layerwise_assets(
        analysis_path=analysis,
        output_root=tmp_path / "assets-a",
        repository_root=tmp_path / "repo",
    )
    second = build_layerwise_assets(
        analysis_path=analysis,
        output_root=tmp_path / "assets-b",
        repository_root=tmp_path / "repo",
    )

    assert first == second
    assert set(first["files"]) == {
        "layerwise_summary.tex",
        "layerwise_results_macros.tex",
        "layerwise_depth.pdf",
        "layerwise_seed_deltas.pdf",
        "assets_manifest.json",
    }
    assert "subject_id" not in (
        tmp_path / "assets-a" / "layerwise_summary.tex"
    ).read_text(encoding="utf-8")
    assert "prediction" not in (
        tmp_path / "assets-a" / "layerwise_summary.tex"
    ).read_text(encoding="utf-8").lower()


def test_layerwise_assets_reject_primary_output_root(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.json"
    _write_analysis(analysis)
    repository = tmp_path / "repo"
    forbidden = repository / "manuscript/generated"

    with pytest.raises(ValueError, match="primary"):
        build_layerwise_assets(
            analysis_path=analysis,
            output_root=forbidden,
            repository_root=repository,
        )
