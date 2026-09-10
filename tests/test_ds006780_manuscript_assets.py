from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobench_age.analysis.ds006780_manuscript_assets import build_assets


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "extensions" / "ds006780_external_v5"


def test_build_assets_renders_v5_summary_and_precision_macros(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    result = build_assets(
        analysis_path=EVIDENCE / "external_analysis.json",
        precision_gate_path=EVIDENCE / "precision_gate.json",
        output_dir=tmp_path,
    )

    assert result["status"] == "complete"
    assert result["primary_training_size"] == 800
    assert result["primary_mean_delta"] == pytest.approx(0.028711339747722463)
    assert result["primary_ci_low"] == pytest.approx(-0.03494959845575062)
    assert result["primary_ci_high"] == pytest.approx(0.09223204814371752)
    assert result["precision_gate_status"] == "failed"
    assert result["precision_gate_width"] == pytest.approx(0.12718164659946815)
    for name in (
        "ds006780_results_macros.tex",
        "ds006780_summary.tex",
        "ds006780_external.pdf",
        "assets_manifest.json",
    ):
        assert (tmp_path / name).is_file()

    macros = (tmp_path / "ds006780_results_macros.tex").read_text(encoding="utf-8")
    assert "\\newcommand{\\DsPrimaryDelta}{0.0287}" in macros
    assert "\\newcommand{\\DsPrecisionGateStatus}{failed}" in macros

    table = (tmp_path / "ds006780_summary.tex").read_text(encoding="utf-8")
    assert "0.3255" in table
    assert "0.3543" in table
    assert "0.1272" in table


def test_build_assets_rejects_analysis_hash_mismatch(tmp_path: Path) -> None:
    analysis = json.loads(
        (EVIDENCE / "external_analysis.json").read_text(encoding="utf-8")
    )
    analysis["subject_count"] = 127
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(analysis), encoding="utf-8")

    with pytest.raises(ValueError, match="analysis_sha256"):
        build_assets(
            analysis_path=altered,
            precision_gate_path=EVIDENCE / "precision_gate.json",
            output_dir=tmp_path / "out",
        )
