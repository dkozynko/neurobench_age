from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import analyze_layerwise_probe as module
from tests.test_layerwise_analysis import _write_synthetic_layerwise_evidence


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/layerwise_probe_analysis.json"


def _arguments(root: Path, output: Path) -> list[str]:
    return [
        "--artifact-root",
        str(root),
        "--analysis-config",
        str(CONFIG),
        "--protocol",
        str(CONFIG),
        "--training-protocol",
        str(CONFIG),
        "--output",
        str(output),
    ]


def test_layerwise_analysis_cli_rejects_relative_artifact_paths(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        module.main(
            [
                "--artifact-root",
                "relative-artifacts",
                "--analysis-config",
                str(CONFIG),
                "--protocol",
                str(CONFIG),
                "--training-protocol",
                str(CONFIG),
                "--output",
                str(tmp_path / "analysis.json"),
            ]
        )


def test_layerwise_analysis_cli_writes_and_exact_resumes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_synthetic_layerwise_evidence(tmp_path / "evidence")
    protocol = SimpleNamespace(sha256="a" * 64)
    training = SimpleNamespace(sha256="b" * 64)
    monkeypatch.setattr(module, "load_study_protocol", lambda *args, **kwargs: protocol)
    monkeypatch.setattr(
        module,
        "load_frozen_probe_training_protocol",
        lambda *args, **kwargs: training,
    )
    output = tmp_path / "analysis.json"

    assert module.main(_arguments(tmp_path / "evidence", output)) == 0
    original = output.read_text(encoding="utf-8")
    assert module.main(_arguments(tmp_path / "evidence", output)) == 0
    assert output.read_text(encoding="utf-8") == original

    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        module.main(_arguments(tmp_path / "evidence", output))
