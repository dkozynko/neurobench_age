from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_article_experiment_launcher_is_portable_and_validation_only() -> None:
    script = (ROOT / "scripts/run_article_experiment.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "REPO_ROOT" in script
    assert "validation_only" in script
    assert "/workspace" not in script
    assert "/venv/main" not in script
    assert "rejects final-test override" in script
    assert "PHASE" in script
    assert "HEAD_VARIANT" in script


def test_article_analysis_launcher_is_portable() -> None:
    script = (ROOT / "scripts/run_article_analysis.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "REPO_ROOT" in script
    assert "ANALYZER" in script
    assert "/workspace" not in script
    assert "/venv/main" not in script


def test_article_wrappers_are_not_ignored() -> None:
    for path in (
        "scripts/run_article_experiment.sh",
        "scripts/run_article_analysis.sh",
        "scripts/analyze_paper_evidence.py",
        "tests/test_article_ready_scripts.py",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "-q", path],
            cwd=ROOT,
            check=False,
        )
        assert result.returncode != 0, f"{path} is ignored"
