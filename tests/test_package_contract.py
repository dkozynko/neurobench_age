from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict[str, object]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_supported_python_matches_neuralbench_requirement() -> None:
    project = _pyproject()["project"]

    assert project["requires-python"] == ">=3.12"


def test_pytest_imports_src_layout_without_manual_pythonpath() -> None:
    pytest_options = _pyproject()["tool"]["pytest"]["ini_options"]

    assert pytest_options["pythonpath"] == ["src"]


def test_external_eeg_dependencies_are_declared_as_an_optional_group() -> None:
    optional = _pyproject()["project"]["optional-dependencies"]
    external = optional["external"]

    assert any(requirement.startswith("mne>=") for requirement in external)
    assert any(requirement.startswith("mne-bids>=") for requirement in external)


def test_large_local_research_artifacts_are_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "/data/" in ignore
    assert "/results/canonical/runs/" in ignore
    assert "/representation_cache/" in ignore
    assert "*.ckpt" in ignore
    assert "*.safetensors" in ignore
