from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repository_declares_article_scope() -> None:
    scope = ROOT / "ARTICLE_SCOPE.md"
    readme = ROOT / "README.md"

    assert scope.is_file()
    scope_text = scope.read_text(encoding="utf-8").lower()
    readme_text = readme.read_text(encoding="utf-8").lower()
    assert "stability of age probing" in scope_text
    assert "limits" in scope_text and "complex" in scope_text
    assert "canonical" in scope_text
    assert "article" in readme_text


def test_package_layout_and_article_entry_points_are_declared() -> None:
    package_root = ROOT / "src" / "neurobench_age"
    pyproject = ROOT / "pyproject.toml"

    assert pyproject.is_file()
    assert (package_root / "__init__.py").is_file()
    assert (package_root / "core").is_dir()
    assert (package_root / "heads").is_dir()
    assert (package_root / "data").is_dir()
    assert (package_root / "pipelines").is_dir()
    assert (package_root / "training").is_dir()

    metadata = json.loads((ROOT / "configs" / "article" / "validation_cycle.json").read_text())
    assert metadata["evaluation_mode"] == "validation_only"
    assert metadata["test_access"] == "sealed"


def test_repository_root_has_no_experiment_implementation_files() -> None:
    allowed_files = {
        ".gitignore",
        "ARTICLE_SCOPE.md",
        "README.md",
        "pyproject.toml",
        "uv.lock",
    }
    root_files = {path.name for path in ROOT.iterdir() if path.is_file()}

    assert root_files <= allowed_files
    assert not (ROOT / "neurobench_age").exists()
