from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_repository_declares_research_scope_without_paper_packaging_claim() -> None:
    scope = ROOT / "ARTICLE_SCOPE.md"
    readme = ROOT / "README.md"

    assert scope.is_file()
    scope_text = scope.read_text(encoding="utf-8").lower()
    readme_text = readme.read_text(encoding="utf-8").lower()
    assert "frozen reve representations" in scope_text
    assert "limits" in scope_text and "expressive" in scope_text
    assert "primary prospective study" in scope_text
    assert "for the paper" not in scope_text
    assert "for the paper" not in readme_text


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


def test_public_research_surface_does_not_advertise_internal_manuscript() -> None:
    paths = [ROOT / "README.md", ROOT / "ARTICLE_SCOPE.md"] + sorted(
        (ROOT / "docs" / "research").glob("*.md")
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths).casefold()
    for forbidden in (
        "manuscript/",
        "make -c manuscript",
        "when more expressive probes do not generalize",
        "download the compiled pdf",
        "retrospective hbn evidence only",
    ):
        assert forbidden not in text


def test_readme_and_registry_record_completed_compact_evidence_neutrally() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    registry = (ROOT / "docs" / "research" / "article_evidence_registry.md").read_text(
        encoding="utf-8"
    )
    for text in (readme, registry):
        assert "TBD_AFTER_EXECUTION" not in text
        assert "3,000" in text
        assert "40" in text
        assert "75" in text
        assert "7747a16e" in text
    assert "compact canonical evidence" in readme.casefold()
    assert "does not establish equivalence" in registry.casefold()


def test_git_candidate_index_excludes_prohibited_research_artifacts_and_secrets() -> None:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    candidates = [ROOT / line for line in completed.stdout.splitlines() if line]
    prohibited_suffixes = {
        ".bdf",
        ".ckpt",
        ".edf",
        ".eeg",
        ".fdt",
        ".joblib",
        ".npy",
        ".npz",
        ".pkl",
        ".pt",
        ".pth",
        ".safetensors",
        ".set",
        ".vhdr",
        ".vmrk",
    }
    assert not [path for path in candidates if path.suffix.casefold() in prohibited_suffixes]
    assert not [
        path
        for path in candidates
        if path.suffix.casefold() in {".log", ".out"}
        or "representation_cache" in path.parts
        or "results/raw" in path.as_posix()
        or "results/runs" in path.as_posix()
    ]
    text_suffixes = {".csv", ".json", ".md", ".py", ".sh", ".tex", ".toml", ".txt", ".bib"}
    sentinel_fixture_paths = {ROOT / "tests" / "test_results_manifest.py"}
    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in candidates
        if path not in sentinel_fixture_paths
        and (path.suffix.casefold() in text_suffixes or path.name in {"Makefile", ".gitignore"})
    )
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", combined)
    assert "-----BEGIN " + "PRIVATE KEY-----" not in combined
    assert "/" + "workspace/" not in combined
    assert "/" + "Users/" not in combined
