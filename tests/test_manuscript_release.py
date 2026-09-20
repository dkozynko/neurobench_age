from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *arguments],
        capture_output=True, text=True,
    )


def test_presentation_preserves_numeric_cells_and_original_evidence(tmp_path: Path) -> None:
    source = ROOT / "results/extensions/capacity_data_regime_v3"
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    output = tmp_path / "presentation"
    result = _run("build_submission_assets.py", "--repository-root", str(ROOT), "--output", str(output))
    assert result.returncode == 0, result.stderr
    original = (source / "capacity_data_regime_contrasts.tex").read_text()
    rendered = (output / "capacity_data_regime_contrasts.tex").read_text()
    original_cells = [row.split(" & ")[2:] for row in original.splitlines() if " & " in row][1:]
    rendered_cells = [row.split(" & ")[2:] for row in rendered.splitlines() if " & " in row][1:]
    assert original_cells == rendered_cells
    assert "800 minus 200" in rendered and "endpoint\\_" not in rendered
    assert "4 units" in rendered and "hidden\\_dim" not in rendered
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    macros = (output / "head_parameters.tex").read_text()
    assert r"\newcommand{\QueryParameters}{6657}" in macros


def test_presentation_refuses_modified_source_assets(tmp_path: Path) -> None:
    source = tmp_path / "results/extensions/capacity_data_regime_v3"
    shutil.copytree(ROOT / "results/extensions/capacity_data_regime_v3", source)
    shutil.copytree(ROOT / "results/canonical/prospective", tmp_path / "results/canonical/prospective")
    with (source / "capacity_data_regime_cells.csv").open("a") as handle:
        handle.write("modified\n")
    result = _run("build_submission_assets.py", "--repository-root", str(tmp_path), "--output", str(tmp_path / "output"))
    assert result.returncode != 0
    assert "hash mismatch" in result.stderr.lower()
    assert not (tmp_path / "output").exists()


def test_arxiv_archive_has_complete_dependency_closure_without_private_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    manuscript = repo / "manuscript"
    (manuscript / "sections").mkdir(parents=True)
    (repo / "results").mkdir()
    (manuscript / "main.tex").write_text(r"\documentclass{article}\input{sections/body}\bibliography{references}")
    (manuscript / "sections/body.tex").write_text(r"\input{../results/table}\includegraphics{../results/figure.pdf}")
    (manuscript / "references.bib").write_text("@misc{example,title={Example}}")
    (repo / "results/table.tex").write_text("An aggregate table")
    (repo / "results/figure.pdf").write_bytes(b"%PDF-aggregate-figure")
    (repo / "results/private.json").write_text('{"subject_id":"not-for-release"}')
    (manuscript / "main.log").write_text("a private path")
    archive = tmp_path / "source.zip"
    result = _run("build_arxiv_source.py", "--repository-root", str(repo), "--output", str(archive))
    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {
            "main.tex", "sections/body.tex", "references.bib", "assets/table.tex", "assets/figure.pdf",
        }
        assert "../results" not in bundle.read("sections/body.tex").decode()
        assert r"\input{assets/table.tex}" in bundle.read("sections/body.tex").decode()


def test_arxiv_archive_rejects_missing_dependencies(tmp_path: Path) -> None:
    (tmp_path / "manuscript").mkdir()
    (tmp_path / "manuscript/main.tex").write_text(r"\input{missing}")
    archive = tmp_path / "source.zip"
    result = _run("build_arxiv_source.py", "--repository-root", str(tmp_path), "--output", str(archive))
    assert result.returncode != 0 and not archive.exists()
    assert "missing" in result.stderr.lower()
