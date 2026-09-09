from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results" / "canonical"


def test_canonical_results_have_an_explicit_index() -> None:
    index = json.loads((CANONICAL / "index.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == 2
    assert index["research_focus"]
    assert index["primary_claim"]
    assert index["evidence"]
    assert index["prospective"]["status"] == "complete"
    assert index["prospective"]["primary_subjects"] == 75
    assert index["prospective"]["run_count"] == 40
    assert index["prospective"]["prediction_count"] == 3000
    assert "finalist/final_comparison.json" in index["evidence"]


def test_canonical_results_are_compact_and_portable() -> None:
    files = [path for path in CANONICAL.rglob("*") if path.is_file()]
    assert files
    assert not any(path.suffix in {".ckpt", ".pt", ".pth"} for path in files)
    for path in files:
        if path.suffix not in {".json", ".jsonl", ".md", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert "/workspace/" not in text
        assert "/venv/main/" not in text
        assert "<external-workspace>/" not in text or path.suffix == ".json"
