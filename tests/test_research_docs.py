from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_article_scope_and_registry_are_explicit() -> None:
    scope = (ROOT / "ARTICLE_SCOPE.md").read_text(encoding="utf-8")
    registry = (ROOT / "docs/research/article_evidence_registry.md").read_text(encoding="utf-8")

    assert "Stability of age probing in REVE" in scope
    assert "increasingly complex" in scope
    assert "canonical 1000-subject nested manifest" in scope
    assert "screening" in scope and "confirmation" in scope
    assert "Head stability" in registry
    assert "Complexity limits" in registry


def test_protocol_declares_holdout_and_required_evidence() -> None:
    protocol = (ROOT / "docs/research/article_ready_protocol.md").read_text(encoding="utf-8")
    protocol_lower = " ".join(protocol.lower().split())

    for required in (
        "validation-only seed-33 screen",
        "seeds 34 and 35",
        "sealed final test",
        "test score is never used",
        "Pearson",
        "MAE",
        "RMSE",
        "R²",
        "paired subject-level bootstrap",
    ):
        assert required.lower() in protocol_lower


def test_canonical_index_points_to_existing_article_evidence() -> None:
    canonical = ROOT / "results/canonical"
    index = json.loads((canonical / "index.json").read_text(encoding="utf-8"))

    assert index["baseline"] == "mean_linear"
    assert index["protocol"] == "strict"
    for relative in index["evidence"]:
        assert (canonical / relative).exists(), relative
