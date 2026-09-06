from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_research_scope_and_registry_are_explicit() -> None:
    scope = (ROOT / "ARTICLE_SCOPE.md").read_text(encoding="utf-8")
    registry = (ROOT / "docs/research/article_evidence_registry.md").read_text(encoding="utf-8")
    scope_normalized = " ".join(scope.split())

    assert "frozen REVE representations" in scope_normalized
    assert "increasingly expressive" in scope_normalized
    assert "MIPDB" in scope_normalized
    assert "prospective" in scope_normalized and "retrospective" in scope_normalized
    assert "Primary prospective study" in registry
    assert "Retrospective HBN/R5 boundary" in registry


def test_protocol_declares_holdout_and_required_evidence() -> None:
    protocol = (ROOT / "docs/research/article_ready_protocol.md").read_text(encoding="utf-8")
    protocol_lower = " ".join(protocol.lower().split())

    for required in (
        "seeds 33 through 42",
        "sealed external holdout",
        "MIPDB must never be used",
        "Pearson",
        "MAE",
        "RMSE",
        "R²",
        "hierarchical paired bootstrap",
    ):
        assert required.lower() in protocol_lower


def test_canonical_index_points_to_existing_article_evidence() -> None:
    canonical = ROOT / "results/canonical"
    index = json.loads((canonical / "index.json").read_text(encoding="utf-8"))

    assert index["baseline"] == "mean_linear"
    assert index["protocol"] == "strict"
    for relative in index["evidence"]:
        assert (canonical / relative).exists(), relative
