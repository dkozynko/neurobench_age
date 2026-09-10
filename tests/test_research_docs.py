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


def test_layerwise_extension_is_registered_with_bounded_interpretation() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    registry = (ROOT / "docs/research/article_evidence_registry.md").read_text(
        encoding="utf-8"
    )
    for text in (readme, registry):
        assert "layer-wise" in text.casefold()
        assert "f1f757ef53a8" in text
        assert "14b5afe5623f" in text
        assert "exploratory" in text.casefold()
    assert "does not modify the primary confirmatory claim" in registry


def test_clean_server_reproduction_guide_is_public_and_complete() -> None:
    guide = (ROOT / "docs/research/reproduction.md").read_text(encoding="utf-8")
    normalized = " ".join(guide.split())

    for required in (
        "article_ready_protocol.md",
        "clean checkout",
        "uv",
        "access-controlled inputs",
        "aggregate artifacts",
        "participant-level predictions",
        "make --directory=manuscript verify",
    ):
        assert required.casefold() in normalized.casefold()

    forbidden_paths = ("/" + "Users/", "/" + "home/")
    for forbidden in ("ssh -p", "hf_", "root@", *forbidden_paths):
        assert forbidden.casefold() not in guide.casefold()
