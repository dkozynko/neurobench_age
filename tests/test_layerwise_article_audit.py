from __future__ import annotations

from pathlib import Path

from scripts.audit_layerwise_article import audit_layerwise_article


ROOT = Path(__file__).resolve().parents[1]


def test_layerwise_article_audit_passes_for_retained_evidence() -> None:
    result = audit_layerwise_article(ROOT)

    assert result["status"] == "pass"
    assert result["subject_count"] == 75
    assert result["seed_count"] == 10
    assert result["comparison_count"] == 3
    assert result["scope"] == "exploratory_secondary"
