from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _research_docs() -> str:
    paths = (
        ROOT / "README.md",
        ROOT / "ARTICLE_SCOPE.md",
        ROOT / "docs/research/article_ready_protocol.md",
        ROOT / "docs/research/article_evidence_registry.md",
        ROOT / "results/canonical/README.md",
        ROOT / "results/canonical/finalist/README.md",
    )
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def test_docs_use_the_correct_evidence_taxonomy() -> None:
    text = _normalized(_research_docs())

    assert "primary prospective study" in text
    assert "secondary reproduction evidence" in text
    assert "retrospective secondary evidence" in text
    assert "hbn r5" in text
    assert "must not be used for model or head selection" in text
    assert "official neuralbench full fine-tuning" in text
    assert "end-to-end age prediction" in text
    assert "primary stability evidence" not in text
    assert "cannot provide untouched confirmation" in text


def test_docs_do_not_present_the_repository_as_a_paper_package() -> None:
    text = _research_docs().lower()

    for forbidden in (
        "for the paper",
        "paper's claims",
        "required by the paper",
        "used by the paper",
        "support the paper",
    ):
        assert forbidden not in text


def test_protocol_lists_only_real_commands_and_completed_adapters() -> None:
    protocol = (
        ROOT / "docs/research/article_ready_protocol.md"
    ).read_text(encoding="utf-8")
    normalized = _normalized(protocol)

    for script in (
        "scripts/build_mipdb_manifest.py",
        "scripts/run_mipdb_pilot.py",
        "scripts/finalize_mipdb_cohort.py",
        "scripts/cache_hbn_representations.py",
        "scripts/run_frozen_probe.py",
        "scripts/seal_external_study.py",
        "scripts/run_external_holdout.py",
        "scripts/analyze_confirmatory.py",
    ):
        assert script in protocol
        assert (ROOT / script).is_file()

    assert "integration blocker" not in protocol.lower()
    assert '--bids-root "$MIPDB_ROOT"' in protocol
    assert '--mapping "$REVE_CHANNEL_MAPPING"' in protocol
    assert '--subject-manifest "$HBN_SUBJECT_MANIFEST"' in protocol
    assert "must not be precomputed before the study start marker" in normalized


def test_docs_use_placeholders_and_bound_negative_claims() -> None:
    text = _research_docs()
    lower = _normalized(text)

    assert "TBD_AFTER_EXECUTION" in text
    assert (
        "no tested complex head established a stable external gain under the "
        "predeclared protocol"
    ) in lower
    assert "does not establish equivalence" in lower
    assert "cross-dataset shift" in lower
    assert "encoder pretraining uncertainty" in lower
