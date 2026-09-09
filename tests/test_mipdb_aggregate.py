from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobench_age.analysis import mipdb_aggregate


def _sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity() -> dict[str, str]:
    return {
        "provider": "NEMAR",
        "dataset_id": "nm000153",
        "release_id": "v1.0.0",
        "source_uri": "https://data.nemar.org/nm000153/v1.0.0/manifest.json",
        "source_manifest_sha256": _sha("source"),
        "mipdb_inventory_sha256": _sha("inventory"),
    }


def test_strict_json_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"outer": {"value": 1, "value": 2}}', encoding="utf-8")

    with pytest.raises(mipdb_aggregate.AggregateError, match="duplicate"):
        mipdb_aggregate.load_json_strict(path)


def test_canonical_json_rejects_nonfinite_and_boolean_counts() -> None:
    with pytest.raises(mipdb_aggregate.AggregateError, match="finite"):
        mipdb_aggregate.canonical_json_bytes({"count": float("nan")})

    with pytest.raises(mipdb_aggregate.AggregateError, match="integer"):
        mipdb_aggregate.validate_integer_count(True, field="count")


def test_candidate_digest_excludes_digest_and_approval_only() -> None:
    body = {
        "schema_version": 1,
        "release_scope_sha256": _sha("scope"),
        "candidate_digest_sha256": _sha("old"),
        "approval_reference": {"candidate_digest_sha256": _sha("old")},
        "count": 5,
    }
    digest = mipdb_aggregate.candidate_digest(body)

    changed = dict(body)
    changed["candidate_digest_sha256"] = _sha("new")
    changed["approval_reference"] = {"reviewer": "different"}
    assert digest == mipdb_aggregate.candidate_digest(changed)

    changed["count"] = 6
    assert digest != mipdb_aggregate.candidate_digest(changed)


def test_source_identity_requires_exact_pinned_release_and_inventory_hash() -> None:
    identity = _identity()
    mipdb_aggregate.validate_source_identity(
        identity,
        expected_inventory_sha256=identity["mipdb_inventory_sha256"],
    )

    for field, value in (
        ("release_id", "latest"),
        ("source_uri", "https://data.nemar.org/nm000153/latest/manifest.json"),
        ("mipdb_inventory_sha256", _sha("different")),
    ):
        invalid = dict(identity)
        invalid[field] = value
        with pytest.raises(mipdb_aggregate.AggregateError):
            mipdb_aggregate.validate_source_identity(
                invalid,
                expected_inventory_sha256=identity["mipdb_inventory_sha256"],
            )


def test_subject_list_normalization_is_sorted_and_rejects_duplicates() -> None:
    assert mipdb_aggregate.canonical_subject_ids([" sub-02 ", "sub-01"]) == (
        "sub-01",
        "sub-02",
    )
    with pytest.raises(mipdb_aggregate.AggregateError, match="duplicate"):
        mipdb_aggregate.canonical_subject_ids(["sub-01", " sub-01 "])


def test_exact_cohort_relationships_are_fail_closed() -> None:
    mipdb_aggregate.validate_cohort_relationships(
        pilot=("sub-01",),
        primary_pre_qc=("sub-02", "sub-03"),
        primary_post_qc=("sub-02",),
        extrapolation=("sub-04",),
        eligible=("sub-01", "sub-02", "sub-03", "sub-04"),
    )

    with pytest.raises(mipdb_aggregate.AggregateError, match="overlap"):
        mipdb_aggregate.validate_cohort_relationships(
            pilot=("sub-01",),
            primary_pre_qc=("sub-01",),
            primary_post_qc=(),
            extrapolation=(),
            eligible=("sub-01",),
        )


def _aggregate_fixture() -> tuple[Path, dict[str, object], dict[str, object], dict[str, object]]:
    participants = Path("participants.tsv")
    draft = {
        "schema_version": 2,
        "status": "draft",
        "subjects": [
            {"subject_id": "sub-01", "age": 5.0, "recordings": ["sub-01/eeg/sub-01_task-block01_eeg.vhdr"]},
            {"subject_id": "sub-02", "age": 15.0, "recordings": ["sub-02/eeg/sub-02_task-block01_eeg.vhdr"]},
            {"subject_id": "sub-03", "age": 25.0, "recordings": ["sub-03/eeg/sub-03_task-block01_eeg.vhdr"]},
            {"subject_id": "sub-04", "age": 35.0, "recordings": ["sub-04/eeg/sub-04_task-block01_eeg.vhdr"]},
        ],
        "cohorts": {
            "pilot": ["sub-01"],
            "primary": ["sub-02", "sub-04"],
            "extrapolation": ["sub-03"],
        },
        "exclusions": [],
    }
    final = {
        "schema_version": 2,
        "status": "finalized",
        "subjects": draft["subjects"],
        "cohorts": {
            "pilot": ["sub-01"],
            "primary": ["sub-02"],
            "extrapolation": ["sub-03"],
        },
        "exclusions": [{"subject_id": "sub-04", "reason": "predeclared_signal_qc_failed"}],
    }
    qc = {
        "schema_version": 1,
        "status": "complete",
        "subjects": [
            {"subject_id": "sub-02", "status": "passed", "window_count": 60},
            {"subject_id": "sub-03", "status": "passed", "window_count": 55},
            {"subject_id": "sub-04", "status": "excluded", "reason": "predeclared_signal_qc_failed"},
        ],
    }
    return participants, draft, final, qc


def test_aggregate_cohort_summary_preserves_pre_and_post_qc_states() -> None:
    _, draft, final, qc = _aggregate_fixture()
    summary = mipdb_aggregate.build_cohort_state_summary(
        metadata_row_count=4,
        draft_manifest=draft,
        final_manifest=final,
        cohort_qc=qc,
    )

    assert summary == {
        "all_metadata_rows": {"count": 0, "suppressed": True},
        "eligible_before_pilot": {"count": 0, "suppressed": True},
        "pilot": {"count": 0, "suppressed": True},
        "primary_pre_qc": {"count": 0, "suppressed": True},
        "primary_post_qc": {"count": 0, "suppressed": True},
        "extrapolation": {"count": 0, "suppressed": True},
    }


def test_recording_and_window_summary_uses_finalized_qc_only() -> None:
    _, draft, final, qc = _aggregate_fixture()
    summary = mipdb_aggregate.build_recording_window_summary(
        draft_manifest=draft,
        final_manifest=final,
        cohort_qc=qc,
    )

    assert summary == {
        "recordings": {"subject_count": 2, "total_count": 2},
        "windows": {"subject_count": 2, "total_count": 115},
    }


def test_recording_summary_rejects_multiple_task_block_recordings() -> None:
    _, draft, final, qc = _aggregate_fixture()
    draft["subjects"][1]["recordings"].append("sub-02/eeg/sub-02_task-block01_eeg.set")  # type: ignore[index]
    with pytest.raises(mipdb_aggregate.AggregateError, match="exactly one"):
        mipdb_aggregate.build_recording_window_summary(
            draft_manifest=draft,
            final_manifest=final,
            cohort_qc=qc,
        )


def test_build_candidate_binds_source_scope_and_self_digest() -> None:
    _, draft, final, qc = _aggregate_fixture()
    source_hashes = {
        "study_lock_sha256": _sha("lock"),
        "protocol_sha256": _sha("protocol"),
        "training_protocol_sha256": _sha("training"),
        "mipdb_manifest_sha256": _sha("inventory"),
        "cohort_qc_sha256": _sha("qc"),
        "preprocessing_sha256": _sha("preprocessing"),
    }
    candidate = mipdb_aggregate.build_candidate(
        metadata_row_count=4,
        draft_manifest=draft,
        final_manifest=final,
        cohort_qc=qc,
        dataset_identity=_identity(),
        source_hashes=source_hashes,
    )

    assert candidate["release_class"] == "controlled_review"
    assert candidate["candidate_digest_sha256"] == mipdb_aggregate.candidate_digest(candidate)
    assert candidate["source_hashes"] == source_hashes
    assert candidate["suppression_policy"] == {
        "minimum_cell_count": 5,
        "cross_tabs_enabled": False,
        "numeric_unit": "integer_count",
    }


def test_demographics_are_aggregated_only_for_primary_and_suppressed() -> None:
    rows = [
        {"subject_id": f"sub-{index:02d}", "sex": "F"}
        for index in range(1, 6)
    ]
    summary = mipdb_aggregate._demographic_summary(rows, tuple(row["subject_id"] for row in rows))
    assert summary == {"sex": [{"category": "female", "count": 5, "suppressed": False}]}

    rows[0]["sex"] = "unreported"
    with pytest.raises(mipdb_aggregate.AggregateError, match="unsupported sex"):
        mipdb_aggregate._demographic_summary(rows, tuple(row["subject_id"] for row in rows))
