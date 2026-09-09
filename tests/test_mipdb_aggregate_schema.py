from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = ROOT / "schemas"


def _schema(name: str) -> dict[str, object]:
    return json.loads((SCHEMA_ROOT / name).read_text(encoding="utf-8"))


def _validate(name: str, value: dict[str, object]) -> None:
    jsonschema.Draft202012Validator(_schema(name)).validate(value)


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _candidate() -> dict[str, object]:
    return {
        "schema_version": 1,
        "release_class": "controlled_review",
        "dataset_identity": {
            "provider": "NEMAR",
            "dataset_id": "nm000153",
            "release_id": "v1.0.0",
            "source_uri": "https://data.nemar.org/nm000153/v1.0.0/manifest.json",
            "source_manifest_sha256": _sha("a"),
            "mipdb_inventory_sha256": _sha("b"),
        },
        "source_hashes": {
            "study_lock_sha256": _sha("c"),
            "protocol_sha256": _sha("d"),
            "training_protocol_sha256": _sha("e"),
            "mipdb_manifest_sha256": _sha("f"),
            "cohort_qc_sha256": _sha("g"),
            "preprocessing_sha256": _sha("h"),
        },
        "cohort_states": {
            "all_metadata_rows": {"count": 100, "suppressed": False},
            "eligible_before_pilot": {"count": 90, "suppressed": False},
            "pilot": {"count": 10, "suppressed": False},
            "primary_pre_qc": {"count": 60, "suppressed": False},
            "primary_post_qc": {"count": 55, "suppressed": False},
            "extrapolation": {"count": 20, "suppressed": False},
        },
        "age_bins": [
            {"label": "0-9", "count": 10, "suppressed": False},
            {"label": "10-19", "count": 20, "suppressed": False},
        ],
        "demographics": {
            "sex": [
                {"category": "female", "count": 10, "suppressed": False},
                {"category": "male", "count": 10, "suppressed": False},
                {"category": "unknown", "count": 5, "suppressed": False},
            ]
        },
        "recording_window_summary": {
            "recordings": {"subject_count": 55, "total_count": 55},
            "windows": {"subject_count": 55, "total_count": 3300},
        },
        "suppression_policy": {
            "minimum_cell_count": 5,
            "cross_tabs_enabled": False,
            "numeric_unit": "integer_count",
        },
        "release_scope_sha256": _sha("i"),
        "candidate_digest_sha256": _sha("j"),
    }


def _approval() -> dict[str, object]:
    return {
        "schema_version": 1,
        "candidate_digest_sha256": _sha("j"),
        "release_scope_sha256": _sha("i"),
        "release_class": "controlled_review",
        "access_scope": "controlled_review",
        "status": "approved",
        "dataset_terms_status": "approved",
        "privacy_status": "approved",
        "disclosure_review_status": "approved",
        "reviewer_id": "reviewer-1",
        "reviewed_at_utc": "2026-09-09T12:00:00+00:00",
        "audience": "named_reviewers",
        "revoked": False,
    }


def _ledger() -> dict[str, object]:
    entry = {
        "candidate_digest_sha256": _sha("j"),
        "release_scope_sha256": _sha("i"),
        "release_class": "controlled_review",
        "audience": "named_reviewers",
        "status": "active",
    }
    return {
        "schema_version": 1,
        "ledger_version": 1,
        "anchor_sha256": _sha("neurobench-age:mipdb-aggregate-release-ledger:v1"),
        "previous_ledger_sha256": None,
        "entries": [{**entry, "entry_sha256": _sha(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")}],
        "ledger_sha256": _sha("l"),
    }


def test_valid_candidate_approval_and_ledger_match_strict_schemas() -> None:
    _validate("mipdb_aggregate_supplement.schema.json", _candidate())
    _validate("mipdb_aggregate_approval.schema.json", _approval())
    _validate("mipdb_aggregate_release_ledger.schema.json", _ledger())


@pytest.mark.parametrize(
    "field",
    ["subject_id", "exact_age", "recording_path", "prediction"],
)
def test_candidate_rejects_participant_level_fields(field: str) -> None:
    candidate = _candidate()
    candidate[field] = "forbidden"
    with pytest.raises(jsonschema.ValidationError):
        _validate("mipdb_aggregate_supplement.schema.json", candidate)


def test_candidate_rejects_unknown_nested_fields() -> None:
    candidate = _candidate()
    candidate["dataset_identity"] = {
        **candidate["dataset_identity"],  # type: ignore[misc]
        "unlisted": "no",
    }
    with pytest.raises(jsonschema.ValidationError):
        _validate("mipdb_aggregate_supplement.schema.json", candidate)


def test_candidate_rejects_public_release_class_and_cross_tabs() -> None:
    candidate = _candidate()
    candidate["release_class"] = "public"
    with pytest.raises(jsonschema.ValidationError):
        _validate("mipdb_aggregate_supplement.schema.json", candidate)

    candidate = _candidate()
    candidate["suppression_policy"] = {
        **candidate["suppression_policy"],  # type: ignore[misc]
        "cross_tabs_enabled": True,
    }
    with pytest.raises(jsonschema.ValidationError):
        _validate("mipdb_aggregate_supplement.schema.json", candidate)


def test_approval_requires_all_three_independent_reviews() -> None:
    approval = _approval()
    approval["privacy_status"] = "pending"
    with pytest.raises(jsonschema.ValidationError):
        _validate("mipdb_aggregate_approval.schema.json", approval)


def test_ledger_rejects_unknown_audience_and_duplicate_scope() -> None:
    ledger = _ledger()
    ledger["entries"] = [
        {
            **ledger["entries"][0],  # type: ignore[index]
            "audience": "everyone",
        }
    ]
    with pytest.raises(jsonschema.ValidationError):
        _validate("mipdb_aggregate_release_ledger.schema.json", ledger)

    duplicate = copy.deepcopy(_ledger())
    duplicate["entries"] = [duplicate["entries"][0], duplicate["entries"][0]]  # type: ignore[index]
    with pytest.raises(jsonschema.ValidationError):
        _validate("mipdb_aggregate_release_ledger.schema.json", duplicate)
