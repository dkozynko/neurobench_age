from __future__ import annotations

import copy
from pathlib import Path

import pytest

from neurobench_age.analysis import mipdb_aggregate_release as release
from neurobench_age.analysis import mipdb_aggregate
from tests.test_mipdb_aggregate_schema import _approval, _candidate, _ledger


def test_approval_must_match_candidate_scope_and_digest() -> None:
    candidate = _candidate()
    candidate["candidate_digest_sha256"] = mipdb_aggregate.candidate_digest(candidate)
    approval = _approval()
    approval["candidate_digest_sha256"] = candidate["candidate_digest_sha256"]
    release.validate_approval(candidate, approval)

    invalid = copy.deepcopy(approval)
    invalid["candidate_digest_sha256"] = "0" * 64
    with pytest.raises(release.ReleaseError, match="candidate digest"):
        release.validate_approval(candidate, invalid)


def test_approval_cannot_authorize_public_access() -> None:
    candidate = _candidate()
    candidate["candidate_digest_sha256"] = mipdb_aggregate.candidate_digest(candidate)
    approval = _approval()
    approval["candidate_digest_sha256"] = candidate["candidate_digest_sha256"]
    approval["access_scope"] = "public"
    with pytest.raises(release.ReleaseError, match="payload"):
        release.validate_approval(candidate, approval)


def test_ledger_rejects_duplicate_active_scope() -> None:
    candidate = _candidate()
    candidate["candidate_digest_sha256"] = mipdb_aggregate.candidate_digest(candidate)
    approval = _approval()
    approval["candidate_digest_sha256"] = candidate["candidate_digest_sha256"]
    ledger = _ledger()
    ledger["entries"][0]["candidate_digest_sha256"] = candidate["candidate_digest_sha256"]  # type: ignore[index]
    ledger["entries"][0]["entry_sha256"] = mipdb_aggregate.canonical_sha256(  # type: ignore[index]
        ledger["entries"][0], exclude_fields={"entry_sha256"}  # type: ignore[arg-type]
    )
    ledger["ledger_sha256"] = mipdb_aggregate.canonical_sha256(
        ledger, exclude_fields={"ledger_sha256"}
    )
    release.validate_approval(candidate, approval)

    with pytest.raises(release.ReleaseError, match="release scope"):
        release.validate_ledger_for_candidate(candidate, ledger)


def test_publish_is_create_only(tmp_path: Path) -> None:
    candidate = _candidate()
    candidate["candidate_digest_sha256"] = mipdb_aggregate.candidate_digest(candidate)
    approval = _approval()
    approval["candidate_digest_sha256"] = candidate["candidate_digest_sha256"]
    output = tmp_path / "published.json"
    release.publish_candidate(candidate, approval, output)
    assert output.is_file()
    with pytest.raises(release.ReleaseError, match="already exists"):
        release.publish_candidate(candidate, approval, output)
