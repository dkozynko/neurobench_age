"""Approval-bound publication for MIPDB aggregate evidence."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Mapping

import jsonschema

from .mipdb_aggregate import AggregateError, canonical_sha256, candidate_digest


class ReleaseError(AggregateError):
    """Raised when a controlled aggregate cannot be approved or published."""


_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA_ROOT = _ROOT / "schemas"
LEDGER_ANCHOR_SHA256 = hashlib.sha256(
    b"neurobench-age:mipdb-aggregate-release-ledger:v1"
).hexdigest()


def _validate_schema(name: str, value: Mapping[str, Any]) -> None:
    try:
        schema = json.loads(
            (_SCHEMA_ROOT / name).read_text(encoding="utf-8")
        )
        jsonschema.Draft202012Validator(schema).validate(value)
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as error:
        raise ReleaseError(f"payload does not match {name}") from error


def _expected_candidate_digest(candidate: Mapping[str, Any]) -> str:
    claimed = candidate.get("candidate_digest_sha256")
    actual = candidate_digest(candidate)
    if claimed != actual:
        raise ReleaseError("candidate digest does not match body")
    return actual


def validate_candidate(candidate: Mapping[str, Any]) -> None:
    """Validate the candidate schema and its self-reported digest."""

    _validate_schema("mipdb_aggregate_supplement.schema.json", candidate)
    _expected_candidate_digest(candidate)


def validate_approval(
    candidate: Mapping[str, Any], approval: Mapping[str, Any]
) -> None:
    validate_candidate(candidate)
    _validate_schema("mipdb_aggregate_approval.schema.json", approval)
    candidate_digest_value = _expected_candidate_digest(candidate)
    if approval.get("candidate_digest_sha256") != candidate_digest_value:
        raise ReleaseError("approval candidate digest does not match candidate digest")
    if approval.get("release_scope_sha256") != candidate.get("release_scope_sha256"):
        raise ReleaseError("approval release scope does not match candidate release scope")
    if approval.get("release_class") != "controlled_review":
        raise ReleaseError("approval must be controlled review")
    if approval.get("access_scope") != "controlled_review":
        raise ReleaseError("approval cannot authorize public access")
    for field in (
        "status",
        "dataset_terms_status",
        "privacy_status",
        "disclosure_review_status",
    ):
        if approval.get(field) != "approved":
            raise ReleaseError(f"approval field is not approved: {field}")
    if approval.get("revoked") is not False:
        raise ReleaseError("approval is revoked")


def _validate_ledger_digest(ledger: Mapping[str, Any]) -> None:
    claimed = ledger.get("ledger_sha256")
    actual = canonical_sha256(ledger, exclude_fields={"ledger_sha256"})
    if claimed != actual:
        raise ReleaseError("release ledger digest does not match its body")


def _entry_digest(entry: Mapping[str, Any]) -> str:
    return canonical_sha256(entry, exclude_fields={"entry_sha256"})


def validate_ledger_for_candidate(
    candidate: Mapping[str, Any], ledger: Mapping[str, Any]
) -> None:
    _validate_schema("mipdb_aggregate_release_ledger.schema.json", ledger)
    if ledger.get("anchor_sha256") != LEDGER_ANCHOR_SHA256:
        raise ReleaseError("release ledger anchor does not match the protocol")
    if ledger["ledger_version"] == 1 and ledger["previous_ledger_sha256"] is not None:
        raise ReleaseError("ledger version 1 must not claim a previous ledger")
    if ledger["ledger_version"] > 1 and ledger["previous_ledger_sha256"] is None:
        raise ReleaseError("later ledger versions must name their predecessor")
    _validate_ledger_digest(ledger)
    digest = _expected_candidate_digest(candidate)
    scope = candidate.get("release_scope_sha256")
    active_digests: set[str] = set()
    active_scopes: set[str] = set()
    for entry in ledger["entries"]:
        if entry["entry_sha256"] != _entry_digest(entry):
            raise ReleaseError("release ledger entry digest does not match its body")
        if entry["status"] == "active":
            if entry["candidate_digest_sha256"] in active_digests:
                raise ReleaseError("release ledger contains duplicate active candidates")
            if entry["release_scope_sha256"] in active_scopes:
                raise ReleaseError("release ledger contains duplicate active scopes")
            active_digests.add(entry["candidate_digest_sha256"])
            active_scopes.add(entry["release_scope_sha256"])
        if (
            entry["status"] == "active"
            and (
                entry["candidate_digest_sha256"] == digest
                or entry["release_scope_sha256"] == scope
            )
        ):
            raise ReleaseError("candidate or release scope already exists in ledger")


def publish_candidate(
    candidate: Mapping[str, Any], approval: Mapping[str, Any], output_path: Path
) -> None:
    """Publish one approved candidate exactly once."""

    validate_approval(candidate, approval)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(candidate, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise ReleaseError(f"published aggregate already exists: {output}") from error
