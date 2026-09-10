"""Explicit target-bearing finalization for the ds006780 cohort."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping

from .ds006780 import sha256_file
from ..research.strict_json import (
    canonical_sha256,
    validate_schema,
    write_create_only_json,
)


HBN_AGE_SUPPORT_YEARS = (5.06, 21.67)


class Ds006780CohortError(ValueError):
    """Raised when target-bearing cohort finalization is unsafe."""


def _normalize_subject_id(value: object) -> str:
    normalized = str(value).lstrip("\ufeff").strip(" \t\r\n")
    if not normalized.startswith("sub-") or not normalized[4:].isalnum():
        raise Ds006780CohortError(f"invalid participant_id: {value!r}")
    return normalized


def load_participant_ages(path: Path) -> tuple[dict[str, float], dict[str, str], str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or "participant_id" not in reader.fieldnames or "age" not in reader.fieldnames:
                raise Ds006780CohortError("participants.tsv requires participant_id and age")
            rows = list(reader)
    except (OSError, csv.Error) as error:
        raise Ds006780CohortError(f"could not parse participant metadata: {path}") from error
    ages: dict[str, float] = {}
    age_exclusions: dict[str, str] = {}
    for row in rows:
        subject_id = _normalize_subject_id(row.get("participant_id", ""))
        if subject_id in ages or subject_id in age_exclusions:
            raise Ds006780CohortError(f"duplicate normalized participant_id: {subject_id}")
        raw_age = str(row.get("age", "")).strip()
        if not raw_age or raw_age.casefold() in {"n/a", "na", "nan"}:
            age_exclusions[subject_id] = "missing_age"
            continue
        try:
            age = float(raw_age)
        except ValueError:
            age_exclusions[subject_id] = "invalid_age"
            continue
        if not math.isfinite(age) or age <= 0:
            age_exclusions[subject_id] = "invalid_age"
            continue
        ages[subject_id] = age
    return ages, age_exclusions, sha256_file(path)


def finalize_ds006780_cohort(
    target_free_manifest: Mapping[str, Any],
    *,
    participant_metadata_path: Path,
    qc_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ages, age_exclusions, participant_digest = load_participant_ages(participant_metadata_path)
    subjects: dict[str, dict[str, Any]] = {}
    exclusions: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    for candidate in target_free_manifest.get("candidate_runs", []):
        subject_id = str(candidate["subject_id"])
        if subject_id in seen_candidates:
            raise Ds006780CohortError(f"duplicate subject in target-free manifest: {subject_id}")
        seen_candidates.add(subject_id)
        qc = qc_reports.get(subject_id)
        if qc is None:
            raise Ds006780CohortError(f"missing signal QC report: {subject_id}")
        claimed_qc_hash = qc.get("qc_sha256")
        if claimed_qc_hash != canonical_sha256(qc, exclude_fields=("qc_sha256",)):
            raise Ds006780CohortError(f"invalid signal QC hash: {subject_id}")
        if subject_id in age_exclusions:
            reason = age_exclusions[subject_id]
        elif subject_id not in ages:
            reason = "missing_participant_metadata"
        else:
            reason = None
        if reason is not None:
            exclusions.append(
                {
                    "subject_id": subject_id,
                    "reason": reason,
                    "signal_qc_sha256": claimed_qc_hash,
                }
            )
            continue
        age = ages[subject_id]
        eligible = HBN_AGE_SUPPORT_YEARS[0] <= age <= HBN_AGE_SUPPORT_YEARS[1]
        subjects[subject_id] = {
            "signal_qc_sha256": claimed_qc_hash,
            "age_years": age,
            "age_units": "years",
            "age_support_eligible": eligible,
        }
    if len(subjects) + len(exclusions) != len(seen_candidates):
        raise Ds006780CohortError("target-free manifest subject accounting is incomplete")
    exclusions.sort(key=lambda item: (item["subject_id"], item["reason"]))
    return {
        "schema_version": 1,
        "target_free_manifest_sha256": str(target_free_manifest["manifest_sha256"]),
        "participant_metadata_sha256": participant_digest,
        "participant_metadata_path": str(participant_metadata_path),
        "hbn_age_support_years": list(HBN_AGE_SUPPORT_YEARS),
        "subjects": dict(sorted(subjects.items())),
        "exclusions": exclusions,
    }


def write_target_manifest(
    output_path: Path,
    manifest: Mapping[str, Any],
    *,
    schema_path: Path,
) -> dict[str, Any]:
    return write_create_only_json(
        output_path,
        manifest,
        schema_path,
        "manifest_sha256",
    )
