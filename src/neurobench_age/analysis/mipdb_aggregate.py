"""Disclosure-safe aggregate evidence for the sealed MIPDB cohort."""

from __future__ import annotations

import hashlib
import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


class AggregateError(ValueError):
    """Raised when aggregate inputs or release evidence are not auditable."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_RE = re.compile(r"^(?:v[0-9]+\.[0-9]+\.[0-9]+|snapshot-[A-Za-z0-9._-]+)$")
_DIGEST_EXCLUDED_FIELDS = frozenset({"candidate_digest_sha256", "approval_reference"})
_MINIMUM_CELL_COUNT = 5
_TASK_RECORDING_TOKEN = "_task-block01_eeg."


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AggregateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise AggregateError(f"JSON number must be finite: {value}")


def load_json_strict(path: Path) -> Any:
    """Load JSON while rejecting duplicate keys and non-finite constants."""

    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except AggregateError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise AggregateError(f"could not load strict JSON: {path}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise AggregateError(f"could not hash file: {path}") from error
    return digest.hexdigest()


def load_participants_tsv(path: Path) -> tuple[dict[str, Any], ...]:
    """Load metadata rows while retaining non-identifying exclusion reasons."""

    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    except (OSError, csv.Error) as error:
        raise AggregateError(f"could not parse participants.tsv: {path}") from error
    if not rows or not {"participant_id", "age"} <= set(rows[0]):
        raise AggregateError("participants.tsv requires participant_id and age columns")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        subject_id = str(row.get("participant_id", "")).strip()
        if not subject_id or subject_id in seen:
            raise AggregateError("participants.tsv contains duplicate or empty subject IDs")
        seen.add(subject_id)
        raw_age = str(row.get("age", "")).strip()
        try:
            age = float(raw_age)
            age_reason = None if math.isfinite(age) and age > 0 else "invalid_age"
        except ValueError:
            age = None
            age_reason = "missing_age" if not raw_age else "invalid_age"
        normalized = {"subject_id": subject_id, "age": age, "age_reason": age_reason}
        if "sex" in row:
            normalized["sex"] = row.get("sex")
        result.append(normalized)
    return tuple(result)


def validate_completed_study_lock(
    lock_path: Path, *, source_hashes: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Validate the immutable study lock and completed lifecycle state."""

    from ..research.study_lock import load_study_lock

    try:
        lock = load_study_lock(Path(lock_path))
        state = load_json_strict(Path(lock_path).with_name("study_state.json"))
    except AggregateError:
        raise
    except Exception as error:
        raise AggregateError("could not validate study lock") from error
    if not isinstance(state, Mapping):
        raise AggregateError("study state must be an object")
    if (
        state.get("schema_version") != 1
        or state.get("study_id") != lock["study_id"]
        or state.get("lock_sha256") != lock["lock_sha256"]
        or state.get("state") != "completed"
    ):
        raise AggregateError("study state is not the completed state for this lock")
    if lock.get("git_dirty") is not False:
        raise AggregateError("study lock must have git_dirty=false")
    if source_hashes.get("study_lock_sha256") != lock.get("lock_sha256"):
        raise AggregateError("source hash differs from study lock: study_lock_sha256")
    expected_fields = {
        "protocol_sha256": "protocol_sha256",
        "training_protocol_sha256": "training_protocol_sha256",
        "mipdb_manifest_sha256": "mipdb_manifest_sha256",
        "cohort_qc_sha256": "mipdb_cohort_qc_sha256",
        "preprocessing_sha256": "preprocessing_sha256",
    }
    for source_field, lock_field in expected_fields.items():
        if source_hashes.get(source_field) != lock.get(lock_field):
            raise AggregateError(f"source hash differs from study lock: {source_field}")
    return lock


def _validate_finite(value: Any, *, path: str = "$", reject_bool: bool = False) -> None:
    if isinstance(value, bool):
        if reject_bool:
            raise AggregateError(f"{path} must be an integer, not boolean")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise AggregateError(f"{path} must be finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite(item, path=f"{path}.{key}", reject_bool=reject_bool)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite(item, path=f"{path}[{index}]", reject_bool=reject_bool)


def canonical_json_bytes(value: Mapping[str, Any], *, exclude_fields: Iterable[str] = ()) -> bytes:
    """Serialize a mapping into the repository's canonical JSON byte form."""

    if not isinstance(value, Mapping):
        raise AggregateError("canonical JSON root must be an object")
    _validate_finite(value)
    excluded = set(exclude_fields)
    body = {str(key): item for key, item in value.items() if key not in excluded}
    try:
        encoded = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError) as error:
        raise AggregateError("value cannot be canonically serialized") from error
    return encoded + b"\n"


def canonical_sha256(value: Mapping[str, Any], *, exclude_fields: Iterable[str] = ()) -> str:
    return hashlib.sha256(
        canonical_json_bytes(value, exclude_fields=exclude_fields)
    ).hexdigest()


def candidate_digest(candidate: Mapping[str, Any]) -> str:
    """Hash the candidate body without its self-reference or approval pointer."""

    return canonical_sha256(candidate, exclude_fields=_DIGEST_EXCLUDED_FIELDS)


def validate_integer_count(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AggregateError(f"{field} must be an integer")
    if value < 0:
        raise AggregateError(f"{field} must be non-negative")
    return value


def _validate_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise AggregateError(f"{field} must be a lowercase SHA-256 digest")
    return value


def validate_source_identity(
    identity: Mapping[str, Any], *, expected_inventory_sha256: str
) -> None:
    """Validate a pinned NEMAR identity against the sealed inventory digest."""

    required = {
        "provider",
        "dataset_id",
        "release_id",
        "source_uri",
        "source_manifest_sha256",
        "mipdb_inventory_sha256",
    }
    if set(identity) != required:
        raise AggregateError("source identity fields are not exact")
    if identity["provider"] != "NEMAR" or identity["dataset_id"] != "nm000153":
        raise AggregateError("source identity is not MIPDB nm000153 on NEMAR")
    release_id = identity["release_id"]
    if not isinstance(release_id, str) or not _RELEASE_RE.fullmatch(release_id):
        raise AggregateError("release_id must be a concrete pinned version or snapshot")
    parsed = urlparse(str(identity["source_uri"]))
    expected_prefix = f"/nm000153/{release_id}/"
    if (
        parsed.scheme != "https"
        or parsed.netloc != "data.nemar.org"
        or not parsed.path.startswith(expected_prefix)
    ):
        raise AggregateError("source_uri must be a versioned NEMAR data URI")
    _validate_sha256(identity["source_manifest_sha256"], field="source_manifest_sha256")
    inventory_sha = _validate_sha256(
        identity["mipdb_inventory_sha256"], field="mipdb_inventory_sha256"
    )
    _validate_sha256(expected_inventory_sha256, field="expected_inventory_sha256")
    if inventory_sha != expected_inventory_sha256:
        raise AggregateError("source inventory digest differs from sealed MIPDB manifest")


def canonical_subject_ids(values: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            raise AggregateError("subject IDs must be strings")
        subject_id = raw.strip()
        if not subject_id:
            raise AggregateError("subject IDs cannot be empty")
        if subject_id in seen:
            raise AggregateError(f"duplicate subject ID: {subject_id}")
        seen.add(subject_id)
        normalized.append(subject_id)
    return tuple(sorted(normalized))


def validate_cohort_relationships(
    *,
    pilot: Sequence[str],
    primary_pre_qc: Sequence[str],
    primary_post_qc: Sequence[str],
    extrapolation: Sequence[str],
    eligible: Sequence[str],
) -> None:
    pilot_ids = set(canonical_subject_ids(pilot))
    primary_ids = set(canonical_subject_ids(primary_pre_qc))
    post_qc_ids = set(canonical_subject_ids(primary_post_qc))
    extrapolation_ids = set(canonical_subject_ids(extrapolation))
    eligible_ids = set(canonical_subject_ids(eligible))
    groups = (pilot_ids, primary_ids, extrapolation_ids)
    if any(left & right for index, left in enumerate(groups) for right in groups[index + 1 :]):
        raise AggregateError("cohort groups overlap")
    if not pilot_ids | primary_ids | extrapolation_ids <= eligible_ids:
        raise AggregateError("cohort group is outside eligible subjects")
    if not post_qc_ids <= primary_ids:
        raise AggregateError("post-QC primary subjects are outside primary pre-QC")


def _count_cell(count: int) -> dict[str, Any]:
    validate_integer_count(count, field="aggregate count")
    if count < _MINIMUM_CELL_COUNT:
        return {"count": 0, "suppressed": True}
    return {"count": count, "suppressed": False}


def _manifest_subjects(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw_subjects = manifest.get("subjects")
    if not isinstance(raw_subjects, list):
        raise AggregateError("manifest subjects must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for item in raw_subjects:
        if not isinstance(item, Mapping) or not isinstance(item.get("subject_id"), str):
            raise AggregateError("manifest contains an invalid subject record")
        subject_id = item["subject_id"].strip()
        if not subject_id or subject_id in result:
            raise AggregateError("manifest contains duplicate or empty subject IDs")
        recordings = item.get("recordings")
        if not isinstance(recordings, list) or not all(
            isinstance(path, str) and path.strip() for path in recordings
        ):
            raise AggregateError(f"recordings are invalid for {subject_id}")
        result[subject_id] = item
    return result


def _manifest_cohort(manifest: Mapping[str, Any], name: str) -> tuple[str, ...]:
    cohorts = manifest.get("cohorts")
    if not isinstance(cohorts, Mapping) or name not in cohorts:
        raise AggregateError(f"manifest is missing cohort: {name}")
    values = cohorts[name]
    if not isinstance(values, list):
        raise AggregateError(f"manifest cohort is not an array: {name}")
    return canonical_subject_ids(values)


def _validate_manifest_status(manifest: Mapping[str, Any], expected: str) -> None:
    if manifest.get("status") != expected:
        raise AggregateError(f"manifest status must be {expected}")


def _qc_rows(cohort_qc: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if cohort_qc.get("status") != "complete":
        raise AggregateError("cohort QC must be complete")
    rows = cohort_qc.get("subjects")
    if not isinstance(rows, list):
        raise AggregateError("cohort QC subjects must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("subject_id"), str):
            raise AggregateError("cohort QC contains an invalid subject row")
        subject_id = row["subject_id"].strip()
        if not subject_id or subject_id in result:
            raise AggregateError("cohort QC contains duplicate or empty subject IDs")
        result[subject_id] = row
    return result


def build_cohort_state_summary(
    *,
    metadata_row_count: int,
    draft_manifest: Mapping[str, Any],
    final_manifest: Mapping[str, Any],
    cohort_qc: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build locked cohort-state counts, suppressing cells smaller than k."""

    metadata_count = validate_integer_count(metadata_row_count, field="metadata_row_count")
    _validate_manifest_status(draft_manifest, "draft")
    _validate_manifest_status(final_manifest, "finalized")
    draft_subjects = _manifest_subjects(draft_manifest)
    final_subjects = _manifest_subjects(final_manifest)
    if set(draft_subjects) != set(final_subjects):
        raise AggregateError("final manifest changed eligible subject membership")
    pilot = _manifest_cohort(draft_manifest, "pilot")
    primary_pre_qc = _manifest_cohort(draft_manifest, "primary")
    extrapolation_pre_qc = _manifest_cohort(draft_manifest, "extrapolation")
    primary_post_qc = _manifest_cohort(final_manifest, "primary")
    extrapolation_post_qc = _manifest_cohort(final_manifest, "extrapolation")
    eligible = tuple(sorted(draft_subjects))
    validate_cohort_relationships(
        pilot=pilot,
        primary_pre_qc=primary_pre_qc,
        primary_post_qc=primary_post_qc,
        extrapolation=extrapolation_pre_qc,
        eligible=eligible,
    )
    if not set(extrapolation_post_qc) <= set(extrapolation_pre_qc):
        raise AggregateError("final extrapolation cohort is outside pre-QC cohort")
    qc = _qc_rows(cohort_qc)
    expected_qc = set(primary_pre_qc) | set(extrapolation_pre_qc)
    if set(qc) != expected_qc:
        raise AggregateError("cohort QC rows do not match candidate cohorts")
    passed = {
        subject_id
        for subject_id, row in qc.items()
        if row.get("status") == "passed"
    }
    if passed != set(primary_post_qc) | set(extrapolation_post_qc):
        raise AggregateError("final cohorts do not match passed QC subjects")
    return {
        "all_metadata_rows": _count_cell(metadata_count),
        "eligible_before_pilot": _count_cell(len(eligible)),
        "pilot": _count_cell(len(pilot)),
        "primary_pre_qc": _count_cell(len(primary_pre_qc)),
        "primary_post_qc": _count_cell(len(primary_post_qc)),
        "extrapolation": _count_cell(len(extrapolation_post_qc)),
    }


def build_recording_window_summary(
    *,
    draft_manifest: Mapping[str, Any],
    final_manifest: Mapping[str, Any],
    cohort_qc: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    """Summarize recording/window counts from the finalized target-free QC."""

    draft_subjects = _manifest_subjects(draft_manifest)
    final_subjects = _manifest_subjects(final_manifest)
    qc = _qc_rows(cohort_qc)
    selected = set(_manifest_cohort(final_manifest, "primary")) | set(
        _manifest_cohort(final_manifest, "extrapolation")
    )
    if not selected <= set(draft_subjects):
        raise AggregateError("final QC cohort contains an unknown subject")
    total_windows = 0
    for subject_id in sorted(selected):
        subject = draft_subjects[subject_id]
        recordings = [
            path for path in subject["recordings"] if _TASK_RECORDING_TOKEN in Path(path).name
        ]
        if len(recordings) != 1:
            raise AggregateError(
                f"{subject_id} requires exactly one task-block01 recording"
            )
        if subject_id not in final_subjects:
            raise AggregateError("final manifest lost a selected subject")
        row = qc.get(subject_id)
        if row is None or row.get("status") != "passed":
            raise AggregateError("selected subject lacks a passed QC row")
        window_count = validate_integer_count(
            row.get("window_count"), field=f"window_count[{subject_id}]"
        )
        total_windows += window_count
    subject_count = len(selected)
    return {
        "recordings": {"subject_count": subject_count, "total_count": subject_count},
        "windows": {"subject_count": subject_count, "total_count": total_windows},
    }


def _age_bin(age: Any) -> str:
    if age is None or isinstance(age, bool):
        return "unknown"
    try:
        numeric = float(age)
    except (TypeError, ValueError) as error:
        raise AggregateError("age must be numeric or missing") from error
    if not math.isfinite(numeric) or numeric < 0:
        raise AggregateError("age must be finite and non-negative")
    if numeric < 10:
        return "0-9"
    if numeric < 20:
        return "10-19"
    if numeric < 30:
        return "20-29"
    if numeric < 40:
        return "30-39"
    if numeric < 50:
        return "40-49"
    if numeric < 60:
        return "50-59"
    if numeric < 70:
        return "60-69"
    return "70+"


def _counted_bins(values: Iterable[str]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    order = ("0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+", "unknown")
    cells = [
        {"label": label, **_count_cell(counts[label])}
        for label in order
        if label in counts
    ]
    return _secondary_suppress_cells(cells)


def _secondary_suppress_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prevent a single suppressed cell from being recovered by subtraction."""

    suppressed = [index for index, cell in enumerate(cells) if cell["suppressed"]]
    if len(suppressed) != 1:
        return cells
    candidates = [
        (int(cell["count"]), index)
        for index, cell in enumerate(cells)
        if not cell["suppressed"]
    ]
    if not candidates:
        return cells
    _, index = max(candidates)
    cells[index] = {**cells[index], "count": 0, "suppressed": True}
    return cells


def _normalize_sex(value: Any) -> str:
    """Map the small, declared sex vocabulary to disclosure-safe categories."""

    if value is None:
        return "unknown"
    normalized = str(value).strip().casefold()
    if normalized in {"", "na", "n/a", "nan", "unknown", "unspecified", "not reported"}:
        return "unknown"
    if normalized in {"f", "female"}:
        return "female"
    if normalized in {"m", "male"}:
        return "male"
    if normalized in {"intersex"}:
        return "intersex"
    raise AggregateError(f"unsupported sex value: {value!r}")


def _demographic_summary(
    metadata_rows: Sequence[Mapping[str, Any]] | None,
    primary_ids: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate declared demographics for the finalized primary cohort only."""

    if not metadata_rows or not any("sex" in row for row in metadata_rows):
        return {}
    by_subject: dict[str, Mapping[str, Any]] = {}
    for row in metadata_rows:
        subject_id = str(row.get("subject_id", "")).strip()
        if not subject_id or subject_id in by_subject:
            raise AggregateError("metadata rows contain duplicate or empty subject IDs")
        by_subject[subject_id] = row
    if any(subject_id not in by_subject for subject_id in primary_ids):
        raise AggregateError("demographic metadata does not cover the primary cohort")
    values = (_normalize_sex(by_subject[subject_id].get("sex")) for subject_id in primary_ids)
    counts: dict[str, int] = {}
    for category in values:
        counts[category] = counts.get(category, 0) + 1
    order = ("female", "male", "intersex", "unknown")
    cells = [
        {"category": category, **_count_cell(counts[category])}
        for category in order
        if category in counts
    ]
    return {"sex": _secondary_suppress_cells(cells)}


def _validate_source_hashes(source_hashes: Mapping[str, Any]) -> dict[str, str]:
    required = {
        "study_lock_sha256",
        "protocol_sha256",
        "training_protocol_sha256",
        "mipdb_manifest_sha256",
        "cohort_qc_sha256",
        "preprocessing_sha256",
    }
    if set(source_hashes) != required:
        raise AggregateError("source hash fields are not exact")
    return {
        field: _validate_sha256(value, field=field)
        for field, value in source_hashes.items()
    }


def build_candidate(
    *,
    metadata_row_count: int,
    draft_manifest: Mapping[str, Any],
    final_manifest: Mapping[str, Any],
    cohort_qc: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
    metadata_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one digest-bound aggregate candidate without participant rows."""

    identity = dict(dataset_identity)
    validate_source_identity(
        identity,
        expected_inventory_sha256=str(identity.get("mipdb_inventory_sha256", "")),
    )
    hashes = _validate_source_hashes(source_hashes)
    if hashes["mipdb_manifest_sha256"] != identity["mipdb_inventory_sha256"]:
        raise AggregateError("source hash manifest differs from dataset identity")
    cohort_states = build_cohort_state_summary(
        metadata_row_count=metadata_row_count,
        draft_manifest=draft_manifest,
        final_manifest=final_manifest,
        cohort_qc=cohort_qc,
    )
    recording_summary = build_recording_window_summary(
        draft_manifest=draft_manifest,
        final_manifest=final_manifest,
        cohort_qc=cohort_qc,
    )
    final_subjects = _manifest_subjects(final_manifest)
    primary_ids = _manifest_cohort(final_manifest, "primary")
    age_bins = _counted_bins(
        _age_bin(final_subjects[subject_id].get("age")) for subject_id in primary_ids
    )
    if metadata_rows is not None and len(metadata_rows) != metadata_row_count:
        raise AggregateError("metadata row count does not match participants.tsv")
    demographics = _demographic_summary(metadata_rows, primary_ids)
    suppression_policy = {
        "minimum_cell_count": _MINIMUM_CELL_COUNT,
        "cross_tabs_enabled": False,
        "numeric_unit": "integer_count",
    }
    scope_body = {
        "schema_version": 1,
        "dataset_identity": identity,
        "source_hashes": hashes,
        "cohort_scope": ["primary_post_qc"],
        "demographic_fields": [],
        "suppression_policy": suppression_policy,
    }
    candidate: dict[str, Any] = {
        "schema_version": 1,
        "release_class": "controlled_review",
        "dataset_identity": identity,
        "source_hashes": hashes,
        "cohort_states": cohort_states,
        "age_bins": age_bins,
        "demographics": demographics,
        "recording_window_summary": recording_summary,
        "suppression_policy": suppression_policy,
        "release_scope_sha256": canonical_sha256(scope_body),
        "candidate_digest_sha256": "0" * 64,
    }
    candidate["candidate_digest_sha256"] = candidate_digest(candidate)
    return candidate
