"""Immutable study sealing and explicit lifecycle transitions."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


LOCK_FIELDS = (
    "study_id",
    "protocol_sha256",
    "source_tree_sha256",
    "git_revision",
    "git_dirty",
    "encoder_checkpoint",
    "encoder_checkpoint_sha256",
    "hbn_manifest_sha256",
    "mipdb_manifest_sha256",
    "subject_list_sha256",
    "heads",
    "seeds",
    "preprocessing_sha256",
    "statistics_sha256",
    "output_root",
)
HASH_FIELDS = (
    "protocol_sha256",
    "source_tree_sha256",
    "encoder_checkpoint_sha256",
    "hbn_manifest_sha256",
    "mipdb_manifest_sha256",
    "preprocessing_sha256",
    "statistics_sha256",
)
SUBJECT_LISTS = (
    "hbn_train",
    "hbn_validation",
    "mipdb_pilot",
    "mipdb_primary",
    "mipdb_extrapolation",
)
HEADS = (
    "mean_linear",
    "mean_layer_linear",
    "mean_rich_stats_residual",
    "multi_query_rich_stats",
)


class StudyLockError(RuntimeError):
    """Raised when the sealed study is missing, altered, or misused."""

    def __init__(
        self, message: str, *, mismatched_fields: Sequence[str] = ()
    ) -> None:
        self.mismatched_fields = tuple(mismatched_fields)
        super().__init__(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(payload) - set(LOCK_FIELDS))
    missing = sorted(set(LOCK_FIELDS) - set(payload))
    if unknown or missing:
        raise StudyLockError(
            f"invalid study lock fields: unknown={unknown} missing={missing}"
        )
    for field in HASH_FIELDS:
        if not _is_sha256(payload[field]):
            raise StudyLockError(f"{field} must be a lowercase SHA-256 digest")
    if not isinstance(payload["study_id"], str) or not payload["study_id"]:
        raise StudyLockError("study_id must be a non-empty string")
    if not isinstance(payload["git_revision"], str) or not payload["git_revision"]:
        raise StudyLockError("git_revision must be a non-empty string")
    if not isinstance(payload["git_dirty"], bool):
        raise StudyLockError("git_dirty must be boolean")
    if payload["encoder_checkpoint"] != "brain-bzh/reve-base":
        raise StudyLockError("unexpected encoder checkpoint")
    subject_hashes = payload["subject_list_sha256"]
    if not isinstance(subject_hashes, Mapping) or set(subject_hashes) != set(SUBJECT_LISTS):
        raise StudyLockError("subject_list_sha256 must contain every declared cohort")
    if any(not _is_sha256(value) for value in subject_hashes.values()):
        raise StudyLockError("every subject-list hash must be a SHA-256 digest")
    if tuple(payload["heads"]) != HEADS:
        raise StudyLockError("study lock must contain exactly the four approved heads")
    if tuple(payload["seeds"]) != tuple(range(33, 43)):
        raise StudyLockError("study lock must contain exactly seeds 33 through 42")
    output_root = Path(str(payload["output_root"]))
    if not output_root.is_absolute():
        raise StudyLockError("output_root must be absolute")
    normalized = dict(payload)
    normalized["subject_list_sha256"] = dict(subject_hashes)
    normalized["heads"] = list(payload["heads"])
    normalized["seeds"] = list(payload["seeds"])
    normalized["output_root"] = str(output_root.resolve())
    return normalized


def _atomic_replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _publish_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise StudyLockError(f"immutable artifact already exists: {path}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _state_path(lock_path: Path) -> Path:
    return Path(lock_path).parent / "study_state.json"


def seal_study(lock_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Publish one immutable lock and initialize its mutable state sidecar."""

    lock_path = Path(lock_path)
    normalized = _validate_payload(payload)
    body = {
        "schema_version": 1,
        "sealed_at_utc": _utc_now(),
        **normalized,
    }
    lock = {**body, "lock_sha256": canonical_sha256(body)}
    _publish_json_create_only(lock_path, lock)
    try:
        _publish_json_create_only(
            _state_path(lock_path),
            {
                "schema_version": 1,
                "study_id": normalized["study_id"],
                "lock_sha256": lock["lock_sha256"],
                "state": "sealed",
                "updated_at_utc": _utc_now(),
            },
        )
    except Exception:
        lock_path.unlink(missing_ok=True)
        raise
    return lock


def load_study_lock(lock_path: Path) -> dict[str, Any]:
    lock_path = Path(lock_path)
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StudyLockError(f"could not read study lock: {lock_path}") from error
    if not isinstance(lock, dict):
        raise StudyLockError("study lock must contain an object")
    claimed = lock.get("lock_sha256")
    body = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if not _is_sha256(claimed) or claimed != canonical_sha256(body):
        raise StudyLockError("study lock digest does not match its content")
    payload = {field: lock.get(field) for field in LOCK_FIELDS}
    _validate_payload(payload)
    return lock


def verify_exact_study(
    lock_path: Path, expected_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify every sealed input before a start or exact resume."""

    expected = _validate_payload(expected_payload)
    lock = load_study_lock(lock_path)
    mismatches = tuple(
        sorted(field for field in LOCK_FIELDS if lock.get(field) != expected.get(field))
    )
    if mismatches:
        raise StudyLockError(
            "sealed study provenance drift: " + ", ".join(mismatches),
            mismatched_fields=mismatches,
        )
    return lock


def _load_state(lock_path: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    path = _state_path(lock_path)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StudyLockError(f"could not read study state: {path}") from error
    if not isinstance(state, dict) or state.get("lock_sha256") != lock["lock_sha256"]:
        raise StudyLockError("study state does not belong to the immutable lock")
    return state


def transition_study(lock_path: Path, target: str) -> dict[str, Any]:
    """Advance the lifecycle without mutating the lock."""

    lock_path = Path(lock_path)
    lock = load_study_lock(lock_path)
    state = _load_state(lock_path, lock)
    current = state.get("state")
    if current in {"completed", "failed"}:
        raise StudyLockError(f"study state {current!r} is terminal")
    allowed = {("sealed", "started"), ("started", "completed")}
    if (current, target) not in allowed:
        raise StudyLockError(f"invalid study transition {current} -> {target}")
    updated = {
        **state,
        "state": target,
        "updated_at_utc": _utc_now(),
    }
    _atomic_replace_json(_state_path(lock_path), updated)
    return updated


def fail_study(lock_path: Path, *, error: str) -> dict[str, Any]:
    """Record a terminal failure separately from the immutable lock."""

    if not isinstance(error, str) or not error.strip():
        raise StudyLockError("failure error must be a non-empty string")
    lock_path = Path(lock_path)
    lock = load_study_lock(lock_path)
    state = _load_state(lock_path, lock)
    if state.get("state") in {"completed", "failed"}:
        raise StudyLockError(f"study state {state.get('state')!r} is terminal")
    if state.get("state") != "started":
        raise StudyLockError("only a started study may transition to failed")
    failed_at = _utc_now()
    failure = {
        "schema_version": 1,
        "study_id": lock["study_id"],
        "lock_sha256": lock["lock_sha256"],
        "error": error,
        "failed_at_utc": failed_at,
    }
    _publish_json_create_only(lock_path.parent / "study_failure.json", failure)
    updated = {**state, "state": "failed", "updated_at_utc": failed_at}
    _atomic_replace_json(_state_path(lock_path), updated)
    return updated
