"""Immutable lock and inventory contracts for the capacity--data extension."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


class CapacityDataRegimeLockError(RuntimeError):
    """Raised when extension evidence is missing, altered, or mis-bound."""


LOCK_CORE_FIELDS = {
    "extension_id",
    "schema_version",
    "parent_primary_study_lock_sha256",
    "parent_primary_prediction_inventory_sha256",
    "extension_protocol_sha256",
    "representation_protocol_sha256",
    "training_protocol_sha256",
    "training_source_sha256",
    "environment_sha256",
    "hardware_sha256",
    "hbn_manifest_sha256",
    "hbn_training_manifest_sha256",
    "representation_cache_manifest_sha256",
    "validation_subject_list_sha256",
    "cohort_hashes",
    "expected_run_count",
    "expected_prediction_count",
    "output_root_identity",
    "preflight",
}
RUN_FIELDS = {
    "training_size",
    "head",
    "seed",
    "run_manifest_sha256",
    "selected_checkpoint_sha256",
    "status",
    "cached_window_count",
    "optimizer_steps",
    "observed_early_stopping_steps",
    "validation_history",
    "selected_validation_subject_metrics",
    "selected_epoch",
    "head_complexity",
    "resource",
}
PREDICTION_FIELDS = {
    "training_size",
    "head",
    "seed",
    "subject_id",
    "true_age",
    "prediction",
    "split",
}
CAPACITY_HEADS = {
    "mean_linear",
    "mean_rich_stats_residual",
    "mean_mlp_residual_matched(hidden_dim=4)",
}
CAPACITY_SIZES = {200, 400, 800}
CAPACITY_SEEDS = set(range(33, 43))


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise CapacityDataRegimeLockError(
            f"{label} fields do not match the strict schema"
        )


def _validate_sha(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise CapacityDataRegimeLockError(f"{label} must be a SHA-256 digest")
    return str(value)


def _validate_core_payload(core: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(core, LOCK_CORE_FIELDS, "lock core fields")
    if core["extension_id"] != "reve_age_capacity_data_regime_v1":
        raise CapacityDataRegimeLockError("lock core extension_id is invalid")
    if core["schema_version"] != 1:
        raise CapacityDataRegimeLockError("lock core schema_version is invalid")
    for field in (
        "parent_primary_study_lock_sha256",
        "parent_primary_prediction_inventory_sha256",
        "extension_protocol_sha256",
        "representation_protocol_sha256",
        "training_protocol_sha256",
        "training_source_sha256",
        "environment_sha256",
        "hardware_sha256",
        "hbn_manifest_sha256",
        "hbn_training_manifest_sha256",
        "representation_cache_manifest_sha256",
        "validation_subject_list_sha256",
    ):
        _validate_sha(core[field], f"lock core {field}")
    cohort_hashes = core["cohort_hashes"]
    if not isinstance(cohort_hashes, dict) or set(cohort_hashes) != {
        "train_200",
        "train_400",
        "train_800",
    }:
        raise CapacityDataRegimeLockError("lock core cohort hashes are invalid")
    for field, value in cohort_hashes.items():
        _validate_sha(value, f"lock core cohort_hashes.{field}")
    if core["expected_run_count"] != 90:
        raise CapacityDataRegimeLockError("lock core expected run count must be 90")
    if core["expected_prediction_count"] != 6_750:
        raise CapacityDataRegimeLockError(
            "lock core expected prediction count must be 6,750"
        )
    if not isinstance(core["output_root_identity"], str) or not core[
        "output_root_identity"
    ]:
        raise CapacityDataRegimeLockError("lock core output_root_identity is invalid")
    preflight = core["preflight"]
    preflight_fields = {
        "representation_cache_bytes",
        "estimated_extension_output_bytes",
        "free_space_bytes",
        "required_free_space_bytes",
        "peak_ram_bytes",
        "upper_bound_optimizer_steps",
        "observed_pilot_seconds",
        "cached_window_count",
    }
    if not isinstance(preflight, dict) or set(preflight) != preflight_fields:
        raise CapacityDataRegimeLockError("lock core preflight fields are invalid")
    for field in preflight_fields - {"observed_pilot_seconds"}:
        value = preflight[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CapacityDataRegimeLockError(
                f"lock core preflight.{field} is invalid"
            )
    if (
        isinstance(preflight["observed_pilot_seconds"], bool)
        or not isinstance(preflight["observed_pilot_seconds"], (int, float))
        or preflight["observed_pilot_seconds"] < 0
    ):
        raise CapacityDataRegimeLockError(
            "lock core preflight.observed_pilot_seconds is invalid"
        )
    return dict(core)


def build_lock_core(core: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a digest-free lock core and attach its canonical digest."""

    validated = _validate_core_payload(core)
    return {
        "lock_core": validated,
        "lock_core_sha256": _canonical_sha256(validated),
    }


def _validate_core_bundle(bundle: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    _exact_fields(bundle, {"lock_core", "lock_core_sha256"}, "lock core bundle")
    core = _validate_core_payload(bundle["lock_core"])
    digest = _validate_sha(bundle["lock_core_sha256"], "lock_core_sha256")
    if digest != _canonical_sha256(core):
        raise CapacityDataRegimeLockError("lock core digest does not match its content")
    return core, digest


def _validate_run_record(run: Mapping[str, Any], index: int) -> tuple[int, str, int]:
    _exact_fields(run, RUN_FIELDS, f"run[{index}]")
    training_size = run["training_size"]
    seed = run["seed"]
    head = run["head"]
    if training_size not in CAPACITY_SIZES:
        raise CapacityDataRegimeLockError(f"run[{index}] training_size is invalid")
    if seed not in CAPACITY_SEEDS:
        raise CapacityDataRegimeLockError(f"run[{index}] seed is invalid")
    if head not in CAPACITY_HEADS:
        raise CapacityDataRegimeLockError(f"run[{index}] head is invalid")
    if run["status"] != "complete":
        raise CapacityDataRegimeLockError(f"run[{index}] is not complete")
    _validate_sha(run["run_manifest_sha256"], f"run[{index}].run_manifest_sha256")
    _validate_sha(
        run["selected_checkpoint_sha256"],
        f"run[{index}].selected_checkpoint_sha256",
    )
    for field in (
        "cached_window_count",
        "optimizer_steps",
        "observed_early_stopping_steps",
        "selected_epoch",
    ):
        if isinstance(run[field], bool) or not isinstance(run[field], int) or run[field] <= 0:
            raise CapacityDataRegimeLockError(f"run[{index}].{field} is invalid")
    if not isinstance(run["validation_history"], list) or not run["validation_history"]:
        raise CapacityDataRegimeLockError(f"run[{index}].validation_history is invalid")
    if (
        not isinstance(run["selected_validation_subject_metrics"], list)
        or not run["selected_validation_subject_metrics"]
    ):
        raise CapacityDataRegimeLockError(
            f"run[{index}].selected_validation_subject_metrics is invalid"
        )
    if not isinstance(run["head_complexity"], dict) or not isinstance(run["resource"], dict):
        raise CapacityDataRegimeLockError(f"run[{index}] retained metadata is invalid")
    return int(training_size), str(head), int(seed)


def build_checkpoint_inventory(
    core_bundle: Mapping[str, Any],
    *,
    runs: Sequence[Mapping[str, Any]],
    expected_prediction_count: int,
) -> dict[str, Any]:
    """Build the complete 90-run inventory without a self-referential digest."""

    core, core_digest = _validate_core_bundle(core_bundle)
    if expected_prediction_count != core["expected_prediction_count"]:
        raise CapacityDataRegimeLockError(
            "checkpoint inventory prediction count differs from lock core"
        )
    if len(runs) != core["expected_run_count"]:
        raise CapacityDataRegimeLockError("checkpoint inventory must contain exactly 90 runs")
    identities: set[tuple[int, str, int]] = set()
    normalized_runs: list[dict[str, Any]] = []
    for index, raw_run in enumerate(runs):
        run = dict(raw_run)
        identity = _validate_run_record(run, index)
        if identity in identities:
            raise CapacityDataRegimeLockError("checkpoint inventory contains duplicate runs")
        identities.add(identity)
        normalized_runs.append(run)
    expected_identities = {
        (size, head, seed)
        for size in CAPACITY_SIZES
        for head in CAPACITY_HEADS
        for seed in CAPACITY_SEEDS
    }
    if identities != expected_identities:
        raise CapacityDataRegimeLockError(
            "checkpoint inventory does not contain the exact 90-run matrix"
        )
    normalized_runs.sort(
        key=lambda run: (run["training_size"], run["head"], run["seed"])
    )
    body = {
        "schema_version": 1,
        "status": "complete",
        "lock_core_sha256": core_digest,
        "run_count": len(normalized_runs),
        "expected_prediction_count": expected_prediction_count,
        "runs": normalized_runs,
    }
    return {
        **body,
        "checkpoint_inventory_body_sha256": _canonical_sha256(body),
    }


def _validate_checkpoint_inventory(
    inventory: Mapping[str, Any], *, expected_core_digest: str | None = None
) -> tuple[dict[str, Any], str]:
    fields = {
        "schema_version",
        "status",
        "lock_core_sha256",
        "run_count",
        "expected_prediction_count",
        "runs",
        "checkpoint_inventory_body_sha256",
    }
    _exact_fields(inventory, fields, "checkpoint inventory")
    body = {key: value for key, value in inventory.items() if key != "checkpoint_inventory_body_sha256"}
    digest = _validate_sha(
        inventory["checkpoint_inventory_body_sha256"],
        "checkpoint_inventory_body_sha256",
    )
    if digest != _canonical_sha256(body):
        raise CapacityDataRegimeLockError(
            "checkpoint inventory digest does not match its lock core-bound content"
        )
    if expected_core_digest is not None and inventory["lock_core_sha256"] != expected_core_digest:
        raise CapacityDataRegimeLockError("checkpoint inventory lock core digest differs")
    if (
        inventory["schema_version"] != 1
        or inventory["status"] != "complete"
        or inventory["run_count"] != 90
        or inventory["expected_prediction_count"] != 6_750
        or not isinstance(inventory["runs"], list)
    ):
        raise CapacityDataRegimeLockError("checkpoint inventory metadata is invalid")
    if len(inventory["runs"]) != 90:
        raise CapacityDataRegimeLockError("checkpoint inventory must contain exactly 90 runs")
    identities = {_validate_run_record(run, index) for index, run in enumerate(inventory["runs"])}
    expected_identities = {
        (size, head, seed)
        for size in CAPACITY_SIZES
        for head in CAPACITY_HEADS
        for seed in CAPACITY_SEEDS
    }
    if identities != expected_identities:
        raise CapacityDataRegimeLockError("checkpoint inventory matrix is incomplete")
    return dict(body), digest


def load_checkpoint_inventory(
    inventory: Mapping[str, Any], *, expected_core_digest: str | None = None
) -> dict[str, Any]:
    _validate_checkpoint_inventory(inventory, expected_core_digest=expected_core_digest)
    return dict(inventory)


def _lock_hash(envelope: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {key: value for key, value in envelope.items() if key != "lock_sha256"}
    )


def build_checkpoint_sealed_lock(
    core_bundle: Mapping[str, Any], checkpoint_inventory: Mapping[str, Any]
) -> dict[str, Any]:
    core, core_digest = _validate_core_bundle(core_bundle)
    _, checkpoint_digest = _validate_checkpoint_inventory(
        checkpoint_inventory, expected_core_digest=core_digest
    )
    envelope = {
        "schema_version": 1,
        "status": "checkpoint_sealed",
        "lock_core": core,
        "lock_core_sha256": core_digest,
        "checkpoint_inventory_body_sha256": checkpoint_digest,
        "prediction_inventory_status": "expected",
        "lifecycle_state": "checkpoint_sealed",
    }
    return {**envelope, "lock_sha256": _lock_hash(envelope)}


def load_checkpoint_sealed_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable pre-external lock envelope."""

    fields = {
        "schema_version",
        "status",
        "lock_core",
        "lock_core_sha256",
        "checkpoint_inventory_body_sha256",
        "prediction_inventory_status",
        "lifecycle_state",
        "lock_sha256",
    }
    _exact_fields(lock, fields, "checkpoint-sealed lock")
    if lock["schema_version"] != 1 or lock["status"] != "checkpoint_sealed":
        raise CapacityDataRegimeLockError("checkpoint-sealed lock metadata is invalid")
    _validate_core_bundle(
        {"lock_core": lock["lock_core"], "lock_core_sha256": lock["lock_core_sha256"]}
    )
    _validate_sha(
        lock["checkpoint_inventory_body_sha256"],
        "checkpoint-sealed lock checkpoint inventory digest",
    )
    if lock["prediction_inventory_status"] != "expected":
        raise CapacityDataRegimeLockError(
            "checkpoint-sealed lock prediction status is invalid"
        )
    if lock["lifecycle_state"] != "checkpoint_sealed":
        raise CapacityDataRegimeLockError(
            "checkpoint-sealed lock lifecycle state is invalid"
        )
    if lock["lock_sha256"] != _lock_hash(lock):
        raise CapacityDataRegimeLockError(
            "checkpoint-sealed lock digest does not match its content"
        )
    return dict(lock)


def _validate_prediction_row(row: Mapping[str, Any], index: int) -> tuple[int, str, int, str]:
    _exact_fields(row, PREDICTION_FIELDS, f"prediction[{index}]")
    training_size = row["training_size"]
    seed = row["seed"]
    head = row["head"]
    subject_id = row["subject_id"]
    if training_size not in CAPACITY_SIZES or seed not in CAPACITY_SEEDS or head not in CAPACITY_HEADS:
        raise CapacityDataRegimeLockError(f"prediction[{index}] identity is invalid")
    if not isinstance(subject_id, str) or not subject_id:
        raise CapacityDataRegimeLockError(f"prediction[{index}] subject_id is invalid")
    for field in ("true_age", "prediction"):
        value = row[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise CapacityDataRegimeLockError(f"prediction[{index}] {field} is invalid")
    if row["split"] != "mipdb_primary":
        raise CapacityDataRegimeLockError(f"prediction[{index}] split is invalid")
    return int(training_size), str(head), int(seed), subject_id


def build_prediction_inventory(
    core_bundle: Mapping[str, Any],
    checkpoint_inventory: Mapping[str, Any],
    *,
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the finalized 6,750-row prediction inventory."""

    core, core_digest = _validate_core_bundle(core_bundle)
    _, checkpoint_digest = _validate_checkpoint_inventory(
        checkpoint_inventory, expected_core_digest=core_digest
    )
    if len(predictions) != core["expected_prediction_count"]:
        raise CapacityDataRegimeLockError(
            "prediction inventory must contain exactly 6,750 rows"
        )
    normalized = [dict(row) for row in predictions]
    identities: set[tuple[int, str, int, str]] = set()
    for index, row in enumerate(normalized):
        identity = _validate_prediction_row(row, index)
        if identity in identities:
            raise CapacityDataRegimeLockError(
                "prediction inventory contains duplicate subject rows"
            )
        identities.add(identity)
    normalized.sort(
        key=lambda row: (
            row["training_size"],
            row["head"],
            row["seed"],
            row["subject_id"].encode("utf-8"),
        )
    )
    body = {
        "schema_version": 1,
        "status": "complete",
        "lock_core_sha256": core_digest,
        "checkpoint_inventory_body_sha256": checkpoint_digest,
        "prediction_count": len(normalized),
        "predictions": normalized,
    }
    return {
        **body,
        "prediction_inventory_body_sha256": _canonical_sha256(body),
    }


def _validate_prediction_inventory(
    inventory: Mapping[str, Any], *, expected_core_digest: str | None = None,
    expected_checkpoint_digest: str | None = None,
) -> tuple[dict[str, Any], str]:
    fields = {
        "schema_version",
        "status",
        "lock_core_sha256",
        "checkpoint_inventory_body_sha256",
        "prediction_count",
        "predictions",
        "prediction_inventory_body_sha256",
    }
    _exact_fields(inventory, fields, "prediction inventory")
    body = {key: value for key, value in inventory.items() if key != "prediction_inventory_body_sha256"}
    digest = _validate_sha(
        inventory["prediction_inventory_body_sha256"],
        "prediction_inventory_body_sha256",
    )
    if digest != _canonical_sha256(body):
        raise CapacityDataRegimeLockError(
            "prediction inventory digest does not match its content"
        )
    if expected_core_digest is not None and inventory["lock_core_sha256"] != expected_core_digest:
        raise CapacityDataRegimeLockError("prediction inventory lock core digest differs")
    if expected_checkpoint_digest is not None and inventory["checkpoint_inventory_body_sha256"] != expected_checkpoint_digest:
        raise CapacityDataRegimeLockError("prediction inventory checkpoint digest differs")
    if (
        inventory["schema_version"] != 1
        or inventory["status"] != "complete"
        or inventory["prediction_count"] != 6_750
        or not isinstance(inventory["predictions"], list)
        or len(inventory["predictions"]) != 6_750
    ):
        raise CapacityDataRegimeLockError("prediction inventory metadata is invalid")
    identities = {
        _validate_prediction_row(row, index)
        for index, row in enumerate(inventory["predictions"])
    }
    if len(identities) != 6_750:
        raise CapacityDataRegimeLockError("prediction inventory contains duplicate rows")
    return dict(body), digest


def load_prediction_inventory(
    inventory: Mapping[str, Any], *, expected_core_digest: str | None = None,
    expected_checkpoint_digest: str | None = None,
) -> dict[str, Any]:
    _validate_prediction_inventory(
        inventory,
        expected_core_digest=expected_core_digest,
        expected_checkpoint_digest=expected_checkpoint_digest,
    )
    return dict(inventory)


def build_final_lock(
    checkpoint_sealed_lock: Mapping[str, Any],
    prediction_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_fields(
        checkpoint_sealed_lock,
        {
            "schema_version",
            "status",
            "lock_core",
            "lock_core_sha256",
            "checkpoint_inventory_body_sha256",
            "prediction_inventory_status",
            "lifecycle_state",
            "lock_sha256",
        },
        "checkpoint-sealed lock",
    )
    checkpoint_without_digest = {
        key: value for key, value in checkpoint_sealed_lock.items() if key != "lock_sha256"
    }
    if checkpoint_sealed_lock["lock_sha256"] != _canonical_sha256(checkpoint_without_digest):
        raise CapacityDataRegimeLockError("checkpoint-sealed lock digest is invalid")
    core = {"lock_core": checkpoint_sealed_lock["lock_core"], "lock_core_sha256": checkpoint_sealed_lock["lock_core_sha256"]}
    core_payload, core_digest = _validate_core_bundle(core)
    _, prediction_digest = _validate_prediction_inventory(
        prediction_inventory,
        expected_core_digest=core_digest,
        expected_checkpoint_digest=checkpoint_sealed_lock["checkpoint_inventory_body_sha256"],
    )
    envelope = {
        "schema_version": 1,
        "status": "final",
        "lock_core": core_payload,
        "lock_core_sha256": core_digest,
        "checkpoint_inventory_body_sha256": checkpoint_sealed_lock[
            "checkpoint_inventory_body_sha256"
        ],
        "prediction_inventory_body_sha256": prediction_digest,
        "checkpoint_sealed_lock_sha256": checkpoint_sealed_lock["lock_sha256"],
        "lifecycle_state": "completed",
    }
    return {**envelope, "lock_sha256": _lock_hash(envelope)}


def load_final_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema_version",
        "status",
        "lock_core",
        "lock_core_sha256",
        "checkpoint_inventory_body_sha256",
        "prediction_inventory_body_sha256",
        "checkpoint_sealed_lock_sha256",
        "lifecycle_state",
        "lock_sha256",
    }
    _exact_fields(lock, fields, "final lock")
    if lock["schema_version"] != 1 or lock["status"] != "final":
        raise CapacityDataRegimeLockError("final lock metadata is invalid")
    _validate_core_bundle(
        {"lock_core": lock["lock_core"], "lock_core_sha256": lock["lock_core_sha256"]}
    )
    for field in (
        "checkpoint_inventory_body_sha256",
        "prediction_inventory_body_sha256",
        "checkpoint_sealed_lock_sha256",
    ):
        _validate_sha(lock[field], f"final lock {field}")
    if lock["lifecycle_state"] != "completed":
        raise CapacityDataRegimeLockError("final lock lifecycle state is invalid")
    if lock["lock_sha256"] != _lock_hash(lock):
        raise CapacityDataRegimeLockError("final lock digest does not match its content")
    return dict(lock)


_LIFECYCLE_STATES = (
    "draft",
    "checkpoint_sealed",
    "external_started",
    "completed",
    "failed",
)


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise CapacityDataRegimeLockError(
                f"lifecycle sidecar is immutable: {path}"
            ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _lifecycle_files(directory: Path) -> list[Path]:
    return sorted(Path(directory).glob("lifecycle-*.json"))


def _current_lifecycle(directory: Path) -> tuple[str, str] | None:
    files = _lifecycle_files(directory)
    if not files:
        return None
    parsed: list[tuple[int, str, str]] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CapacityDataRegimeLockError("lifecycle sidecar is unreadable") from error
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "state",
            "lock_sha256",
        }:
            raise CapacityDataRegimeLockError("lifecycle sidecar schema is invalid")
        if payload["state"] not in _LIFECYCLE_STATES:
            raise CapacityDataRegimeLockError("lifecycle sidecar state is invalid")
        lock_sha = _validate_sha(payload["lock_sha256"], "lifecycle lock_sha256")
        parsed.append((_LIFECYCLE_STATES.index(payload["state"]), str(payload["state"]), lock_sha))
    _, state, lock_sha = max(parsed, key=lambda item: item[0])
    return state, lock_sha


def create_lifecycle_sidecar(
    directory: Path, *, state: str, lock_sha256: str
) -> Path:
    if state != "draft":
        raise CapacityDataRegimeLockError("lifecycle must start in draft")
    _validate_sha(lock_sha256, "lifecycle lock_sha256")
    directory = Path(directory)
    if _current_lifecycle(directory) is not None:
        raise CapacityDataRegimeLockError("lifecycle sidecar is immutable")
    path = directory / "lifecycle-draft.json"
    _write_create_only(
        path,
        {"schema_version": 1, "state": state, "lock_sha256": lock_sha256},
    )
    return path


def transition_lifecycle(
    directory: Path, *, state: str, lock_sha256: str
) -> Path:
    if state not in _LIFECYCLE_STATES:
        raise CapacityDataRegimeLockError("lifecycle transition state is invalid")
    _validate_sha(lock_sha256, "lifecycle lock_sha256")
    directory = Path(directory)
    current = _current_lifecycle(directory)
    if current is None:
        raise CapacityDataRegimeLockError("lifecycle has not started")
    current_state, current_lock_sha = current
    if current_lock_sha != lock_sha256:
        raise CapacityDataRegimeLockError("lifecycle lock binding changed")
    if current_state in {"completed", "failed"}:
        raise CapacityDataRegimeLockError("lifecycle is immutable after completion")
    allowed = {
        "draft": {"checkpoint_sealed", "failed"},
        "checkpoint_sealed": {"external_started", "failed"},
        "external_started": {"completed", "failed"},
    }
    if state not in allowed[current_state]:
        raise CapacityDataRegimeLockError(
            f"invalid lifecycle transition {current_state} -> {state}"
        )
    path = directory / f"lifecycle-{state}.json"
    _write_create_only(
        path,
        {"schema_version": 1, "state": state, "lock_sha256": lock_sha256},
    )
    return path
