"""Target-free adapter and provenance utilities for OpenNeuro ds006780."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from ..research.strict_json import (
    StrictJsonError,
    canonical_sha256,
    load_json_strict,
    reject_target_fields,
    validate_schema,
    write_create_only_json,
)
from .preprocessing import PreprocessingError, preprocess_rest_blocks


DATASET_ID = "ds006780"
PRIMARY_TASK = "Restingstate"
PRIMARY_RUN = "run-01"
EXPECTED_SOURCE_SAMPLING_FREQUENCY_HZ = 512.0
PRIMARY_RECORDING_RE = re.compile(
    r"^sub-(?P<subject>[A-Za-z0-9]+)/eeg/"
    r"sub-(?P=subject)_task-Restingstate_run-01_eeg\.bdf$"
)
SUBJECT_RE = re.compile(r"^sub-[A-Za-z0-9]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class Ds006780Error(ValueError):
    """Raised when ds006780 violates the target-free adapter contract."""


@dataclass(frozen=True)
class RecordingCandidate:
    subject_id: str
    run_id: str
    recording_path: str
    channels_path: str
    events_path: str
    sidecar_path: str


@dataclass(frozen=True)
class RawMetadata:
    sampling_frequency_hz: float
    duration_seconds: float
    n_times: int


@dataclass(frozen=True)
class Ds006780PreprocessingContract:
    sample_rate_hz: float
    bandpass_hz: tuple[float, float]
    notch_hz: tuple[float, ...] | None
    scaler: str
    clamp: float
    window_seconds: float
    stride_seconds: float
    max_seconds_per_subject: float
    cross_block_windows: bool
    spatial_interpolation: bool
    subject_aggregation: str


def load_preprocessing_contract(config: Mapping[str, Any]) -> Ds006780PreprocessingContract:
    preprocessing = config.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise Ds006780Error("external config is missing preprocessing")
    bandpass = preprocessing.get("bandpass_hz")
    if not isinstance(bandpass, list) or len(bandpass) != 2:
        raise Ds006780Error("preprocessing.bandpass_hz must contain two values")
    notch = preprocessing.get("notch_hz")
    return Ds006780PreprocessingContract(
        sample_rate_hz=_finite_float(preprocessing.get("sample_rate_hz"), "sample_rate_hz", minimum=0.0),
        bandpass_hz=(float(bandpass[0]), float(bandpass[1])),
        notch_hz=None if notch is None else tuple(float(value) for value in notch),
        scaler=str(preprocessing.get("scaler")),
        clamp=_finite_float(preprocessing.get("clamp"), "clamp", minimum=0.0),
        window_seconds=_finite_float(preprocessing.get("window_seconds"), "window_seconds", minimum=0.0),
        stride_seconds=_finite_float(preprocessing.get("stride_seconds"), "stride_seconds", minimum=0.0),
        max_seconds_per_subject=_finite_float(
            preprocessing.get("max_seconds_per_subject"),
            "max_seconds_per_subject",
            minimum=0.0,
        ),
        cross_block_windows=bool(preprocessing.get("cross_block_windows")),
        spatial_interpolation=bool(preprocessing.get("spatial_interpolation")),
        subject_aggregation=str(preprocessing.get("subject_aggregation")),
    )


def validate_external_config(
    config: Mapping[str, Any], *, require_approved_precision: bool = False
) -> None:
    precision_gate = config.get("precision_gate")
    if not isinstance(precision_gate, Mapping):
        raise Ds006780Error("external config is missing precision_gate")
    status = precision_gate.get("status")
    if status not in {"pending_threshold_selection", "approved"}:
        raise Ds006780Error("precision_gate.status is invalid")
    threshold = precision_gate.get("max_primary_confidence_interval_width")
    if status == "approved":
        if threshold is None:
            raise Ds006780Error("approved precision gate requires numeric threshold")
        _finite_float(threshold, "max_primary_confidence_interval_width", minimum=0.0)
    if require_approved_precision and status != "approved":
        raise Ds006780Error("precision gate is not approved")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise Ds006780Error(f"path escapes BIDS root: {path}") from error
    normalized = relative.as_posix()
    if not normalized or normalized.startswith("/") or ".." in Path(normalized).parts:
        raise Ds006780Error(f"unsafe relative path: {normalized}")
    return normalized


def _looks_like_git_annex_pointer(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        sample = path.read_bytes()[:4096]
    except OSError as error:
        raise Ds006780Error(f"could not inspect recording payload: {path}") from error
    try:
        text = sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    normalized = text.casefold()
    return (
        normalized.startswith("version https://git-annex.branchable.com/")
        or "git-annex" in normalized
        or normalized.startswith("key ")
    )


def _read_tsv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise Ds006780Error(f"missing TSV header: {path}")
            rows = [dict(row) for row in reader]
    except (OSError, csv.Error) as error:
        raise Ds006780Error(f"could not parse TSV: {path}") from error
    return rows


def _finite_float(value: object, field: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise Ds006780Error(f"{field} must be numeric") from error
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise Ds006780Error(f"{field} must be finite and >= {minimum}")
    return result


def _file_identity(
    root: Path,
    path: Path,
    *,
    role: str,
    openneuro_version: str,
    source_commit: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise Ds006780Error(f"required file is missing: {path}")
    pointer = _looks_like_git_annex_pointer(path)
    if pointer:
        raise Ds006780Error(f"unresolved_git_annex_pointer: {path}")
    return {
        "role": role,
        "path": _relative_path(root, path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "openneuro_version": openneuro_version,
        "source_commit": source_commit,
        "git_annex_pointer_resolved": bool(role == "recording" and not pointer),
    }


def _environment_identity(root: Path) -> dict[str, Any]:
    package_names = ("numpy", "jsonschema", "rfc8785", "scipy", "mne")
    packages: dict[str, str] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    lockfile = root / "uv.lock"
    if not lockfile.is_file():
        raise Ds006780Error(f"missing project lockfile: {lockfile}")
    return {
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "packages": packages,
        "lockfile_sha256": sha256_file(lockfile),
    }


def _load_mne() -> Any:
    try:
        import mne
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise Ds006780Error(
            "ds006780 signal inspection requires the project 'external' dependencies"
        ) from error
    return mne


def montage_identity(*, mne_module: Any | None = None) -> dict[str, Any]:
    mne = mne_module or _load_mne()
    montage = mne.channels.make_standard_montage("biosemi64")
    positions = montage.get_positions()
    coordinates = {
        name: [float(value) for value in position]
        for name, position in positions["ch_pos"].items()
    }
    coordinate_frame = positions.get("coord_frame")
    if coordinate_frame is None:
        coordinate_frame = "head"
    return {
        "name": "biosemi64",
        "mne_version": str(mne.__version__),
        "coordinates_sha256": canonical_sha256(coordinates),
        "coordinate_frame": str(coordinate_frame),
    }


def _canonical_biosemi64_names(*, mne_module: Any | None = None) -> tuple[str, ...]:
    mne = mne_module or _load_mne()
    return tuple(mne.channels.make_standard_montage("biosemi64").ch_names)


def discover_runs(bids_root: Path) -> tuple[list[RecordingCandidate], list[dict[str, str]]]:
    """Discover the one-run-per-subject primary contract without targets."""

    root = Path(bids_root).resolve()
    if not root.is_dir():
        raise Ds006780Error(f"BIDS root is not a directory: {root}")
    candidates: list[RecordingCandidate] = []
    out_of_scope: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.bdf"), key=lambda item: item.as_posix()):
        relative = _relative_path(root, path)
        match = PRIMARY_RECORDING_RE.fullmatch(relative)
        if match:
            subject_id = f"sub-{match.group('subject')}"
            candidates.append(
                RecordingCandidate(
                    subject_id=subject_id,
                    run_id=PRIMARY_RUN,
                    recording_path=relative,
                    channels_path=relative.removesuffix("_eeg.bdf") + "_channels.tsv",
                    events_path=relative.removesuffix("_eeg.bdf") + "_events.tsv",
                    sidecar_path=relative.removesuffix("_eeg.bdf") + "_eeg.json",
                )
            )
            continue
        parts = Path(relative).parts
        subject_id = next((part for part in parts if SUBJECT_RE.fullmatch(part)), "sub-unknown")
        if any(part.startswith("ses-") for part in parts):
            code = "session_variant"
        elif "task-Restingstate" in relative and "run-01" not in relative:
            code = "non_primary_run"
        else:
            code = "unsupported_path"
        out_of_scope.append(
            {
                "subject_id": subject_id,
                "run_id": "unknown",
                "recording_path": relative,
                "exclusion_code": code,
            }
        )
    candidates.sort(key=lambda item: (item.subject_id, item.run_id, item.recording_path))
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (candidate.subject_id, candidate.run_id)
        if key in seen:
            out_of_scope.append(
                {
                    "subject_id": candidate.subject_id,
                    "run_id": candidate.run_id,
                    "recording_path": candidate.recording_path,
                    "exclusion_code": "duplicate_primary_run",
                }
            )
        seen.add(key)
    unique = {(item.subject_id, item.run_id): item for item in candidates}
    candidates = [unique[key] for key in sorted(unique)]
    out_of_scope.sort(key=lambda item: (item["subject_id"], item["run_id"], item["recording_path"]))
    return candidates, out_of_scope


def parse_channel_inventory(
    channels_path: Path,
    *,
    expected_sampling_frequency_hz: float,
    mne_module: Any | None = None,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    rows = _read_tsv(channels_path)
    required = {"name", "type", "units"}
    if not rows or not required.issubset(rows[0]):
        raise Ds006780Error("channels.tsv is missing required columns")
    inventory: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("name", "")).strip()
        raw_type = str(row.get("type", "")).strip().upper()
        if not name:
            raise Ds006780Error("channel layout mismatch: empty channel name")
        if raw_type == "EEG":
            channel_type = "EEG"
        elif raw_type == "EMG":
            channel_type = "EMG"
        elif raw_type in {"TRIG", "TRIGGER", "STATUS"}:
            channel_type = "TRIG"
        else:
            raise Ds006780Error(f"channel layout mismatch: unsupported channel type {raw_type}")
        sampling = row.get("sampling_frequency", expected_sampling_frequency_hz)
        sampling_frequency = _finite_float(sampling, "channel sampling_frequency", minimum=0.0)
        if sampling_frequency <= 0:
            raise Ds006780Error("channel layout mismatch: non-positive sampling frequency")
        inventory.append(
            {
                "name": name,
                "type": channel_type,
                "units": str(row.get("units", "")).strip() or "n/a",
                "status": str(row.get("status", "n/a")).strip() or "n/a",
                "sampling_frequency": sampling_frequency,
            }
        )
    labels = tuple(item["name"] for item in inventory)
    eeg_labels = tuple(item["name"] for item in inventory if item["type"] == "EEG")
    if len(set(labels)) != len(labels) or len(set(eeg_labels)) != len(eeg_labels):
        raise Ds006780Error("channel layout mismatch: duplicate channel name")
    canonical = _canonical_biosemi64_names(mne_module=mne_module)
    if eeg_labels != canonical:
        raise Ds006780Error("channel layout mismatch: EEG labels are not canonical BioSemi-64")
    if len(eeg_labels) != 64:
        raise Ds006780Error("channel layout mismatch: expected exactly 64 EEG channels")
    return inventory, labels


def _raw_metadata(raw: Any) -> RawMetadata:
    sfreq = _finite_float(raw.info["sfreq"], "raw sampling frequency", minimum=0.0)
    n_times = int(raw.n_times)
    if n_times <= 0:
        raise Ds006780Error("recording has no samples")
    return RawMetadata(sfreq, n_times / sfreq, n_times)


def parse_event_inventory(
    events_path: Path,
    *,
    sampling_frequency_hz: float,
    n_times: int,
) -> list[dict[str, Any]]:
    rows = _read_tsv(events_path)
    required = {"onset", "duration", "trial_type"}
    if not rows or not required.issubset(rows[0]):
        raise Ds006780Error("event boundary invalid: events.tsv is missing required columns")
    events: list[dict[str, Any]] = []
    for row in rows:
        onset = _finite_float(row.get("onset"), "event onset", minimum=0.0)
        duration = _finite_float(row.get("duration", 0.0), "event duration", minimum=0.0)
        trial_type = str(row.get("trial_type", "")).strip()
        if trial_type not in {"Recording_start", "Recording_end"}:
            raise Ds006780Error(f"event boundary invalid: unsupported trial_type {trial_type}")
        expected_sample = int(round(onset * sampling_frequency_hz))
        raw_sample = row.get("sample", "")
        if str(raw_sample).strip() == "":
            sample = expected_sample
        else:
            try:
                sample_float = float(raw_sample)
            except (TypeError, ValueError) as error:
                raise Ds006780Error("event boundary invalid: sample is not numeric") from error
            if not math.isfinite(sample_float) or not sample_float.is_integer():
                raise Ds006780Error("event boundary invalid: sample is not an integer")
            sample = int(sample_float)
        if abs(sample - expected_sample) > 1:
            raise Ds006780Error("event boundary invalid: sample/onset mismatch")
        if sample < 0 or sample > n_times:
            raise Ds006780Error("event boundary invalid: sample out of bounds")
        events.append(
            {
                "onset": onset,
                "duration": duration,
                "trial_type": trial_type,
                "value": str(row.get("value", trial_type)).strip() or trial_type,
                "sample": sample,
            }
        )
    starts = [event for event in events if event["trial_type"] == "Recording_start"]
    ends = [event for event in events if event["trial_type"] == "Recording_end"]
    if len(starts) != 1 or len(ends) > 1:
        raise Ds006780Error("event boundary invalid: ambiguous recording boundary")
    if ends and not starts[0]["sample"] < ends[0]["sample"] <= n_times:
        raise Ds006780Error("event boundary invalid: invalid half-open boundary")
    return events


def recording_bounds(events: Iterable[Mapping[str, Any]], *, n_times: int) -> tuple[int, int]:
    starts = [event for event in events if event["trial_type"] == "Recording_start"]
    ends = [event for event in events if event["trial_type"] == "Recording_end"]
    if len(starts) != 1 or len(ends) > 1:
        raise Ds006780Error("event boundary invalid: start/end count")
    start = int(starts[0]["sample"])
    end = int(ends[0]["sample"]) if ends else n_times
    if not 0 <= start < end <= n_times:
        raise Ds006780Error("event boundary invalid: bounds")
    return start, end


def _sidecar_metadata(path: Path) -> dict[str, Any]:
    value = load_json_strict(path)
    if not isinstance(value, Mapping):
        raise Ds006780Error("EEG sidecar must be a JSON object")
    return dict(value)


def _load_raw(path: Path, raw_loader: Callable[[Path], Any] | None = None) -> Any:
    if raw_loader is not None:
        return raw_loader(path)
    mne = _load_mne()
    try:
        return mne.io.read_raw_bdf(path, preload=False, verbose="ERROR")
    except Exception as error:  # pragma: no cover - real-data failure path
        raise Ds006780Error(f"recording_load_failed: {path}") from error


def _candidate_manifest_record(
    root: Path,
    candidate: RecordingCandidate,
    *,
    openneuro_version: str,
    source_commit: str,
    raw_loader: Callable[[Path], Any] | None = None,
    mne_module: Any | None = None,
) -> dict[str, Any]:
    recording_path = root / candidate.recording_path
    channels_path = root / candidate.channels_path
    events_path = root / candidate.events_path
    sidecar_path = root / candidate.sidecar_path
    files = [recording_path, channels_path, events_path, sidecar_path]
    if not all(path.is_file() for path in files):
        missing = [str(path) for path in files if not path.is_file()]
        raise Ds006780Error(f"missing_required_file: {missing}")
    raw = _load_raw(recording_path, raw_loader)
    metadata = _raw_metadata(raw)
    if not math.isclose(
        metadata.sampling_frequency_hz,
        EXPECTED_SOURCE_SAMPLING_FREQUENCY_HZ,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise Ds006780Error("unsupported_sampling_frequency")
    sidecar = _sidecar_metadata(sidecar_path)
    sidecar_frequency = _finite_float(
        sidecar.get("SamplingFrequency", metadata.sampling_frequency_hz),
        "sidecar SamplingFrequency",
        minimum=0.0,
    )
    if not math.isclose(sidecar_frequency, metadata.sampling_frequency_hz, abs_tol=1e-9, rel_tol=0.0):
        raise Ds006780Error("metadata_mismatch: sampling frequency")
    sidecar_duration = _finite_float(
        sidecar.get("RecordingDuration", metadata.duration_seconds),
        "sidecar RecordingDuration",
        minimum=0.0,
    )
    if abs(sidecar_duration - metadata.duration_seconds) > 1.0 / metadata.sampling_frequency_hz:
        raise Ds006780Error("duration_mismatch")
    channel_inventory, channel_names = parse_channel_inventory(
        channels_path,
        expected_sampling_frequency_hz=metadata.sampling_frequency_hz,
        mne_module=mne_module,
    )
    raw_names = tuple(str(name) for name in raw.ch_names)
    if raw_names != channel_names:
        raise Ds006780Error("metadata_mismatch: raw channel order differs from channels.tsv")
    events = parse_event_inventory(
        events_path,
        sampling_frequency_hz=metadata.sampling_frequency_hz,
        n_times=metadata.n_times,
    )
    acquisition_files = [
        _file_identity(root, recording_path, role="recording", openneuro_version=openneuro_version, source_commit=source_commit),
        _file_identity(root, channels_path, role="channels", openneuro_version=openneuro_version, source_commit=source_commit),
        _file_identity(root, events_path, role="events", openneuro_version=openneuro_version, source_commit=source_commit),
        _file_identity(root, sidecar_path, role="sidecar", openneuro_version=openneuro_version, source_commit=source_commit),
    ]
    return {
        "subject_id": candidate.subject_id,
        "run_id": candidate.run_id,
        "recording_path": candidate.recording_path,
        "channels_path": candidate.channels_path,
        "events_path": candidate.events_path,
        "sidecar_path": candidate.sidecar_path,
        "source_sampling_frequency_hz": metadata.sampling_frequency_hz,
        "source_duration_seconds": metadata.duration_seconds,
        "sidecar_recording_duration_seconds": sidecar_duration,
        "channel_names": list(channel_names),
        "channel_inventory": channel_inventory,
        "event_inventory": events,
        "acquisition_files": acquisition_files,
        "target_free_status": "pending",
        "target_free_exclusion_code": "none",
    }


def _structured_exclusion_code(error: Exception) -> str:
    message = str(error)
    for code in (
        "missing_required_file",
        "unresolved_git_annex_pointer",
        "metadata_mismatch",
        "channel_layout_mismatch",
        "event_boundary_invalid",
        "duration_mismatch",
        "unsupported_sampling_frequency",
        "signal_nonfinite",
        "no_complete_window",
    ):
        if code in message:
            return code
    return "metadata_mismatch"


def build_target_free_manifest(
    bids_root: Path,
    *,
    config_path: Path,
    project_root: Path,
    source_commit: str,
    openneuro_version: str,
    raw_loader: Callable[[Path], Any] | None = None,
    mne_module: Any | None = None,
) -> dict[str, Any]:
    root = Path(bids_root).resolve()
    config = load_json_strict(config_path)
    if not isinstance(config, Mapping):
        raise Ds006780Error("external config must be a JSON object")
    validate_external_config(config)
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("dataset_id") != DATASET_ID:
        raise Ds006780Error("config dataset_id must be ds006780")
    if dataset.get("source_commit") != source_commit:
        raise Ds006780Error("source commit does not match external config")
    if dataset.get("openneuro_version") != openneuro_version:
        raise Ds006780Error("OpenNeuro version does not match external config")
    if dataset.get("source_sampling_frequency_hz") != EXPECTED_SOURCE_SAMPLING_FREQUENCY_HZ:
        raise Ds006780Error("external config must declare 512 Hz")
    candidates, out_of_scope = discover_runs(root)
    if not candidates:
        raise Ds006780Error("no primary ds006780 run-01 recordings found")
    candidate_records: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            candidate_records.append(
                _candidate_manifest_record(
                    root,
                    candidate,
                    openneuro_version=openneuro_version,
                    source_commit=source_commit,
                    raw_loader=raw_loader,
                    mne_module=mne_module,
                )
            )
        except Ds006780Error as error:
            out_of_scope.append(
                {
                    "subject_id": candidate.subject_id,
                    "run_id": candidate.run_id,
                    "recording_path": candidate.recording_path,
                    "exclusion_code": _structured_exclusion_code(error),
                }
            )
    candidate_records.sort(
        key=lambda item: (item["subject_id"], item["run_id"], item["recording_path"])
    )
    metadata_files: list[dict[str, Any]] = []
    for relative, role in (("dataset_description.json", "dataset_description"), ("README", "README")):
        path = root / relative
        if path.is_file():
            metadata_files.append(
                {
                    "role": role,
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "openneuro_version": openneuro_version,
                    "source_commit": source_commit,
                    "git_annex_pointer_resolved": False,
                }
            )
    if not metadata_files:
        raise Ds006780Error("missing dataset metadata files")
    if not candidate_records:
        if out_of_scope:
            first_exclusion = out_of_scope[0]
            raise Ds006780Error(
                "no primary ds006780 run-01 recordings passed metadata QC; "
                f"first exclusion {first_exclusion['exclusion_code']}: "
                f"{first_exclusion['recording_path']}"
            )
        raise Ds006780Error("no primary ds006780 run-01 recordings passed metadata QC")
    out_of_scope.sort(
        key=lambda item: (item["subject_id"], item["run_id"], item["recording_path"])
    )
    manifest = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "dataset_version": str(dataset.get("dataset_version", openneuro_version)),
        "source_commit": source_commit,
        "protocol_sha256": canonical_sha256(config),
        "environment_identity": _environment_identity(project_root),
        "reference_policy": str(config["reference"]["primary_policy"]),
        "montage_identity": montage_identity(mne_module=mne_module),
        "candidate_runs": candidate_records,
        "out_of_scope_runs": out_of_scope,
        "dataset_metadata_files": metadata_files,
    }
    reject_target_fields(manifest)
    return manifest


def verify_ds006780_manifest(
    bids_root: Path,
    manifest: Mapping[str, Any],
    *,
    config_path: Path,
    project_root: Path,
) -> None:
    """Verify declared content identities without rewriting the manifest."""

    if not isinstance(manifest, Mapping):
        raise Ds006780Error("manifest must be an object")
    config = load_json_strict(config_path)
    if not isinstance(config, Mapping):
        raise Ds006780Error("external config must be a JSON object")
    validate_external_config(config)
    expected_protocol = canonical_sha256(config)
    if manifest.get("protocol_sha256") != expected_protocol:
        raise Ds006780Error("protocol_sha256 differs from supplied config")
    expected_environment = _environment_identity(project_root)
    if manifest.get("environment_identity") != expected_environment:
        raise Ds006780Error("environment identity differs from current environment")
    root = Path(bids_root).resolve()
    for candidate in manifest.get("candidate_runs", []):
        for file_record in candidate.get("acquisition_files", []):
            path = root / file_record["path"]
            if not path.is_file():
                raise Ds006780Error(f"manifest file is missing: {path}")
            actual = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if actual != {
                "size_bytes": file_record["size_bytes"],
                "sha256": file_record["sha256"],
            }:
                raise Ds006780Error(f"manifest file identity drift: {path}")
    for file_record in manifest.get("dataset_metadata_files", []):
        path = root / file_record["path"]
        if not path.is_file():
            raise Ds006780Error(f"metadata file is missing: {path}")
        if path.stat().st_size != file_record["size_bytes"] or sha256_file(path) != file_record["sha256"]:
            raise Ds006780Error(f"metadata file identity drift: {path}")
    claimed = manifest.get("manifest_sha256")
    if claimed != canonical_sha256(manifest, exclude_fields=("manifest_sha256",)):
        raise Ds006780Error("manifest_sha256 is invalid")


def write_manifest(
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


def apply_reference_policy(
    raw: Any,
    *,
    policy: str,
    source_reference_value: str,
) -> tuple[Any, str, str]:
    """Apply the explicit reference policy and return raw, provenance, operation."""

    if policy == "preserve_acquisition":
        if source_reference_value == "n/a":
            provenance = "unknown"
        else:
            provenance = "declared"
        return raw, provenance, "none"
    if policy == "average_reference":
        if not hasattr(raw, "set_eeg_reference"):
            raise Ds006780Error("reference_policy_invalid: raw has no reference operation")
        raw.set_eeg_reference(ref_channels="average", verbose="ERROR")
        provenance = "unknown" if source_reference_value == "n/a" else "declared"
        return raw, provenance, "average_reference_applied"
    raise Ds006780Error(f"reference_policy_invalid: {policy}")


def preprocess_candidate_windows(
    raw: Any,
    *,
    start_sample: int,
    end_sample: int,
    channel_labels: tuple[str, ...],
    contract: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    data = np.asarray(raw.get_data()[:, start_sample:end_sample], dtype=np.float64)
    try:
        return preprocess_rest_blocks(
            [data],
            original_frequency_hz=float(raw.info["sfreq"]),
            channel_labels=channel_labels,
            contract=contract,
        )
    except PreprocessingError as error:
        raise Ds006780Error(str(error)) from error


def build_target_free_qc(
    bids_root: Path,
    candidate: Mapping[str, Any],
    *,
    manifest_sha256: str,
    config: Mapping[str, Any],
    raw_loader: Callable[[Path], Any] | None = None,
    mne_module: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one manifest candidate and emit only signal-level QC."""

    root = Path(bids_root).resolve()
    recording_path = root / str(candidate["recording_path"])
    channels_path = root / str(candidate["channels_path"])
    events_path = root / str(candidate["events_path"])
    sidecar_path = root / str(candidate["sidecar_path"])
    raw = _load_raw(recording_path, raw_loader)
    metadata = _raw_metadata(raw)
    if not math.isclose(
        metadata.sampling_frequency_hz,
        EXPECTED_SOURCE_SAMPLING_FREQUENCY_HZ,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise Ds006780Error("unsupported_sampling_frequency")
    inventory, channel_names = parse_channel_inventory(
        channels_path,
        expected_sampling_frequency_hz=metadata.sampling_frequency_hz,
        mne_module=mne_module,
    )
    raw_names = tuple(str(name) for name in raw.ch_names)
    if raw_names != channel_names:
        raise Ds006780Error("sidecar_mismatch: raw channel order differs from channels.tsv")
    events = parse_event_inventory(
        events_path,
        sampling_frequency_hz=metadata.sampling_frequency_hz,
        n_times=metadata.n_times,
    )
    start_sample, end_sample = recording_bounds(events, n_times=metadata.n_times)
    sidecar = _sidecar_metadata(sidecar_path)
    source_reference = str(sidecar.get("EEGReference", "n/a")).strip() or "n/a"
    policy = str(config["reference"]["primary_policy"])
    raw, reference_provenance, reference_operation = apply_reference_policy(
        raw,
        policy=policy,
        source_reference_value=source_reference,
    )
    eeg_indices = [index for index, item in enumerate(inventory) if item["type"] == "EEG"]
    if len(eeg_indices) != 64:
        raise Ds006780Error("channel_layout_mismatch: expected 64 EEG channels")
    source_data = np.asarray(raw.get_data(), dtype=np.float64)
    if source_data.ndim != 2 or source_data.shape[0] != len(raw_names):
        raise Ds006780Error("sidecar_mismatch: raw geometry differs from channel inventory")
    selected = source_data[eeg_indices, start_sample:end_sample]
    contract = load_preprocessing_contract(config)
    try:
        windows, preprocessing_qc = preprocess_rest_blocks(
            [selected],
            original_frequency_hz=metadata.sampling_frequency_hz,
            channel_labels=tuple(channel_names[index] for index in eeg_indices),
            contract=contract,
        )
    except PreprocessingError as error:
        raise Ds006780Error(str(error)) from error
    rejected_channels = [item["name"] for item in inventory if item["type"] != "EEG"]
    qc = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "subject_id": str(candidate["subject_id"]),
        "run_id": str(candidate["run_id"]),
        "channel_labels": [channel_names[index] for index in eeg_indices],
        "mapped_channel_count": len(eeg_indices),
        "rejected_channels": rejected_channels,
        "reference_policy": policy,
        "reference_provenance": reference_provenance,
        "reference_operation": reference_operation,
        "source_reference_value": source_reference,
        "montage_identity": montage_identity(mne_module=mne_module),
        "original_frequency_hz": metadata.sampling_frequency_hz,
        "output_frequency_hz": preprocessing_qc["output_frequency_hz"],
        "selected_duration_seconds": preprocessing_qc["selected_duration_seconds"],
        "window_count": int(windows.shape[0]),
        "window_shape": [int(value) for value in windows.shape],
        "cross_block_windows": False,
        "spatial_interpolation": False,
        "finite_output": bool(np.isfinite(windows).all()),
        "qc_reasons": ["none"],
    }
    reject_target_fields(qc)
    if not qc["finite_output"]:
        raise Ds006780Error("signal_nonfinite")
    return windows, qc


def write_qc_report(
    output_path: Path,
    qc: Mapping[str, Any],
    *,
    schema_path: Path,
) -> dict[str, Any]:
    reject_target_fields(qc)
    return write_create_only_json(output_path, qc, schema_path, "qc_sha256")
