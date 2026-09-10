"""Content-addressed MIPDB inventory and deterministic cohort assignment."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..research.protocol import PreprocessingContract
from ..research.protocol import StudyProtocol
from .preprocessing import PreprocessingError
from .preprocessing import preprocess_rest_blocks as _shared_preprocess_rest_blocks


# BrainVision ``.eeg`` files are binary companions whose recording entry point
# is the corresponding ``.vhdr`` file. Counting both creates duplicate logical
# recordings and permits an unreadable binary payload to be selected as input.
EEG_EXTENSIONS = {".bdf", ".edf", ".fif", ".set", ".vhdr"}


class MipdbInventoryError(ValueError):
    """Raised when MIPDB metadata cannot define an auditable cohort."""


MipdbPreprocessingError = PreprocessingError


@dataclass(frozen=True)
class RestSegment:
    """One condition-bounded MIPDB resting interval in recording seconds."""

    condition: str
    start_s: float
    stop_s: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _subject_list_sha256(subject_ids: Iterable[str]) -> str:
    return _sha256_json(list(subject_ids))


def _eeg_recordings(root: Path, subject_id: str) -> list[str]:
    subject_root = root / subject_id
    if not subject_root.is_dir():
        return []
    return sorted(
        str(path.relative_to(root))
        for path in subject_root.rglob("*")
        if path.is_file() and path.suffix.lower() in EEG_EXTENSIONS
    )


def _acquisition_file_inventory(
    root: Path, subject_ids: Iterable[str]
) -> list[dict[str, Any]]:
    """Hash every file that can affect interpretation of the selected subjects."""

    root = Path(root).resolve()
    paths = {
        path.resolve()
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".json", ".tsv"}
    }
    for subject_id in subject_ids:
        subject_root = (root / subject_id).resolve()
        try:
            subject_root.relative_to(root)
        except ValueError as error:
            raise MipdbInventoryError("subject path escapes the BIDS root") from error
        if subject_root.is_dir():
            paths.update(path.resolve() for path in subject_root.rglob("*") if path.is_file())
    inventory: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise MipdbInventoryError("acquisition file escapes the BIDS root") from error
        inventory.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return inventory


def _parse_age(raw: object) -> tuple[float | None, str | None]:
    text = "" if raw is None else str(raw).strip()
    if not text or text.lower() in {"n/a", "na", "nan"}:
        return None, "missing_age"
    try:
        age = float(text)
    except ValueError:
        return None, "invalid_age"
    if not math.isfinite(age) or age <= 0:
        return None, "invalid_age"
    return age, None


def verify_mipdb_acquisition(
    bids_root: Path, manifest: dict[str, Any]
) -> str:
    """Verify the exact content-addressed MIPDB acquisition declared by a manifest."""

    root = Path(bids_root).resolve()
    subjects = manifest.get("subjects")
    expected_files = manifest.get("acquisition_files")
    if not isinstance(subjects, list) or not all(
        isinstance(item, dict) and isinstance(item.get("subject_id"), str)
        for item in subjects
    ):
        raise MipdbInventoryError("MIPDB manifest subjects are invalid")
    if not isinstance(expected_files, list) or not all(
        isinstance(item, dict)
        and set(item) == {"path", "size_bytes", "sha256"}
        and isinstance(item["path"], str)
        and isinstance(item["size_bytes"], int)
        and item["size_bytes"] >= 0
        and isinstance(item["sha256"], str)
        and len(item["sha256"]) == 64
        for item in expected_files
    ):
        raise MipdbInventoryError("MIPDB acquisition inventory is invalid")
    actual_files = _acquisition_file_inventory(
        root, (str(item["subject_id"]) for item in subjects)
    )
    if actual_files != expected_files:
        raise MipdbInventoryError("MIPDB acquisition differs from the manifest")
    by_path = {item["path"]: item["sha256"] for item in actual_files}
    try:
        identity = {
            "nemar_id": manifest["nemar_id"],
            "release": manifest["release"],
            "participants_sha256": by_path["participants.tsv"],
            "dataset_description_sha256": by_path["dataset_description.json"],
            "subjects": subjects,
            "acquisition_files": actual_files,
            "hbn_age_support": manifest["hbn_age_support"],
            "age_source_sha256": manifest.get("age_source_sha256"),
        }
    except KeyError as error:
        raise MipdbInventoryError(
            "MIPDB acquisition identity is incomplete"
        ) from error
    actual_dataset_sha256 = _sha256_json(identity)
    if manifest.get("dataset_manifest_sha256") != actual_dataset_sha256:
        raise MipdbInventoryError("MIPDB dataset identity differs from the manifest")
    return actual_dataset_sha256


def build_mipdb_inventory(
    bids_root: Path,
    *,
    protocol: StudyProtocol,
    hbn_age_support: tuple[float, float],
    hbn_age_support_manifest_sha256: str | None = None,
    age_overrides: Mapping[str, float] | None = None,
    age_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Inspect BIDS metadata without loading EEG samples or model code."""

    bids_root = Path(bids_root)
    participants_path = bids_root / "participants.tsv"
    description_path = bids_root / "dataset_description.json"
    if not participants_path.is_file():
        raise MipdbInventoryError(f"missing participants.tsv: {participants_path}")
    if not description_path.is_file():
        raise MipdbInventoryError(f"missing dataset_description.json: {description_path}")
    lower, upper = (float(hbn_age_support[0]), float(hbn_age_support[1]))
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise MipdbInventoryError("HBN age support must be two finite increasing values")
    if hbn_age_support_manifest_sha256 is not None and (
        len(hbn_age_support_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in hbn_age_support_manifest_sha256
        )
    ):
        raise MipdbInventoryError("HBN age-support manifest hash is invalid")
    if age_source_sha256 is not None and (
        len(age_source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in age_source_sha256)
    ):
        raise MipdbInventoryError("MIPDB age-source hash is invalid")
    normalized_age_overrides = {
        str(subject): float(age) for subject, age in (age_overrides or {}).items()
    }
    if any(
        not math.isfinite(age) or age <= 0
        for age in normalized_age_overrides.values()
    ):
        raise MipdbInventoryError("MIPDB age overrides must be finite positive values")

    try:
        with participants_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    except (OSError, csv.Error) as error:
        raise MipdbInventoryError("could not parse participants.tsv") from error
    if not rows or "participant_id" not in rows[0] or "age" not in rows[0]:
        raise MipdbInventoryError("participants.tsv must contain participant_id and age")

    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for row in rows:
        subject_id = str(row.get("participant_id", "")).strip()
        if not subject_id:
            raise MipdbInventoryError("participants.tsv contains an empty participant_id")
        if subject_id in seen:
            raise MipdbInventoryError(f"duplicate participant_id: {subject_id}")
        seen.add(subject_id)
        age, age_error = _parse_age(row.get("age"))
        if age_error == "missing_age" and subject_id in normalized_age_overrides:
            age = normalized_age_overrides[subject_id]
            age_error = None
        elif age_error is None and subject_id in normalized_age_overrides:
            if not math.isclose(age, normalized_age_overrides[subject_id], rel_tol=0.0, abs_tol=1e-9):
                raise MipdbInventoryError(
                    f"MIPDB age source disagrees for {subject_id}"
                )
        recordings = _eeg_recordings(bids_root, subject_id)
        if age_error is not None:
            exclusions.append({"subject_id": subject_id, "reason": age_error})
            continue
        if not recordings:
            exclusions.append(
                {"subject_id": subject_id, "reason": "missing_eeg_recording"}
            )
            continue
        normalized.append(
            {
                "subject_id": subject_id,
                "age": age,
                "recordings": recordings,
            }
        )
    normalized.sort(key=lambda item: item["subject_id"])
    exclusions.sort(key=lambda item: (item["subject_id"], item["reason"]))
    if len(normalized) < protocol.datasets.pilot_size:
        raise MipdbInventoryError(
            f"MIPDB requires at least {protocol.datasets.pilot_size} eligible subjects"
        )

    acquisition_files = _acquisition_file_inventory(
        bids_root, (str(item["subject_id"]) for item in normalized)
    )
    dataset_identity = {
        "nemar_id": protocol.datasets.external_nemar_id,
        "release": protocol.datasets.external_release,
        "participants_sha256": _sha256_file(participants_path),
        "dataset_description_sha256": _sha256_file(description_path),
        "subjects": normalized,
        "acquisition_files": acquisition_files,
        "hbn_age_support": {
            "minimum": lower,
            "maximum": upper,
            "source_manifest_sha256": hbn_age_support_manifest_sha256,
        },
        "age_source_sha256": age_source_sha256,
    }
    dataset_sha = _sha256_json(dataset_identity)

    def pilot_key(subject: dict[str, Any]) -> tuple[str, str]:
        subject_id = str(subject["subject_id"])
        value = (
            dataset_sha
            + "\0"
            + subject_id
            + "\0"
            + protocol.datasets.pilot_salt
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest(), subject_id

    pilot = [
        str(subject["subject_id"])
        for subject in sorted(normalized, key=pilot_key)[: protocol.datasets.pilot_size]
    ]
    pilot_set = set(pilot)
    primary: list[str] = []
    extrapolation: list[str] = []
    for subject in normalized:
        subject_id = str(subject["subject_id"])
        if subject_id in pilot_set:
            continue
        age = float(subject["age"])
        if lower <= age <= upper:
            primary.append(subject_id)
        elif age > upper:
            extrapolation.append(subject_id)
        else:
            exclusions.append(
                {"subject_id": subject_id, "reason": "below_hbn_age_support"}
            )
    exclusions.sort(key=lambda item: (item["subject_id"], item["reason"]))
    cohorts = {
        "pilot": pilot,
        "primary": primary,
        "extrapolation": extrapolation,
    }
    return {
        "schema_version": 2,
        "status": "draft",
        "dataset": "MIPDB",
        "nemar_id": protocol.datasets.external_nemar_id,
        "release": protocol.datasets.external_release,
        "protocol_sha256": protocol.sha256,
        "dataset_manifest_sha256": dataset_sha,
        "hbn_age_support": dataset_identity["hbn_age_support"],
        "age_source_sha256": age_source_sha256,
        "subjects": normalized,
        "acquisition_files": acquisition_files,
        "exclusions": exclusions,
        "cohorts": cohorts,
        "subject_list_sha256": {
            name: _subject_list_sha256(subject_ids)
            for name, subject_ids in cohorts.items()
        },
        "underpowered": len(primary) < protocol.datasets.minimum_primary_subjects,
        "minimum_primary_subjects": protocol.datasets.minimum_primary_subjects,
    }


def parse_mipdb_rest_segments(
    events_path: Path,
    *,
    recording_duration_s: float,
    contract: PreprocessingContract,
) -> tuple[RestSegment, ...]:
    """Parse the predeclared block01 marker stream without inventing boundaries."""

    duration = float(recording_duration_s)
    if not math.isfinite(duration) or duration <= 0:
        raise MipdbPreprocessingError("recording duration must be finite and positive")
    try:
        with Path(events_path).open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    except (OSError, csv.Error) as error:
        raise MipdbPreprocessingError(
            f"could not parse MIPDB events: {events_path}"
        ) from error
    if not rows or not {"onset", "trial_type"}.issubset(rows[0]):
        raise MipdbPreprocessingError(
            "MIPDB events must contain onset and trial_type columns"
        )

    condition_by_marker = dict(contract.external_condition_markers)
    relevant_markers = {
        contract.external_paradigm_start_marker,
        *condition_by_marker,
    }
    relevant: list[tuple[float, str]] = []
    previous_onset = -float("inf")
    for index, row in enumerate(rows):
        marker = str(row.get("trial_type", "")).strip()
        if marker not in relevant_markers:
            continue
        try:
            onset = float(str(row.get("onset", "")).strip())
        except ValueError as error:
            raise MipdbPreprocessingError(
                f"MIPDB events row {index} has an invalid onset"
            ) from error
        if not math.isfinite(onset) or onset < 0 or onset >= duration:
            raise MipdbPreprocessingError(
                f"MIPDB events row {index} onset is outside the recording"
            )
        if onset <= previous_onset:
            raise MipdbPreprocessingError(
                "MIPDB relevant event onsets must be strictly increasing"
            )
        previous_onset = onset
        relevant.append((onset, marker))

    starts = [item for item in relevant if item[1] == contract.external_paradigm_start_marker]
    if not starts:
        raise MipdbPreprocessingError(
            "MIPDB resting stream requires a paradigm-start marker"
        )
    # Some recordings contain an incomplete first attempt followed by a
    # restarted, complete resting stream.  The last start marker is the
    # deterministic boundary; all earlier markers are precondition data.
    start_onset = starts[-1][0]
    conditions = [
        (onset, marker)
        for onset, marker in relevant
        if marker in condition_by_marker and onset > start_onset
    ]
    for (_, previous), (_, current) in zip(conditions, conditions[1:], strict=False):
        if previous == current:
            raise MipdbPreprocessingError(
                "MIPDB eyes-open and eyes-closed markers must alternate"
            )
    observed_conditions = {condition_by_marker[marker] for _, marker in conditions}
    expected_conditions = set(condition_by_marker.values())
    if observed_conditions != expected_conditions:
        raise MipdbPreprocessingError(
            "MIPDB resting stream must contain both eyes-open and eyes-closed markers"
        )

    segments: list[RestSegment] = []
    for index, (onset, marker) in enumerate(conditions):
        stop = conditions[index + 1][0] if index + 1 < len(conditions) else duration
        if stop <= onset:
            raise MipdbPreprocessingError("MIPDB resting segment has invalid duration")
        segments.append(
            RestSegment(
                condition=condition_by_marker[marker],
                start_s=onset,
                stop_s=stop,
            )
        )
    return tuple(segments)


def _mipdb_recording_path(
    bids_root: Path,
    subject: Mapping[str, Any],
    contract: PreprocessingContract,
) -> Path:
    subject_id = subject.get("subject_id")
    recordings = subject.get("recordings")
    if not isinstance(subject_id, str) or not subject_id.strip():
        raise MipdbPreprocessingError("MIPDB subject record has no valid subject_id")
    if not isinstance(recordings, list) or not all(
        isinstance(item, str) for item in recordings
    ):
        raise MipdbPreprocessingError("MIPDB subject recordings must be an array")
    token = f"_task-{contract.external_resting_task}_eeg."
    candidates = [item for item in recordings if token in Path(item).name]
    if len(candidates) != 1:
        raise MipdbPreprocessingError(
            f"{subject_id} requires exactly one task-{contract.external_resting_task} recording"
        )
    root = Path(bids_root).resolve()
    path = (root / candidates[0]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise MipdbPreprocessingError("MIPDB recording escapes the BIDS root") from error
    if not path.is_file():
        raise MipdbPreprocessingError(f"MIPDB recording is missing: {path}")
    return path


def _read_mipdb_raw(path: Path) -> object:
    try:
        import mne
    except ImportError as error:
        raise MipdbPreprocessingError(
            "MIPDB signal loading requires the project 'external' dependencies"
        ) from error
    try:
        raw = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
        channels_path = path.with_name(
            path.name.replace("_eeg.vhdr", "_channels.tsv")
        )
        if not channels_path.is_file():
            raise MipdbPreprocessingError(f"MIPDB channels are missing: {channels_path}")
        with channels_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        required = {"name", "type", "status"}
        if not rows or not required <= set(rows[0]):
            raise MipdbPreprocessingError(
                f"MIPDB channels sidecar lacks required columns: {channels_path}"
            )
        raw_names = [str(name) for name in raw.ch_names]
        sidecar_names = [str(row["name"]).strip() for row in rows]
        if sidecar_names != [name.strip() for name in raw_names]:
            raise MipdbPreprocessingError(
                f"MIPDB channels differ from raw data: {channels_path}"
            )
        type_map = {
            "EEG": "eeg",
            "MISC": "misc",
            "EOG": "eog",
            "ECG": "ecg",
            "EMG": "emg",
            "STIM": "stim",
        }
        channel_types: dict[str, str] = {}
        bads: list[str] = []
        for raw_name, row in zip(raw_names, rows, strict=True):
            name = str(row["name"]).strip()
            label = str(row["type"]).strip().upper()
            if label not in type_map:
                raise MipdbPreprocessingError(
                    f"unsupported MIPDB channel type {label!r} for {name}"
                )
            channel_types[raw_name] = type_map[label]
            if str(row["status"]).strip().lower() == "bad":
                bads.append(raw_name)
        raw.set_channel_types(channel_types, on_unit_change="ignore")
        raw.info["bads"] = bads
        return raw
    except Exception as error:
        if isinstance(error, MipdbPreprocessingError):
            raise
        raise MipdbPreprocessingError(f"could not load MIPDB recording: {path}") from error


def load_mipdb_resting_subject(
    bids_root: Path,
    subject: Mapping[str, Any],
    *,
    contract: PreprocessingContract,
    raw_loader: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and preprocess one block01 subject under the sealed event contract."""

    recording_path = _mipdb_recording_path(bids_root, subject, contract)
    stem = recording_path.name.rsplit("_eeg.", 1)[0]
    events_path = recording_path.with_name(f"{stem}_events.tsv")
    if not events_path.is_file():
        raise MipdbPreprocessingError(f"MIPDB events are missing: {events_path}")
    loader = raw_loader or _read_mipdb_raw
    try:
        raw = loader(recording_path)
        source_frequency = float(raw.info["sfreq"])
        n_times = int(raw.n_times)
        channel_names = [str(name).strip() for name in raw.ch_names]
        channel_types = list(raw.get_channel_types())
        bads = {str(name).strip() for name in raw.info.get("bads", [])}
        all_data = np.asarray(raw.get_data(), dtype=np.float64)
    except MipdbPreprocessingError:
        raise
    except Exception as error:
        raise MipdbPreprocessingError(
            f"MIPDB raw object is invalid for {recording_path}"
        ) from error
    if (
        not math.isfinite(source_frequency)
        or source_frequency <= 0
        or n_times <= 0
        or len(channel_names) != len(channel_types)
        or all_data.shape != (len(channel_names), n_times)
    ):
        raise MipdbPreprocessingError("MIPDB raw geometry is invalid")

    eeg_by_name: dict[str, int] = {}
    for index, (name, channel_type) in enumerate(zip(channel_names, channel_types, strict=True)):
        if channel_type == "eeg":
            if name in eeg_by_name:
                raise MipdbPreprocessingError("duplicate mapped EEG channel label")
            eeg_by_name[name] = index
    expected_labels = tuple(
        f"E{index}" for index in range(1, contract.required_mapped_channels + 1)
    )
    if contract.mapped_channel_layout != "egi_hydrocel_e1_e128":
        raise MipdbPreprocessingError("unsupported mapped channel layout")
    if set(eeg_by_name) != set(expected_labels):
        missing = sorted(set(expected_labels) - set(eeg_by_name))
        extra = sorted(set(eeg_by_name) - set(expected_labels))
        raise MipdbPreprocessingError(
            f"mapped channel layout mismatch: missing={missing} extra={extra}"
        )
    bad_mapped = sorted(bads & set(expected_labels))
    if bad_mapped:
        raise MipdbPreprocessingError(f"bad mapped channels are forbidden: {bad_mapped}")

    recording_duration = n_times / source_frequency
    segments = parse_mipdb_rest_segments(
        events_path,
        recording_duration_s=recording_duration,
        contract=contract,
    )
    eeg_data = all_data[[eeg_by_name[name] for name in expected_labels]]
    blocks: list[np.ndarray] = []
    for segment in segments:
        start = int(round(segment.start_s * source_frequency))
        stop = min(n_times, int(round(segment.stop_s * source_frequency)))
        if stop <= start:
            raise MipdbPreprocessingError("MIPDB segment resolves to no samples")
        blocks.append(eeg_data[:, start:stop])
    windows, qc = preprocess_rest_blocks(
        blocks,
        original_frequency_hz=source_frequency,
        channel_labels=expected_labels,
        contract=contract,
    )
    root = Path(bids_root).resolve()
    qc.update(
        {
            "resting_task": contract.external_resting_task,
            "paradigm_start_marker": contract.external_paradigm_start_marker,
            "condition_markers": dict(contract.external_condition_markers),
            "condition_sequence": [segment.condition for segment in segments],
            "condition_segments": [
                {
                    "condition": segment.condition,
                    "start_s": segment.start_s,
                    "stop_s": segment.stop_s,
                }
                for segment in segments
            ],
            "precondition_discarded_seconds": segments[0].start_s,
            "source_recording": str(recording_path.relative_to(root)),
            "events_file": str(events_path.relative_to(root)),
        }
    )
    return windows, qc


def _contains_outcome_field(value: object) -> bool:
    forbidden = ("age", "target", "prediction", "metric", "pearson", "mae", "rmse")
    if isinstance(value, dict):
        return any(
            any(token in str(key).casefold() for token in forbidden)
            or _contains_outcome_field(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_outcome_field(item) for item in value)
    return False


def _write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise MipdbInventoryError(f"immutable artifact already exists: {path}") from error


def finalize_mipdb_cohort(
    *,
    bids_root: Path,
    draft_manifest_path: Path,
    protocol: StudyProtocol,
    qc_output_path: Path,
    output_path: Path,
    subject_loader: Any = None,
) -> dict[str, Any]:
    """Finalize evaluation cohorts after predeclared, target-free signal QC."""

    draft_path = Path(draft_manifest_path)
    try:
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MipdbInventoryError("could not read draft MIPDB manifest") from error
    if (
        not isinstance(draft, dict)
        or draft.get("schema_version") != 2
        or draft.get("status") != "draft"
        or draft.get("protocol_sha256") != protocol.sha256
    ):
        raise MipdbInventoryError("MIPDB cohort finalization requires the exact draft manifest")
    verify_mipdb_acquisition(Path(bids_root), draft)
    cohorts = draft.get("cohorts")
    subjects_raw = draft.get("subjects")
    if not isinstance(cohorts, dict) or not isinstance(subjects_raw, list):
        raise MipdbInventoryError("draft MIPDB cohort metadata is invalid")
    pilot = cohorts.get("pilot")
    primary = cohorts.get("primary")
    extrapolation = cohorts.get("extrapolation")
    if not all(isinstance(value, list) for value in (pilot, primary, extrapolation)):
        raise MipdbInventoryError("draft MIPDB cohorts must be ordered arrays")
    if len(pilot) != protocol.datasets.pilot_size:
        raise MipdbInventoryError("draft MIPDB pilot size differs from the protocol")
    candidate_ids = [*primary, *extrapolation]
    if (
        len(candidate_ids) != len(set(candidate_ids))
        or set(pilot) & set(candidate_ids)
    ):
        raise MipdbInventoryError("draft MIPDB cohorts overlap or contain duplicates")
    subjects = {
        item.get("subject_id"): item
        for item in subjects_raw
        if isinstance(item, dict) and isinstance(item.get("subject_id"), str)
    }
    if len(subjects) != len(subjects_raw) or any(
        subject_id not in subjects for subject_id in [*pilot, *candidate_ids]
    ):
        raise MipdbInventoryError("draft MIPDB subject inventory is incomplete")

    loader = subject_loader or load_mipdb_resting_subject
    passed: set[str] = set()
    qc_rows: list[dict[str, Any]] = []
    qc_exclusions: list[dict[str, str]] = []
    for subject_id in candidate_ids:
        try:
            windows, raw_qc = loader(
                Path(bids_root), subjects[subject_id], contract=protocol.preprocessing
            )
            array = np.asarray(windows)
            if (
                array.ndim != 3
                or array.shape[0] <= 0
                or not np.issubdtype(array.dtype, np.number)
                or not np.isfinite(array).all()
                or not isinstance(raw_qc, dict)
                or raw_qc.get("window_count") != int(array.shape[0])
                or raw_qc.get("qc_reasons") != []
                or _contains_outcome_field(raw_qc)
            ):
                raise MipdbPreprocessingError("target-free QC evidence is invalid")
        except (MipdbPreprocessingError, OSError, ValueError) as error:
            detail = str(error).strip() or type(error).__name__
            qc_rows.append(
                {
                    "subject_id": subject_id,
                    "status": "excluded",
                    "reason": "predeclared_signal_qc_failed",
                    "detail": detail,
                }
            )
            qc_exclusions.append(
                {
                    "subject_id": subject_id,
                    "reason": "predeclared_signal_qc_failed",
                    "detail": detail,
                }
            )
            continue
        passed.add(subject_id)
        qc_rows.append(
            {
                "subject_id": subject_id,
                "status": "passed",
                "window_count": int(array.shape[0]),
                "qc_sha256": _sha256_json(raw_qc),
            }
        )

    draft_sha256 = _sha256_file(draft_path)
    qc_body: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "protocol_sha256": protocol.sha256,
        "dataset_manifest_sha256": draft["dataset_manifest_sha256"],
        "draft_manifest_sha256": draft_sha256,
        "candidate_subject_count": len(candidate_ids),
        "passed_subject_count": len(passed),
        "excluded_subject_count": len(qc_exclusions),
        "subjects": qc_rows,
    }
    qc_report = {**qc_body, "cohort_qc_sha256": _sha256_json(qc_body)}
    _write_json_create_only(Path(qc_output_path), qc_report)

    final_cohorts = {
        "pilot": list(pilot),
        "primary": [subject_id for subject_id in primary if subject_id in passed],
        "extrapolation": [
            subject_id for subject_id in extrapolation if subject_id in passed
        ],
    }
    final_exclusions = [*draft.get("exclusions", []), *qc_exclusions]
    final_exclusions.sort(key=lambda item: (item["subject_id"], item["reason"]))
    finalized = {
        **draft,
        "status": "finalized",
        "draft_manifest_sha256": draft_sha256,
        "cohort_qc_sha256": qc_report["cohort_qc_sha256"],
        "cohorts": final_cohorts,
        "subject_list_sha256": {
            name: _subject_list_sha256(subject_ids)
            for name, subject_ids in final_cohorts.items()
        },
        "exclusions": final_exclusions,
        "underpowered": len(final_cohorts["primary"])
        < protocol.datasets.minimum_primary_subjects,
    }
    _write_json_create_only(Path(output_path), finalized)
    return finalized


def preprocess_rest_blocks(
    blocks: Iterable[np.ndarray],
    *,
    original_frequency_hz: float,
    channel_labels: tuple[str, ...],
    contract: PreprocessingContract,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compatibility wrapper for the shared REVE preprocessing primitive."""

    return _shared_preprocess_rest_blocks(
        blocks,
        original_frequency_hz=original_frequency_hz,
        channel_labels=channel_labels,
        contract=contract,
    )


def read_mipdb_bids_raw(bids_path: object) -> object:
    """Read one recording through a lazy optional MNE-BIDS dependency."""

    try:
        from mne_bids import read_raw_bids
    except ImportError as error:
        raise MipdbPreprocessingError(
            "MIPDB signal loading requires the project 'external' dependencies"
        ) from error
    return read_raw_bids(bids_path=bids_path, verbose=False)
