"""Metadata-only MIPDB inventory and deterministic cohort assignment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..research.protocol import PreprocessingContract
from ..research.protocol import StudyProtocol


EEG_EXTENSIONS = {".bdf", ".edf", ".eeg", ".fif", ".set", ".vhdr"}


class MipdbInventoryError(ValueError):
    """Raised when MIPDB metadata cannot define an auditable cohort."""


class MipdbPreprocessingError(ValueError):
    """Raised when an external recording violates the sealed signal contract."""


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


def build_mipdb_inventory(
    bids_root: Path,
    *,
    protocol: StudyProtocol,
    hbn_age_support: tuple[float, float],
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

    try:
        with participants_path.open("r", encoding="utf-8", newline="") as handle:
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

    dataset_identity = {
        "nemar_id": protocol.datasets.external_nemar_id,
        "release": protocol.datasets.external_release,
        "participants_sha256": _sha256_file(participants_path),
        "dataset_description_sha256": _sha256_file(description_path),
        "subjects": normalized,
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
        "schema_version": 1,
        "dataset": "MIPDB",
        "nemar_id": protocol.datasets.external_nemar_id,
        "release": protocol.datasets.external_release,
        "protocol_sha256": protocol.sha256,
        "dataset_manifest_sha256": dataset_sha,
        "hbn_age_support": {"minimum": lower, "maximum": upper},
        "subjects": normalized,
        "exclusions": exclusions,
        "cohorts": cohorts,
        "subject_list_sha256": {
            name: _subject_list_sha256(subject_ids)
            for name, subject_ids in cohorts.items()
        },
        "underpowered": len(primary) < protocol.datasets.minimum_primary_subjects,
        "minimum_primary_subjects": protocol.datasets.minimum_primary_subjects,
    }


def preprocess_rest_blocks(
    blocks: Iterable[np.ndarray],
    *,
    original_frequency_hz: float,
    channel_labels: tuple[str, ...],
    contract: PreprocessingContract,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the common REVE input contract without crossing block boundaries."""

    from fractions import Fraction
    from scipy.signal import butter, resample_poly, sosfiltfilt

    source_frequency = float(original_frequency_hz)
    if not math.isfinite(source_frequency) or source_frequency <= 0:
        raise MipdbPreprocessingError("original frequency must be finite and positive")
    if not channel_labels or any(not label.strip() for label in channel_labels):
        raise MipdbPreprocessingError("channel labels must be non-empty")
    normalized_labels = [label.strip().casefold() for label in channel_labels]
    if len(set(normalized_labels)) != len(normalized_labels):
        raise MipdbPreprocessingError("duplicate channel labels are not allowed")

    arrays = [np.asarray(block, dtype=np.float64) for block in blocks]
    if not arrays:
        raise MipdbPreprocessingError("at least one resting block is required")
    for index, block in enumerate(arrays):
        if block.ndim != 2:
            raise MipdbPreprocessingError(f"block {index} must have channels x samples shape")
        if block.shape[0] != len(channel_labels):
            raise MipdbPreprocessingError(f"block {index} channel count does not match labels")
        if block.shape[1] < 1:
            raise MipdbPreprocessingError(f"block {index} has no samples")
        if not np.isfinite(block).all():
            raise MipdbPreprocessingError(f"block {index} contains non-finite samples")

    target_frequency = float(contract.sample_rate_hz)
    ratio = Fraction(target_frequency / source_frequency).limit_denominator(100_000)
    sos = butter(
        4,
        contract.bandpass_hz,
        btype="bandpass",
        fs=target_frequency,
        output="sos",
    )
    processed: list[np.ndarray] = []
    for index, block in enumerate(arrays):
        resampled = resample_poly(block, ratio.numerator, ratio.denominator, axis=1)
        try:
            filtered = sosfiltfilt(sos, resampled, axis=1)
        except ValueError as error:
            raise MipdbPreprocessingError(
                f"block {index} is too short for the declared band-pass filter"
            ) from error
        if contract.notch_hz:
            raise MipdbPreprocessingError(
                "notch filtering is not implemented because the approved protocol declares none"
            )
        processed.append(filtered)

    maximum_samples = int(round(contract.max_seconds_per_subject * target_frequency))
    selected: list[np.ndarray] = []
    included_blocks: list[dict[str, int]] = []
    remaining = maximum_samples
    for index, block in enumerate(processed):
        if remaining <= 0:
            break
        count = min(block.shape[1], remaining)
        if count > 0:
            selected.append(block[:, :count])
            included_blocks.append({"block_index": index, "selected_samples": count})
            remaining -= count
    if not selected:
        raise MipdbPreprocessingError("no usable samples remain after acquisition-order selection")

    scaler_source = np.concatenate(selected, axis=1)
    means = scaler_source.mean(axis=1, keepdims=True)
    scales = scaler_source.std(axis=1, keepdims=True)
    scales[scales == 0.0] = 1.0
    standardized = [
        np.clip((block - means) / scales, -contract.clamp, contract.clamp)
        for block in selected
    ]

    window_samples = int(round(contract.window_seconds * target_frequency))
    stride_samples = int(round(contract.stride_seconds * target_frequency))
    windows: list[np.ndarray] = []
    for block in standardized:
        for start in range(0, block.shape[1] - window_samples + 1, stride_samples):
            windows.append(block[:, start : start + window_samples])
    if not windows:
        raise MipdbPreprocessingError("no complete windows remain within individual blocks")
    stacked = np.stack(windows).astype(np.float32, copy=False)
    if not np.isfinite(stacked).all():
        raise MipdbPreprocessingError("preprocessing produced non-finite samples")
    qc = {
        "original_frequency_hz": source_frequency,
        "output_frequency_hz": target_frequency,
        "channel_labels": list(channel_labels),
        "mapped_channel_count": len(channel_labels),
        "rejected_channels": [],
        "included_blocks": included_blocks,
        "usable_duration_seconds": sum(block.shape[1] for block in processed)
        / target_frequency,
        "selected_duration_seconds": sum(block.shape[1] for block in selected)
        / target_frequency,
        "window_count": int(stacked.shape[0]),
        "window_samples": window_samples,
        "cross_block_windows": False,
        "spatial_interpolation": False,
        "scaler": contract.scaler,
        "clamp": contract.clamp,
        "qc_reasons": [],
    }
    return stacked, qc


def read_mipdb_bids_raw(bids_path: object) -> object:
    """Read one recording through a lazy optional MNE-BIDS dependency."""

    try:
        from mne_bids import read_raw_bids
    except ImportError as error:
        raise MipdbPreprocessingError(
            "MIPDB signal loading requires the project 'external' dependencies"
        ) from error
    return read_raw_bids(bids_path=bids_path, verbose=False)
