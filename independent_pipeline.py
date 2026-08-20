#!/usr/bin/env python3
"""Independent reproduction of the NeuralBench REVE Age pipeline.

The official NeuralBench command remains the reference implementation.  This
module deliberately does not call ``neuralbench.run_benchmark``.  It owns the
Age manifest, HBN reader, preprocessing, PyTorch training loop, and metric
calculation, while loading the same pretrained REVE backbone and channel
mapping.

The expensive HBN/REVE dependencies are imported lazily.  Consequently the
manifest and preprocessing contracts can be tested with NumPy alone.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

from .reve_baseline import (
    AGE_TASK,
    REVE_MODEL,
    AgeTaskConfig,
    ReveModelConfig,
    ReveDependencyError,
    build_window_starts,
    pearsonr,
)
from .medium_subset import (
    MANIFEST_FIELDS,
    filter_recordings_by_manifest,
    manifest_sha256,
    read_manifest,
)


HBN_RELEASES = tuple(f"R{i}" for i in range(1, 12))

# This version is part of the cache key.  It prevents a cache made from the
# intended full-recording preprocessing from being reused after matching the
# official NeuralSet chunking semantics.
PREPROCESSING_CACHE_VERSION = "neuralset-mneraw-chunk-v1"

# These exclusions are part of the official Shirazi2024Hbn study reader.  They
# are not an optional quality filter: omitting them changes the benchmark set.
BAD_HBN_SUBJECTS = frozenset(
    {
        "sub-NDARWV769JM7",
        "sub-NDARME789TD2",
        "sub-NDARUA442ZVF",
        "sub-NDARJP304NK1",
        "sub-NDARTY128YLU",
        "sub-NDARDW550GU6",
        "sub-NDARLD243KRE",
        "sub-NDARUJ292JXV",
        "sub-NDARBA381JGH",
    }
)


class IndependentPipelineError(RuntimeError):
    """Raised when the independent runner cannot satisfy its contract."""


@dataclass(frozen=True)
class HbnRecording:
    """One HBN EEGLAB recording and its participant-level target metadata."""

    path: Path
    release: str
    subject: str
    task: str
    age: float | None
    duration_s: float


@dataclass(frozen=True)
class AgeWindow:
    """A manifest row for one 2-second Age example."""

    path: Path
    release: str
    subject: str
    age: float
    start_s: float
    duration_s: float
    recording_duration_s: float
    split: Literal["train", "val", "test"]


@dataclass(frozen=True)
class PreparedRecording:
    """Preprocessed continuous recording used to slice windows."""

    data: np.ndarray
    channel_names: tuple[str, ...]


def build_global_channel_order(
    prepared_recordings: Iterable[PreparedRecording],
) -> tuple[str, ...]:
    """Build NeuralBench's first-seen global channel order.

    ``EegExtractor.prepare`` does not treat channel position as local to each
    recording.  It discovers one global order while preparing the study and
    zero-pads channels that are absent from an individual recording.  Keeping
    that operation explicit prevents the independent runner from silently
    feeding different physical channels into the same REVE input position.
    """

    order: list[str] = []
    seen: set[str] = set()
    for prepared in prepared_recordings:
        data = np.asarray(prepared.data)
        if data.ndim != 2:
            raise ValueError("prepared recording data must have shape (channels, time)")
        if len(prepared.channel_names) != data.shape[0]:
            raise ValueError("channel_names must match the data channel dimension")
        if len(set(prepared.channel_names)) != len(prepared.channel_names):
            raise ValueError("each prepared recording must have unique channel names")
        for channel_name in prepared.channel_names:
            if channel_name not in seen:
                seen.add(channel_name)
                order.append(channel_name)
    if not order:
        raise ValueError("at least one EEG channel is required")
    return tuple(order)


def align_prepared_recording(
    prepared: PreparedRecording,
    channel_order: Sequence[str],
) -> PreparedRecording:
    """Align one recording to a global channel order with zero-padding."""

    source_data = np.asarray(prepared.data)
    order = tuple(channel_order)
    if source_data.ndim != 2:
        raise ValueError("prepared recording data must have shape (channels, time)")
    if len(prepared.channel_names) != source_data.shape[0]:
        raise ValueError("channel_names must match the data channel dimension")
    if len(set(prepared.channel_names)) != len(prepared.channel_names):
        raise ValueError("each prepared recording must have unique channel names")
    if len(set(order)) != len(order) or not order:
        raise ValueError("channel_order must contain unique channel names")

    # HBN recordings normally already share the global order.  Reusing that
    # float32 array avoids copying the complete recording during preload.
    if tuple(prepared.channel_names) == order and source_data.dtype == np.float32:
        return PreparedRecording(source_data, order)

    data = np.asarray(source_data, dtype=np.float32)

    target_indices = {name: index for index, name in enumerate(order)}
    unknown = set(prepared.channel_names).difference(target_indices)
    if unknown:
        raise ValueError(f"prepared recording contains channels missing from the global order: {sorted(unknown)}")
    aligned = np.zeros((len(order), data.shape[1]), dtype=np.float32)
    for source_index, channel_name in enumerate(prepared.channel_names):
        aligned[target_indices[channel_name]] = data[source_index]
    return PreparedRecording(aligned, order)


@dataclass(frozen=True)
class IndependentTrainConfig:
    """Training defaults copied from NeuralBench's global configuration."""

    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    patience: int = 7
    gradient_clip_norm: float = 1.0
    seed: int = 0
    data_seed: int = 33
    device: str = "auto"
    num_workers: int = 2
    prefetch_factor: int | None = None
    persistent_workers: bool = True
    preload_recordings: bool = True
    freeze_backbone: bool = False
    validate_before_training: bool = True

    def __post_init__(self) -> None:
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive or None")


@dataclass(frozen=True)
class TrainingResult:
    """Scores and predictions produced by the independent trainer."""

    best_epoch: int
    val_pearsonr: float
    test_pearsonr: float
    test_mse: float
    test_metrics: dict[str, float]
    timings: dict[str, float]
    targets: np.ndarray
    predictions: np.ndarray


@dataclass(frozen=True)
class PredictionComparison:
    """Comparison of two predictions on the exact same target rows."""

    own_pearsonr: float
    official_pearsonr: float
    pearsonr_delta: float
    matches: bool
    n_observations: int


@dataclass(frozen=True)
class ScoreComparison:
    """Comparison when the official runner exposes only its aggregate score."""

    own_pearsonr: float
    official_pearsonr: float
    pearsonr_delta: float
    matches: bool


_WINDOW_CSV_FIELDS = (
    "path",
    "release",
    "subject",
    "age",
    "start_s",
    "duration_s",
    "recording_duration_s",
    "split",
)
_PREDICTION_CSV_FIELDS = _WINDOW_CSV_FIELDS + ("prediction",)


def _window_row(window: AgeWindow) -> dict[str, Any]:
    return {
        "path": str(window.path),
        "release": window.release,
        "subject": window.subject,
        "age": float(window.age),
        "start_s": float(window.start_s),
        "duration_s": float(window.duration_s),
        "recording_duration_s": float(window.recording_duration_s),
        "split": window.split,
    }


def build_manifest_rows(windows: Sequence[AgeWindow]) -> list[dict[str, Any]]:
    """Serialize the exact window identity used by the independent runner."""

    return [_window_row(window) for window in windows]


def build_prediction_rows(
    windows: Sequence[AgeWindow], predictions: Sequence[float]
) -> list[dict[str, Any]]:
    """Attach predictions to manifest rows without losing window identity."""

    values = np.asarray(predictions, dtype=float).reshape(-1)
    if len(windows) != values.size:
        raise ValueError("windows and predictions must have equal lengths")
    return [
        {**_window_row(window), "prediction": float(prediction)}
        for window, prediction in zip(windows, values, strict=True)
    ]


def _write_rows(
    rows: Sequence[Mapping[str, Any]], path: Path, fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest_rows(windows: Sequence[AgeWindow], path: Path) -> None:
    """Write the complete Age window manifest as a comparison artifact."""

    _write_rows(build_manifest_rows(windows), path, _WINDOW_CSV_FIELDS)


def write_prediction_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write per-window predictions in a stable, comparison-friendly format."""

    _write_rows(rows, path, _PREDICTION_CSV_FIELDS)


def read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    """Read a prediction artifact written by ``write_prediction_rows``."""

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = set(_PREDICTION_CSV_FIELDS)
    if rows and not required.issubset(rows[0]):
        missing = sorted(required.difference(rows[0]))
        raise ValueError(f"prediction file is missing columns: {missing}")
    numeric_fields = {
        "age",
        "start_s",
        "duration_s",
        "recording_duration_s",
        "prediction",
    }
    for row in rows:
        for field in numeric_fields:
            row[field] = float(row[field])
    return rows


def regression_metrics(
    targets: Sequence[float], predictions: Sequence[float]
) -> dict[str, float]:
    """Compute NeuralBench's regression metrics on one complete evaluation set."""

    y_true = np.asarray(targets, dtype=float).reshape(-1)
    y_pred = np.asarray(predictions, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("targets and predictions must be non-empty and equally sized")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("targets and predictions must be finite")

    error = y_true - y_pred
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    try:
        correlation = float(pearsonr(y_true, y_pred))
    except ValueError:
        correlation = float("nan")
    target_mean = float(np.mean(y_true))
    total_sum_of_squares = float(np.sum((y_true - target_mean) ** 2))
    r2 = (
        float(1.0 - np.sum(error**2) / total_sum_of_squares)
        if total_sum_of_squares > 0
        else float("nan")
    )
    target_std = float(np.std(y_true))
    normalized_rmse = rmse / target_std if target_std > 0 else float("nan")
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": float(np.mean(np.abs(error))),
        "pearsonr": correlation,
        "r2_score": r2,
        "normalized_rmse": float(normalized_rmse),
    }


def filter_age_recordings(
    recordings: Iterable[HbnRecording],
    config: AgeTaskConfig = AGE_TASK,
) -> list[HbnRecording]:
    """Apply the official Age query and HBN study-level exclusions."""

    eligible: list[HbnRecording] = []
    for recording in recordings:
        if recording.subject in BAD_HBN_SUBJECTS:
            continue
        if recording.task != config.resting_state_task:
            continue
        if recording.age is None or not np.isfinite(recording.age):
            continue
        if recording.duration_s <= config.minimum_recording_duration_s:
            continue
        eligible.append(recording)
    return eligible


def _official_group_split(
    groups: Sequence[str], validation_ratio: float, random_state: int
) -> tuple[list[str], list[str]]:
    """Reproduce sklearn's ``train_test_split(..., shuffle=True)`` ordering.

    The official transform uses sklearn, but this small implementation keeps the
    manifest contract usable without making sklearn a hard dependency.  For a
    finite list, sklearn's ``ShuffleSplit`` takes the first permuted indices as
    validation and the remainder as train.
    """

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be strictly between 0 and 1")
    if len(groups) < 2:
        raise ValueError("at least two non-test release groups are required")

    n_validation = max(1, int(np.ceil(len(groups) * validation_ratio)))
    if n_validation >= len(groups):
        raise ValueError("validation split leaves no training release")

    permutation = np.random.RandomState(random_state).permutation(len(groups))
    validation = [groups[index] for index in permutation[:n_validation]]
    train = [groups[index] for index in permutation[n_validation:]]
    return train, validation


def assign_release_splits(
    releases: Iterable[str],
    config: AgeTaskConfig = AGE_TASK,
) -> dict[str, Literal["train", "val", "test"]]:
    """Assign train/val/test at release level exactly as ``PredefinedSplit``."""

    unique_releases = list(dict.fromkeys(releases))
    if config.test_release not in unique_releases:
        raise ValueError(f"test release {config.test_release!r} is missing")

    train_groups = [release for release in unique_releases if release != config.test_release]
    train, validation = _official_group_split(
        train_groups,
        validation_ratio=config.validation_release_ratio,
        random_state=config.validation_random_state,
    )
    assignments: dict[str, Literal["train", "val", "test"]] = {
        config.test_release: "test"
    }
    assignments.update({release: "train" for release in train})
    assignments.update({release: "val" for release in validation})
    return {release: assignments[release] for release in unique_releases}


def build_age_window_manifest(
    recordings: Iterable[HbnRecording],
    config: AgeTaskConfig = AGE_TASK,
) -> list[AgeWindow]:
    """Create the official 60-window-per-recording Age manifest."""

    eligible = sorted(
        filter_age_recordings(recordings, config),
        key=lambda recording: (
            int(recording.release[1:]),
            recording.subject,
            str(recording.path),
        ),
    )
    if not eligible:
        raise ValueError("the Age query returned no eligible recordings")

    split_by_release = assign_release_splits((recording.release for recording in eligible), config)
    manifest: list[AgeWindow] = []
    for recording in eligible:
        assert recording.age is not None
        for start_s in build_window_starts(recording.duration_s, config):
            manifest.append(
                AgeWindow(
                    path=recording.path,
                    release=recording.release,
                    subject=recording.subject,
                    age=float(recording.age),
                    start_s=float(start_s),
                    duration_s=config.window_duration_s,
                    recording_duration_s=recording.duration_s,
                    split=split_by_release[recording.release],
                )
            )
    return manifest


def preprocess_array(
    data: np.ndarray,
    *,
    source_frequency_hz: float,
    target_frequency_hz: float,
    clamp: float | None = REVE_MODEL.clamp,
) -> np.ndarray:
    """Apply the dependency-light part of the REVE preprocessing contract.

    The MNE-backed path below performs the official band-pass before calling
    this function.  Keeping resampling/scaling/clamping here makes those
    ordering rules directly testable without importing MNE.
    """

    # MNE stores raw EEG as float64 and the official StandardScaler is fitted
    # before the final float32 cache copy.  Preserve that precision here so the
    # independent path does not introduce avoidable preprocessing drift.
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("data must have shape (channels, time)")
    if source_frequency_hz <= 0 or target_frequency_hz <= 0:
        raise ValueError("sampling frequencies must be positive")

    if not np.isclose(source_frequency_hz, target_frequency_hz):
        try:
            from scipy.signal import resample_poly
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise IndependentPipelineError("scipy is required when source and target frequencies differ") from exc
        from fractions import Fraction

        ratio = Fraction(target_frequency_hz / source_frequency_hz).limit_denominator()
        values = resample_poly(values, ratio.numerator, ratio.denominator, axis=1)

    mean = values.mean(axis=1, keepdims=True)
    standard_deviation = values.std(axis=1, keepdims=True)
    standard_deviation = np.where(standard_deviation == 0, 1.0, standard_deviation)
    values = (values - mean) / standard_deviation
    if clamp is not None:
        values = np.clip(values, -clamp, clamp)
    return np.asarray(values, dtype=np.float32)


def _task_info_path(recording: HbnRecording) -> Path:
    """Return the HBN task JSON path used by the official study reader."""

    # .../<release>/download/<subject>/eeg/file.set -> .../<release>/download
    return recording.path.parents[2] / f"{recording.task}_eeg.json"


def _read_hbn_raw(recording: HbnRecording) -> Any:
    """Load HBN raw EEG and apply the same reference step as Shirazi2024Hbn."""

    try:
        import mne
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise IndependentPipelineError("MNE is required for the raw HBN reader; install the pipeline extra") from exc

    raw = mne.io.read_raw_eeglab(recording.path, preload=True, verbose="ERROR")
    info_path = _task_info_path(recording)
    if info_path.is_file():
        task_info = json.loads(info_path.read_text())
        reference = task_info.get("EEGReference")
        if reference in raw.ch_names:
            raw.set_eeg_reference(ref_channels=[reference], verbose="ERROR")
    return raw


def preprocess_hbn_recording(
    recording: HbnRecording,
    config: ReveModelConfig = REVE_MODEL,
) -> PreparedRecording:
    """Load and preprocess one HBN recording using the official REVE values."""

    try:
        import mne
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise IndependentPipelineError("MNE is required for HBN preprocessing; install the pipeline extra") from exc

    raw = _read_hbn_raw(recording)

    # NeuralSet's Eeg event has a 60-second timeline start but a zero file
    # offset.  Its MneRaw reader consequently loads the first 120 seconds;
    # the extractor then filters, resamples, and scales that chunk.
    source_frequency_hz = float(raw.info["sfreq"])
    crop_duration_s = AGE_TASK.max_crop_duration_s
    raw.crop(tmin=0.0, tmax=crop_duration_s - 1.0 / source_frequency_hz)
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        raise IndependentPipelineError(f"no EEG channels found in {recording.path}")
    raw.pick(eeg_picks)

    low, high = config.bandpass_hz
    raw.filter(low, high, n_jobs=config.mne_n_jobs, verbose="ERROR")
    raw.resample(config.frequency_hz, n_jobs=config.mne_n_jobs, verbose="ERROR")

    processed = preprocess_array(raw.get_data(), source_frequency_hz=config.frequency_hz, target_frequency_hz=config.frequency_hz, clamp=config.clamp)
    return PreparedRecording(processed, tuple(raw.ch_names))


class PreprocessedRecordingStore:
    """Small disk-backed cache for preprocessed continuous recordings."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_paths(self, recording: HbnRecording) -> tuple[Path, Path]:
        if self.cache_dir is None:
            raise RuntimeError("cache paths requested without cache_dir")
        fingerprint = hashlib.sha256(f"{PREPROCESSING_CACHE_VERSION}|{recording.path.resolve()}|{REVE_MODEL}".encode()).hexdigest()[:24]
        return (
            self.cache_dir / f"{fingerprint}.npy",
            self.cache_dir / f"{fingerprint}.json",
        )

    def load(self, recording: HbnRecording) -> PreparedRecording:
        if self.cache_dir is None:
            return preprocess_hbn_recording(recording)

        data_path, metadata_path = self._cache_paths(recording)
        if data_path.is_file() and metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text())
                return PreparedRecording(np.load(data_path, mmap_mode="r"), tuple(metadata["channel_names"]))
            except (OSError, TypeError, ValueError, KeyError):
                # A worker can be interrupted between the data and metadata
                # writes. Treat that pair as a cache miss and rebuild it.
                pass

        prepared = preprocess_hbn_recording(recording)
        temporary_data = data_path.with_suffix(".tmp.npy")
        np.save(temporary_data, prepared.data)
        temporary_data.replace(data_path)
        metadata_path.write_text(
            json.dumps(
                {
                    "cache_version": PREPROCESSING_CACHE_VERSION,
                    "channel_names": prepared.channel_names,
                    "recording_duration_s": float(recording.duration_s),
                }
            )
        )
        return prepared


def _manifest_fieldnames(path: Path) -> tuple[str, ...]:
    """Read only the header of a subject-selection CSV."""

    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle).fieldnames or ())


def _resolve_manifest_recording(
    relative_path: str, *, data_root: Path
) -> Path:
    """Resolve and validate a recording path supplied by a manifest."""

    root = data_root.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"manifest recording is outside data root: {relative_path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"manifest recording does not exist: {path}")
    if path.suffix != ".set":
        raise ValueError(f"manifest recording is not an EEGLAB .set file: {path}")
    return path


def _task_from_set_path(path: Path) -> str:
    """Extract the BIDS task label from an EEGLAB filename."""

    for part in path.stem.split("_"):
        if part.startswith("task-"):
            return part
    raise ValueError(f"cannot find a task label in recording filename: {path.name}")


def _cached_duration_seconds(
    recording: HbnRecording, cache_dir: Path | None
) -> float | None:
    """Infer duration from a preprocessed cache without opening the raw file."""

    if cache_dir is None:
        return None
    store = PreprocessedRecordingStore(cache_dir)
    data_path, metadata_path = store._cache_paths(recording)
    if not data_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
        duration = float(metadata["recording_duration_s"])
    except (OSError, TypeError, ValueError, KeyError):
        duration = None
    if duration is not None and np.isfinite(duration) and duration > 0:
        return duration
    try:
        data = np.load(data_path, mmap_mode="r")
    except (OSError, ValueError):
        return None
    if data.ndim != 2:
        return None
    return float(data.shape[1] / REVE_MODEL.frequency_hz)


def _raw_duration_seconds(path: Path) -> float:
    """Read raw duration only when a preprocessed cache is unavailable."""

    try:
        import mne
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise IndependentPipelineError("MNE is required to inspect HBN recordings; install the pipeline extra") from exc
    raw = mne.io.read_raw_eeglab(path, preload=False, verbose="ERROR")
    return float(raw.n_times / raw.info["sfreq"])


def load_recordings_from_subject_manifest(
    subjects_file: Path,
    *,
    data_root: Path,
    cache_dir: Path | None = None,
) -> list[HbnRecording]:
    """Build recordings from either supported subject-manifest format.

    The canonical Age manifest stores ``recording_relpath`` and duration.  The
    download-selection manifest used for the 500-subject run stores release,
    subject, age, and ``set_file`` instead.  The latter is intentionally
    resolved directly so a selected run does not scan every HBN recording.
    """

    fields = _manifest_fieldnames(subjects_file)
    if fields == MANIFEST_FIELDS:
        canonical_rows = read_manifest(subjects_file)
        specifications = [
            (
                row.release,
                row.subject,
                float(row.age),
                row.recording_relpath,
                float(row.duration_s),
            )
            for row in canonical_rows
        ]
    else:
        required = {"release", "subject", "age", "set_file"}
        if not required.issubset(fields):
            raise ValueError(f"unsupported subject manifest fields; expected either {MANIFEST_FIELDS} or fields containing {sorted(required)}")
        with subjects_file.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        specifications = []
        for row in rows:
            release = row["release"]
            subject = row["subject"]
            set_file = Path(row["set_file"])
            if set_file.parts and set_file.parts[0] == subject:
                relative_path = Path(release) / "download" / set_file
            else:
                relative_path = (
                    Path(release) / "download" / subject / "eeg" / set_file
                )
            specifications.append((release, subject, float(row["age"]), relative_path.as_posix(), None))

    recordings: list[HbnRecording] = []
    seen_paths: set[Path] = set()
    for release, subject, age, relative_path, duration in specifications:
        if release not in HBN_RELEASES:
            raise ValueError(f"unsupported HBN release in subject manifest: {release}")
        path = _resolve_manifest_recording(relative_path, data_root=data_root)
        if path in seen_paths:
            raise ValueError(f"subject manifest contains duplicate recording: {path}")
        seen_paths.add(path)
        if not np.isfinite(age):
            raise ValueError(f"subject manifest age is not finite for {subject}")

        task = _task_from_set_path(path)
        recording = HbnRecording(path=path, release=release, subject=subject, task=task, age=age, duration_s=0.0)
        if duration is None:
            duration = _cached_duration_seconds(recording, cache_dir)
        if duration is None:
            duration = _raw_duration_seconds(path)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"recording duration is invalid for {path}: {duration}")
        recordings.append(HbnRecording(path=path, release=release, subject=subject, task=task, age=age, duration_s=float(duration)))
    if not recordings:
        raise ValueError(f"subject manifest contains no recordings: {subjects_file}")
    return recordings


def discover_hbn_recordings(data_root: Path) -> list[HbnRecording]:
    """Discover HBN EEGLAB recordings and participant ages from local files."""

    try:
        import mne
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise IndependentPipelineError("MNE is required to inspect HBN recordings; install the pipeline extra") from exc

    recordings: list[HbnRecording] = []
    for release in HBN_RELEASES:
        release_root = data_root / release
        download_root = release_root / "download"
        if not download_root.is_dir():
            continue
        participant_path = download_root / "participants.tsv"
        ages: dict[str, float | None] = {}
        if participant_path.is_file():
            with participant_path.open(newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    participant = row.get("participant_id")
                    if not participant:
                        continue
                    raw_age = row.get("age", "")
                    try:
                        ages[participant] = float(raw_age)
                    except (TypeError, ValueError):
                        ages[participant] = None

        for path in sorted(download_root.glob("sub-*/eeg/*.set")):
            parts = path.stem.split("_")
            if len(parts) < 3:
                continue
            subject, task = parts[0], parts[1]
            raw = mne.io.read_raw_eeglab(path, preload=False, verbose="ERROR")
            duration_s = float(raw.n_times / raw.info["sfreq"])
            recordings.append(HbnRecording(path=path, release=release, subject=subject, task=task, age=ages.get(subject), duration_s=duration_s))
    if not recordings:
        raise FileNotFoundError(f"no HBN recordings found below {data_root}; expected R1..R11/download")
    return recordings


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise IndependentPipelineError("PyTorch is required for REVE training; install the pipeline extra") from exc
    return torch, nn


def _seed_data_loader_worker(worker_id: int) -> None:
    """Match NeuralBench's NumPy/Python worker seeding contract."""

    del worker_id
    torch, _ = _require_torch()
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    seed = int(worker_info.seed) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


try:  # Keep manifest-only imports usable without PyTorch installed.
    import torch as _torch
    from torch import nn as _nn
except Exception:  # pragma: no cover - depends on the active environment
    _torch = None
    _nn = None


if _nn is not None:

    class IndependentReveRegressor(_nn.Module):
        """REVE encoder plus the official mean aggregation and linear head."""

        def __init__(self, backbone: Any, freeze_backbone: bool = False):
            super().__init__()
            self.backbone = backbone
            if freeze_backbone:
                for parameter in self.backbone.parameters():
                    parameter.requires_grad = False
            self.head = _nn.LazyLinear(1)

        def forward(self, x: Any) -> Any:
            output = self.backbone(x)
            if isinstance(output, dict):
                raise IndependentPipelineError("REVE backbone returned a dict; expected encoder tensor")
            if output.ndim == 3:
                features = output.mean(dim=1)
            elif output.ndim == 2:
                features = output
            else:
                raise IndependentPipelineError(f"unexpected REVE encoder output shape: {tuple(output.shape)}")
            return self.head(features)

else:

    class IndependentReveRegressor:  # type: ignore[no-redef]
        """Placeholder that gives a clear error in manifest-only environments."""

        def __init__(self, *_args: Any, **_kwargs: Any):
            _require_torch()


class AgeWindowDataset:  # Defined without a torch base to keep imports lazy.
    """Lazy PyTorch dataset over the manifest and preprocessed-recording cache."""

    def __init__(
        self,
        windows: Sequence[AgeWindow],
        store: PreprocessedRecordingStore,
        channel_order: Sequence[str],
        aligned_recordings: Mapping[str, PreparedRecording] | None = None,
    ):
        self.windows = tuple(windows)
        self.store = store
        self.channel_order = tuple(channel_order)
        self._recordings = {
            str(window.path): HbnRecording(
                path=window.path,
                release=window.release,
                subject=window.subject,
                task=AGE_TASK.resting_state_task,
                age=window.age,
                duration_s=window.recording_duration_s,
            )
            for window in windows
        }
        self._aligned_recordings: dict[str, PreparedRecording] = dict(aligned_recordings or {})

    def __len__(self) -> int:
        return len(self.windows)

    def preload(self) -> None:
        """Load and align every recording used by this split exactly once.

        This method is intentionally called before DataLoader workers are
        created. On Linux, worker processes fork the dataset and can read the
        resulting arrays without re-reading the disk cache or repeating the
        channel alignment work.
        """

        for recording_key, recording in self._recordings.items():
            if recording_key not in self._aligned_recordings:
                self._aligned_recordings[recording_key] = align_prepared_recording(self.store.load(recording), self.channel_order)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        torch, _ = _require_torch()
        window = self.windows[index]
        recording_key = str(window.path)
        prepared = self._aligned_recordings.get(recording_key)
        if prepared is None:
            prepared = align_prepared_recording(self.store.load(self._recordings[recording_key]), self.channel_order)
            self._aligned_recordings[recording_key] = prepared
        start = int(round(window.start_s * REVE_MODEL.frequency_hz))
        stop = start + int(round(window.duration_s * REVE_MODEL.frequency_hz))
        values = np.asarray(prepared.data[:, start:stop], dtype=np.float32)
        expected_samples = int(round(window.duration_s * REVE_MODEL.frequency_hz))
        if values.shape[1] != expected_samples:
            raise IndependentPipelineError(f"window {window.path}:{window.start_s} has shape {values.shape}; expected {expected_samples} samples")
        values = np.clip(values, -REVE_MODEL.clamp, REVE_MODEL.clamp)
        return torch.from_numpy(values), torch.tensor([window.age], dtype=torch.float32)


def build_data_loaders(
    datasets: Mapping[str, Any],
    generators: Mapping[str, Any],
    config: IndependentTrainConfig,
) -> dict[str, Any]:
    """Build train/validation/test loaders with NeuralBench-style settings."""

    torch, _ = _require_torch()
    expected_splits = ("train", "val", "test")
    if any(split not in datasets for split in expected_splits):
        raise ValueError(f"datasets must contain {expected_splits}")
    if any(split not in generators for split in expected_splits):
        raise ValueError(f"generators must contain {expected_splits}")

    loaders: dict[str, Any] = {}
    for split in expected_splits:
        loader_kwargs: dict[str, Any] = {
            "batch_size": config.batch_size,
            "shuffle": split == "train",
            "num_workers": config.num_workers,
            "drop_last": False,
            "pin_memory": True,
            "generator": generators[split],
        }
        if config.num_workers > 0:
            loader_kwargs["persistent_workers"] = config.persistent_workers
            loader_kwargs["worker_init_fn"] = _seed_data_loader_worker
            if config.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = config.prefetch_factor
        loaders[split] = torch.utils.data.DataLoader(datasets[split], **loader_kwargs)
    return loaders


def load_reve_backbone(
    channel_names: Sequence[str],
    *,
    mapping_path: Path | None,
    pretrained_name: str = REVE_MODEL.pretrained_name,
    n_times: int = AGE_TASK.window_sample_count,
) -> Any:
    """Build the same pretrained encoder used by official NeuralBench."""

    _require_torch()
    try:
        from neuraltrain.models.reve import NtReve
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise ReveDependencyError("neuraltrain.NtReve is unavailable. Install the project's [pipeline] extra.") from exc

    mapping = load_reve_channel_mapping(mapping_path)
    model_config = NtReve(from_pretrained_name=pretrained_name, channel_mapping=mapping)
    return model_config.build(
        n_spatial_locations=len(channel_names),
        n_temporal_samples=n_times,
        n_outputs=None,
        chs_info=[{"ch_name": name} for name in channel_names],
        frequency=REVE_MODEL.frequency_hz,
    )


def load_reve_channel_mapping(mapping_path: Path | None) -> dict[str, str]:
    """Load the official ``reve.json`` without importing NeuralBench."""

    resolved = mapping_path
    if resolved is None:
        try:
            spec = importlib.util.find_spec("neuralbench")
        except (ImportError, ValueError):
            spec = None
        if spec is not None and spec.origin is not None:
            resolved = Path(spec.origin).parent / "models" / "channel_mappings" / "reve.json"
    if resolved is None or not resolved.is_file():
        raise FileNotFoundError("official reve.json was not found; pass --mapping pointing to neuralbench/models/channel_mappings/reve.json")
    mapping = json.loads(resolved.read_text())
    if not isinstance(mapping, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()):
        raise ValueError(f"invalid REVE channel mapping: {resolved}")
    return mapping


def _select_device(requested: str) -> str:
    torch, _ = _require_torch()
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, _ = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reset_training_rng(seed: int) -> None:
    """Reset RNGs at NeuralBench's model-to-training boundary."""

    _set_seed(seed)


def _score_or_nan(targets: np.ndarray, predictions: np.ndarray) -> float:
    try:
        return pearsonr(targets, predictions)
    except ValueError:
        return float("nan")


def _synchronize_device(device: str) -> None:
    """Synchronize CUDA only when timing or transferring GPU work."""

    torch, _ = _require_torch()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _evaluate(model: Any, loader: Any, device: str) -> tuple[float, float, np.ndarray, np.ndarray]:
    torch, _ = _require_torch()
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for values, age in loader:
            prediction = model(values.to(device, non_blocking=True))
            predictions.append(prediction.detach().cpu().numpy().reshape(-1))
            targets.append(age.detach().cpu().numpy().reshape(-1))
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    metrics = regression_metrics(y_true, y_pred)
    return metrics["pearsonr"], metrics["mse"], y_true, y_pred


def _materialize_model(model: Any, sample_values: Any, device: str) -> None:
    """Initialize lazy layers with the same eval-mode dummy forward as NeuralBench."""

    torch, _ = _require_torch()
    model.eval()
    with torch.no_grad():
        model(sample_values[:1].to(device, non_blocking=True))
    model.train()


def fit_independent_model(
    model: Any,
    train_loader: Any,
    validation_loader: Any,
    test_loader: Any,
    config: IndependentTrainConfig = IndependentTrainConfig(),
    *,
    model_initialized: bool = False,
    initialization_time_s: float = 0.0,
) -> TrainingResult:
    """Train with the official AdamW/OneCycle/early-stopping defaults."""

    torch, _ = _require_torch()
    if len(train_loader) == 0 or len(validation_loader) == 0 or len(test_loader) == 0:
        raise ValueError("train, validation, and test loaders must all be non-empty")
    device = _select_device(config.device)
    model.to(device)

    if model_initialized:
        # The official model factory materializes the probe before its final
        # seed reset.  The caller has already done that work in this mode.
        first_batch_s = initialization_time_s
    else:
        # Keep the standalone trainer useful for tests and small experiments.
        # This path initializes lazy layers immediately before the optimizer.
        first_batch_started = time.perf_counter()
        sample_values, _ = next(iter(train_loader))
        _synchronize_device(device)
        first_batch_s = time.perf_counter() - first_batch_started
        _materialize_model(model, sample_values, device)

    if config.validate_before_training:
        _synchronize_device(device)
        _evaluate(model, validation_loader, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )
    loss_fn = torch.nn.MSELoss()

    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    training_s = 0.0
    validation_s = 0.0
    train_batches = 0
    completed_epochs = 0

    for epoch in range(config.epochs):
        model.train()
        _synchronize_device(device)
        training_started = time.perf_counter()
        for values, age in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(values.to(device, non_blocking=True))
            loss = loss_fn(prediction, age.to(device, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            train_batches += 1
        _synchronize_device(device)
        training_s += time.perf_counter() - training_started

        _synchronize_device(device)
        validation_started = time.perf_counter()
        validation_score, _, _, _ = _evaluate(model, validation_loader, device)
        _synchronize_device(device)
        validation_s += time.perf_counter() - validation_started
        completed_epochs = epoch + 1
        comparable_score = validation_score if np.isfinite(validation_score) else -float("inf")
        if comparable_score > best_score:
            best_score = comparable_score
            best_epoch = epoch + 1
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise IndependentPipelineError("training never produced a valid validation score")
    model.load_state_dict(best_state)
    _synchronize_device(device)
    test_started = time.perf_counter()
    test_score, test_mse, targets, predictions = _evaluate(model, test_loader, device)
    _synchronize_device(device)
    test_s = time.perf_counter() - test_started
    test_metrics = regression_metrics(targets, predictions)
    timings = {
        "first_batch_s": float(first_batch_s),
        "train_s": float(training_s),
        "validation_s": float(validation_s),
        "test_s": float(test_s),
        "train_batches_per_s": float(train_batches / training_s)
        if training_s > 0
        else float("nan"),
        "completed_epochs": float(completed_epochs),
    }
    return TrainingResult(
        best_epoch=best_epoch,
        val_pearsonr=float(best_score),
        test_pearsonr=float(test_score),
        test_mse=test_mse,
        test_metrics=test_metrics,
        timings=timings,
        targets=targets,
        predictions=predictions,
    )


def compare_predictions(
    targets: Sequence[float],
    own_predictions: Sequence[float],
    official_predictions: Sequence[float],
    *,
    atol: float = 1e-6,
) -> PredictionComparison:
    """Compare scores after verifying both runners used identical rows."""

    y_true = np.asarray(targets, dtype=float).reshape(-1)
    own = np.asarray(own_predictions, dtype=float).reshape(-1)
    official = np.asarray(official_predictions, dtype=float).reshape(-1)
    if not (y_true.shape == own.shape == official.shape):
        raise ValueError("targets and both prediction arrays must have equal shapes")
    own_score = pearsonr(y_true, own)
    official_score = pearsonr(y_true, official)
    delta = own_score - official_score
    return PredictionComparison(
        own_pearsonr=own_score,
        official_pearsonr=official_score,
        pearsonr_delta=delta,
        matches=bool(np.isclose(own_score, official_score, atol=atol, rtol=0.0)),
        n_observations=int(y_true.size),
    )


def compare_prediction_rows(
    own_rows: Sequence[Mapping[str, Any]],
    official_rows: Sequence[Mapping[str, Any]],
    *,
    atol: float = 1e-6,
) -> PredictionComparison:
    """Align two per-window artifacts before comparing their predictions."""

    def key(row: Mapping[str, Any]) -> tuple[str, str, str, float, float]:
        return (
            str(row["path"]),
            str(row["release"]),
            str(row["subject"]),
            float(row["start_s"]),
            float(row["duration_s"]),
        )

    own_by_key = {key(row): row for row in own_rows}
    official_by_key = {key(row): row for row in official_rows}
    if len(own_by_key) != len(own_rows) or len(official_by_key) != len(official_rows):
        raise ValueError("prediction artifacts contain duplicate window identities")
    if set(own_by_key) != set(official_by_key):
        raise ValueError("prediction artifacts do not contain the same windows")

    ordered_keys = sorted(own_by_key)
    targets: list[float] = []
    own_predictions: list[float] = []
    official_predictions: list[float] = []
    for window_key in ordered_keys:
        own_row = own_by_key[window_key]
        official_row = official_by_key[window_key]
        own_age = float(own_row["age"])
        official_age = float(official_row["age"])
        if not np.isclose(own_age, official_age, atol=0.0, rtol=0.0):
            raise ValueError(f"target age differs for window {window_key}")
        targets.append(own_age)
        own_predictions.append(float(own_row["prediction"]))
        official_predictions.append(float(official_row["prediction"]))
    return compare_predictions(targets, own_predictions, official_predictions, atol=atol)


def compare_prediction_files(
    own_path: Path, official_path: Path, *, atol: float = 1e-6
) -> PredictionComparison:
    """Compare two CSV prediction artifacts after deterministic row alignment."""

    return compare_prediction_rows(read_prediction_rows(own_path), read_prediction_rows(official_path), atol=atol)


def compare_scores(
    own_pearsonr: float,
    official_pearsonr: float,
    *,
    atol: float = 1e-6,
) -> ScoreComparison:
    """Compare aggregate test scores from the two runners."""

    own = float(own_pearsonr)
    official = float(official_pearsonr)
    if not np.isfinite(own) or not np.isfinite(official):
        raise ValueError("both scores must be finite")
    delta = own - official
    return ScoreComparison(
        own_pearsonr=own,
        official_pearsonr=official,
        pearsonr_delta=delta,
        matches=bool(np.isclose(own, official, atol=atol, rtol=0.0)),
    )


def extract_official_pearson(results: Any) -> float:
    """Extract ``test/pearsonr`` from ``run_official_reve`` output."""

    candidates = results if isinstance(results, list) else [results]
    if len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise ValueError("expected one official NeuralBench result dictionary")
    value = candidates[0].get("test/pearsonr")
    if value is None:
        raise KeyError("official result does not contain 'test/pearsonr'")
    return float(value)


def run_independent_reve(
    data_root: Path,
    *,
    cache_dir: Path | None = None,
    mapping_path: Path | None = None,
    subjects_file: Path | None = None,
    train_config: IndependentTrainConfig = IndependentTrainConfig(),
    manifest_output: Path | None = None,
    predictions_output: Path | None = None,
    official_predictions_path: Path | None = None,
    score_atol: float = 1e-6,
) -> tuple[TrainingResult, dict[str, Any]]:
    """Run the complete independent HBN→REVE→Age experiment."""

    total_started = time.perf_counter()
    _set_seed(train_config.seed)
    selected_manifest = None
    if subjects_file is None:
        recordings = discover_hbn_recordings(data_root)
    else:
        if _manifest_fieldnames(subjects_file) == MANIFEST_FIELDS:
            selected_manifest = read_manifest(subjects_file)
        recordings = load_recordings_from_subject_manifest(subjects_file, data_root=data_root, cache_dir=cache_dir)
    if selected_manifest is not None:
        recordings = filter_recordings_by_manifest(recordings, selected_manifest, data_root=data_root)
    eligible_recordings = filter_age_recordings(recordings)
    manifest = build_age_window_manifest(recordings)
    if selected_manifest is not None:
        expected_splits = {
            row.recording_relpath: row.split for row in selected_manifest
        }
        for window in manifest:
            relative = window.path.resolve().relative_to(data_root.resolve()).as_posix()
            if expected_splits.get(relative) != window.split:
                raise IndependentPipelineError(f"manifest split mismatch for {relative}: {window.split!r} != {expected_splits.get(relative)!r}")
    store = PreprocessedRecordingStore(cache_dir)
    preload_started = time.perf_counter()
    prepared_by_path = {
        str(recording.path): store.load(recording)
        for recording in eligible_recordings
    }
    channel_order = build_global_channel_order(prepared_by_path.values())
    aligned_by_path = (
        {
            path: align_prepared_recording(prepared, channel_order)
            for path, prepared in prepared_by_path.items()
        }
        if train_config.preload_recordings
        else {}
    )
    preload_s = time.perf_counter() - preload_started

    torch, _ = _require_torch()
    train_state, _, val_state, test_state = np.random.SeedSequence(train_config.data_seed).generate_state(4)
    loader_generators = {
        "train": torch.Generator().manual_seed(int(train_state)),
        "val": torch.Generator().manual_seed(int(val_state)),
        "test": torch.Generator().manual_seed(int(test_state)),
    }
    datasets = {}
    for split in ("train", "val", "test"):
        split_windows = [window for window in manifest if window.split == split]
        split_paths = {str(window.path) for window in split_windows}
        datasets[split] = AgeWindowDataset(
            split_windows,
            store,
            channel_order,
            aligned_recordings={
                path: aligned_by_path[path]
                for path in split_paths
                if path in aligned_by_path
            },
        )
    loaders = build_data_loaders(datasets, loader_generators, train_config)

    # NeuralBench consumes one train batch while building the model.  The
    # probe is materialized on CPU, then the final seed reset happens after
    # every model-construction forward has completed.
    first_batch_started = time.perf_counter()
    sample_values, _ = next(iter(loaders["train"]))
    first_batch_s = time.perf_counter() - first_batch_started
    backbone = load_reve_backbone(channel_order, mapping_path=mapping_path, n_times=AGE_TASK.window_sample_count)
    model = IndependentReveRegressor(backbone, freeze_backbone=train_config.freeze_backbone)
    _materialize_model(model, sample_values, "cpu")

    # NeuralBench reseeds after constructing the model and before validation or
    # training.  Keeping this boundary means the training RNG stream is not
    # shifted by lazy-head initialization or model-construction forwards.
    _reset_training_rng(train_config.seed)

    result = fit_independent_model(
        model,
        loaders["train"],
        loaders["val"],
        loaders["test"],
        train_config,
        model_initialized=True,
        initialization_time_s=first_batch_s,
    )
    test_windows = [window for window in manifest if window.split == "test"]
    prediction_rows = build_prediction_rows(test_windows, result.predictions)
    if manifest_output is not None:
        write_manifest_rows(manifest, manifest_output)
    if predictions_output is not None:
        write_prediction_rows(prediction_rows, predictions_output)
    report = {
        "pipeline": "independent",
        "model": asdict(REVE_MODEL),
        "training": asdict(train_config),
        "recording_count": len(eligible_recordings),
        "window_count": len(manifest),
        "windows_by_split": {
            split: sum(window.split == split for window in manifest)
            for split in ("train", "val", "test")
        },
        "channel_count": len(channel_order),
        "channel_names": list(channel_order),
        "best_epoch": result.best_epoch,
        "val_pearsonr": result.val_pearsonr,
        "test_pearsonr": result.test_pearsonr,
        "test_mse": result.test_mse,
        "test_metrics": result.test_metrics,
    }
    if subjects_file is not None:
        report["manifest_path"] = str(subjects_file)
        report["manifest_sha256"] = manifest_sha256(subjects_file)
        report["manifest_rows"] = len(selected_manifest or recordings)
    if official_predictions_path is not None:
        comparison = compare_prediction_rows(prediction_rows, read_prediction_rows(official_predictions_path), atol=score_atol)
        report["prediction_comparison"] = asdict(comparison)
    report["timings"] = {
        "preload_s": float(preload_s),
        **result.timings,
        "total_s": float(time.perf_counter() - total_started),
    }
    return result, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--subjects-file", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument(
        "--no-preload",
        dest="preload_recordings",
        action="store_false",
        help="load and align recordings lazily in each DataLoader process",
    )
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="recreate worker processes for each DataLoader iterator",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=33)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--official-score", type=float)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--official-predictions", type=Path)
    parser.add_argument("--score-atol", type=float, default=1e-6)
    args = parser.parse_args(argv)

    config = IndependentTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        preload_recordings=args.preload_recordings,
        seed=args.seed,
        data_seed=args.data_seed,
        device=args.device,
    )
    result, report = run_independent_reve(
        args.data_root,
        cache_dir=args.cache_dir,
        mapping_path=args.mapping,
        subjects_file=args.subjects_file,
        train_config=config,
        manifest_output=args.manifest_output,
        predictions_output=args.predictions_output,
        official_predictions_path=args.official_predictions,
        score_atol=args.score_atol,
    )
    if args.official_score is not None:
        comparison = compare_scores(result.test_pearsonr, args.official_score, atol=args.score_atol)
        report["score_comparison"] = asdict(comparison)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
