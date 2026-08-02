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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from .reve_baseline import (
    AGE_TASK,
    REVE_MODEL,
    AgeTaskConfig,
    ReveDependencyError,
    build_window_starts,
    pearsonr,
)


HBN_RELEASES = tuple(f"R{i}" for i in range(1, 12))

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
    device: str = "auto"
    num_workers: int = 0
    freeze_backbone: bool = False


@dataclass(frozen=True)
class TrainingResult:
    """Scores and predictions produced by the independent trainer."""

    best_epoch: int
    val_pearsonr: float
    test_pearsonr: float
    test_mse: float
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

    eligible = filter_age_recordings(recordings, config)
    if not eligible:
        raise ValueError("the Age query returned no eligible recordings")

    split_by_release = assign_release_splits(
        (recording.release for recording in eligible), config
    )
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
            raise IndependentPipelineError(
                "scipy is required when source and target frequencies differ"
            ) from exc
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
        raise IndependentPipelineError(
            "MNE is required for the raw HBN reader; install the pipeline extra"
        ) from exc

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
        raise IndependentPipelineError(
            "MNE is required for HBN preprocessing; install the pipeline extra"
        ) from exc

    raw = _read_hbn_raw(recording)
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        raise IndependentPipelineError(f"no EEG channels found in {recording.path}")
    raw.pick(eeg_picks)

    low, high = config.bandpass_hz
    raw.filter(low, high, n_jobs=1, verbose="ERROR")
    raw.resample(config.frequency_hz, n_jobs=1, verbose="ERROR")

    processed = preprocess_array(
        raw.get_data(),
        source_frequency_hz=config.frequency_hz,
        target_frequency_hz=config.frequency_hz,
        clamp=None,
    )
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
        fingerprint = hashlib.sha256(
            f"{recording.path.resolve()}|{REVE_MODEL}".encode()
        ).hexdigest()[:24]
        return (
            self.cache_dir / f"{fingerprint}.npy",
            self.cache_dir / f"{fingerprint}.json",
        )

    def load(self, recording: HbnRecording) -> PreparedRecording:
        if self.cache_dir is None:
            return preprocess_hbn_recording(recording)

        data_path, metadata_path = self._cache_paths(recording)
        if data_path.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            return PreparedRecording(
                np.load(data_path, mmap_mode="r"), tuple(metadata["channel_names"])
            )

        prepared = preprocess_hbn_recording(recording)
        temporary_data = data_path.with_suffix(".tmp.npy")
        np.save(temporary_data, prepared.data)
        temporary_data.replace(data_path)
        metadata_path.write_text(json.dumps({"channel_names": prepared.channel_names}))
        return prepared


def discover_hbn_recordings(data_root: Path) -> list[HbnRecording]:
    """Discover HBN EEGLAB recordings and participant ages from local files."""

    try:
        import mne
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise IndependentPipelineError(
            "MNE is required to inspect HBN recordings; install the pipeline extra"
        ) from exc

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
            recordings.append(
                HbnRecording(
                    path=path,
                    release=release,
                    subject=subject,
                    task=task,
                    age=ages.get(subject),
                    duration_s=duration_s,
                )
            )
    if not recordings:
        raise FileNotFoundError(
            f"no HBN recordings found below {data_root}; expected R1..R11/download"
        )
    return recordings


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise IndependentPipelineError(
            "PyTorch is required for REVE training; install the pipeline extra"
        ) from exc
    return torch, nn


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
                raise IndependentPipelineError(
                    "REVE backbone returned a dict; expected encoder tensor"
                )
            if output.ndim == 3:
                features = output.mean(dim=1)
            elif output.ndim == 2:
                features = output
            else:
                raise IndependentPipelineError(
                    f"unexpected REVE encoder output shape: {tuple(output.shape)}"
                )
            return self.head(features)

else:

    class IndependentReveRegressor:  # type: ignore[no-redef]
        """Placeholder that gives a clear error in manifest-only environments."""

        def __init__(self, *_args: Any, **_kwargs: Any):
            _require_torch()


class AgeWindowDataset:  # Defined without a torch base to keep imports lazy.
    """Lazy PyTorch dataset over the manifest and preprocessed-recording cache."""

    def __init__(self, windows: Sequence[AgeWindow], store: PreprocessedRecordingStore):
        self.windows = tuple(windows)
        self.store = store
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

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        torch, _ = _require_torch()
        window = self.windows[index]
        prepared = self.store.load(self._recordings[str(window.path)])
        start = int(round(window.start_s * REVE_MODEL.frequency_hz))
        stop = start + int(round(window.duration_s * REVE_MODEL.frequency_hz))
        values = np.asarray(prepared.data[:, start:stop], dtype=np.float32)
        expected_samples = int(round(window.duration_s * REVE_MODEL.frequency_hz))
        if values.shape[1] != expected_samples:
            raise IndependentPipelineError(
                f"window {window.path}:{window.start_s} has shape {values.shape}; "
                f"expected {expected_samples} samples"
            )
        values = np.clip(values, -REVE_MODEL.clamp, REVE_MODEL.clamp)
        return torch.from_numpy(values), torch.tensor([window.age], dtype=torch.float32)


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
        raise ReveDependencyError(
            "neuraltrain.NtReve is unavailable. Install the project's [pipeline] extra."
        ) from exc

    mapping = load_reve_channel_mapping(mapping_path)
    model_config = NtReve(
        from_pretrained_name=pretrained_name,
        channel_mapping=mapping,
    )
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
        raise FileNotFoundError(
            "official reve.json was not found; pass --mapping pointing to "
            "neuralbench/models/channel_mappings/reve.json"
        )
    mapping = json.loads(resolved.read_text())
    if not isinstance(mapping, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in mapping.items()
    ):
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


def _score_or_nan(targets: np.ndarray, predictions: np.ndarray) -> float:
    try:
        return pearsonr(targets, predictions)
    except ValueError:
        return float("nan")


def _evaluate(model: Any, loader: Any, device: str) -> tuple[float, float, np.ndarray, np.ndarray]:
    torch, _ = _require_torch()
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for values, age in loader:
            prediction = model(values.to(device))
            predictions.append(prediction.detach().cpu().numpy().reshape(-1))
            targets.append(age.detach().cpu().numpy().reshape(-1))
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    score = _score_or_nan(y_true, y_pred)
    mse = float(np.mean((y_true - y_pred) ** 2))
    return score, mse, y_true, y_pred


def fit_independent_model(
    model: Any,
    train_loader: Any,
    validation_loader: Any,
    test_loader: Any,
    config: IndependentTrainConfig = IndependentTrainConfig(),
) -> TrainingResult:
    """Train with the official AdamW/OneCycle/early-stopping defaults."""

    torch, _ = _require_torch()
    if len(train_loader) == 0 or len(validation_loader) == 0 or len(test_loader) == 0:
        raise ValueError("train, validation, and test loaders must all be non-empty")
    device = _select_device(config.device)
    model.to(device)

    # Initialize LazyLinear before constructing AdamW.
    sample_values, _ = next(iter(train_loader))
    with torch.no_grad():
        model(sample_values[:1].to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
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

    for epoch in range(config.epochs):
        model.train()
        for values, age in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(values.to(device))
            loss = loss_fn(prediction, age.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()

        validation_score, _, _, _ = _evaluate(model, validation_loader, device)
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
    test_score, test_mse, targets, predictions = _evaluate(model, test_loader, device)
    return TrainingResult(
        best_epoch=best_epoch,
        val_pearsonr=float(best_score),
        test_pearsonr=float(test_score),
        test_mse=test_mse,
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
    train_config: IndependentTrainConfig = IndependentTrainConfig(),
) -> tuple[TrainingResult, dict[str, Any]]:
    """Run the complete independent HBN→REVE→Age experiment."""

    _set_seed(train_config.seed)
    recordings = discover_hbn_recordings(data_root)
    manifest = build_age_window_manifest(recordings)
    store = PreprocessedRecordingStore(cache_dir)
    first_recording = filter_age_recordings(recordings)[0]
    first_prepared = store.load(first_recording)

    backbone = load_reve_backbone(
        first_prepared.channel_names,
        mapping_path=mapping_path,
        n_times=AGE_TASK.window_sample_count,
    )
    model = IndependentReveRegressor(
        backbone, freeze_backbone=train_config.freeze_backbone
    )

    torch, _ = _require_torch()
    datasets = {
        split: AgeWindowDataset(
            [window for window in manifest if window.split == split], store
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=train_config.num_workers,
        ),
        "val": torch.utils.data.DataLoader(
            datasets["val"],
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
        ),
        "test": torch.utils.data.DataLoader(
            datasets["test"],
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
        ),
    }
    result = fit_independent_model(
        model,
        loaders["train"],
        loaders["val"],
        loaders["test"],
        train_config,
    )
    report = {
        "pipeline": "independent",
        "model": asdict(REVE_MODEL),
        "training": asdict(train_config),
        "recording_count": len(filter_age_recordings(recordings)),
        "window_count": len(manifest),
        "windows_by_split": {
            split: sum(window.split == split for window in manifest)
            for split in ("train", "val", "test")
        },
        "channel_count": len(first_prepared.channel_names),
        "channel_names": list(first_prepared.channel_names),
        "best_epoch": result.best_epoch,
        "val_pearsonr": result.val_pearsonr,
        "test_pearsonr": result.test_pearsonr,
        "test_mse": result.test_mse,
    }
    return result, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--official-score", type=float)
    parser.add_argument("--score-atol", type=float, default=1e-6)
    args = parser.parse_args(argv)

    config = IndependentTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    result, report = run_independent_reve(
        args.data_root,
        cache_dir=args.cache_dir,
        mapping_path=args.mapping,
        train_config=config,
    )
    if args.official_score is not None:
        comparison = compare_scores(
            result.test_pearsonr,
            args.official_score,
            atol=args.score_atol,
        )
        report["score_comparison"] = asdict(comparison)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
