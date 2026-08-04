#!/usr/bin/env python3
"""Dry-run harness for the official NeuralBench REVE Age baseline.

This module deliberately does not download HBN data or instantiate the large
REVE checkpoint.  It captures the benchmark contract in executable Python so
that data and training can be added after the local pipeline is verified.

Run from the repository root:

    python -m neurobench_age.reve_baseline --dry-run

The production training commands are exposed by :func:`build_official_commands`
but are not executed by this module.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


REVE_FREQUENCY_HZ = 200.0


@dataclass(frozen=True)
class AgeTaskConfig:
    """The data and evaluation contract from NeuralBench's Age config."""

    source_name: str = "Shirazi2024Hbn"
    recording_event_type: str = "Eeg"
    resting_state_task: str = "task-RestingState"
    minimum_recording_duration_s: float = 180.0
    crop_start_s: float = 60.0
    max_crop_duration_s: float = 120.0
    window_duration_s: float = 2.0
    window_stride_s: float = 2.0
    raw_channel_count: int = 129
    test_release: str = "R5"
    # NeuralBench's released config uses the decimal ratio, rather than the
    # mathematically similar-looking 1/11 approximation.
    validation_release_ratio: float = 0.091
    validation_random_state: int = 33
    target_field: str = "age"
    target_aggregation: str = "single"
    loss: str = "MSELoss"
    monitored_metric: str = "val/pearsonr"
    metric_mode: str = "max"

    @property
    def window_sample_count(self) -> int:
        """Samples per window after the REVE preprocessing resample."""

        return int(self.window_duration_s * REVE_FREQUENCY_HZ)


@dataclass(frozen=True)
class ReveModelConfig:
    """The standardized REVE downstream configuration."""

    pretrained_name: str = "brain-bzh/reve-base"
    frequency_hz: float = REVE_FREQUENCY_HZ
    # NeuralSet's official EegExtractor defaults to all available CPUs.
    mne_n_jobs: int = -1
    bandpass_hz: tuple[float, float] = (0.5, 99.5)
    notch_hz: tuple[float, ...] | None = None
    scaler: str = "StandardScaler"
    clamp: float = 15.0
    channel_mapping: str = "channel_mappings/reve.json"
    mapped_channel_count: int = 128
    aggregation: str = "mean"
    probe: str = "linear"
    output_size: int = 1


AGE_TASK = AgeTaskConfig()
REVE_MODEL = ReveModelConfig()


class ReveDependencyError(RuntimeError):
    """Raised when the official NeuralBench REVE runner is unavailable."""


def build_window_starts(
    recording_duration_s: float,
    config: AgeTaskConfig = AGE_TASK,
) -> np.ndarray:
    """Return non-overlapping window starts for one eligible recording.

    NeuralBench's ``CropTimelines`` transform moves the EEG event's timeline
    start to 60 seconds, but leaves its file offset at zero.  ``MneRaw.read``
    therefore caches the first 120 seconds of the file, and the segmenter
    addresses that cached array from zero.  This reproduces the observed
    official cache semantics rather than applying the transform's start as a
    second file offset.
    """

    if not np.isfinite(recording_duration_s):
        raise ValueError("recording_duration_s must be finite")
    if recording_duration_s <= config.minimum_recording_duration_s:
        raise ValueError(
            "recording must be longer than the Age task's 180-second minimum"
        )
    if config.window_duration_s <= 0 or config.window_stride_s <= 0:
        raise ValueError("window duration and stride must be positive")
    if config.max_crop_duration_s < config.window_duration_s:
        raise ValueError("crop duration must contain at least one full window")

    usable_duration_s = min(config.max_crop_duration_s, recording_duration_s)
    window_count = int(
        np.floor(
            (usable_duration_s - config.window_duration_s)
            / config.window_stride_s
        )
    ) + 1
    if window_count <= 0:
        raise ValueError("recording does not contain a complete usable window")

    return np.arange(window_count) * config.window_stride_s


def validate_split_assignments(
    assignments: Mapping[str, str],
    config: AgeTaskConfig = AGE_TASK,
) -> None:
    """Validate the release-level split invariant required by the benchmark."""

    if not assignments:
        raise ValueError("release assignments must not be empty")
    allowed_splits = {"train", "valid", "test"}
    invalid_splits = set(assignments.values()) - allowed_splits
    if invalid_splits:
        raise ValueError(f"unsupported split labels: {sorted(invalid_splits)}")
    if config.test_release not in assignments:
        raise ValueError(f"test release {config.test_release!r} is missing")
    if assignments[config.test_release] != "test":
        raise ValueError(
            f"test release {config.test_release!r} must be assigned to test"
        )
    if "train" not in assignments.values() or "valid" not in assignments.values():
        raise ValueError("both train and valid releases are required")
    if any(
        split == "test" and release != config.test_release
        for release, split in assignments.items()
    ):
        raise ValueError("only the configured R5 release may be assigned to test")


def pearsonr(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute Pearson correlation without adding a SciPy dependency."""

    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true.size < 2:
        raise ValueError("Pearson correlation requires at least two observations")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Pearson correlation inputs must be finite")

    true_centered = true - true.mean()
    pred_centered = pred - pred.mean()
    denominator = np.linalg.norm(true_centered) * np.linalg.norm(pred_centered)
    if denominator == 0:
        raise ValueError("Pearson correlation is undefined for constant inputs")
    return float(np.dot(true_centered, pred_centered) / denominator)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def dependency_status() -> dict[str, bool]:
    """Report whether the optional official REVE stack is installed."""

    return {
        name: _module_available(name)
        for name in (
            "neuralbench",
            "neuraltrain",
            "lightning",
            "huggingface_hub",
            "safetensors",
        )
    }


def build_reve_run_request(
    *,
    debug: bool = False,
    download: bool = False,
    prepare: bool = False,
    force: bool = False,
    retry: bool = False,
    checkpoint: str | None = None,
) -> dict[str, object]:
    """Build kwargs for NeuralBench's official programmatic API.

    The request mirrors ``neuralbench eeg age -m reve``.  Data-affecting
    actions are opt-in so calling the project code cannot download HBN by
    accident.
    """

    if download and prepare:
        raise ValueError("download and prepare cannot be requested together")
    if force and retry:
        raise ValueError("force and retry cannot be requested together")

    return {
        "device": "eeg",
        "task": "age",
        "model": "reve",
        "debug": debug,
        "download": download,
        "prepare": prepare,
        "force": force,
        "retry": retry,
        "checkpoint": checkpoint,
    }


def run_official_reve(
    *,
    debug: bool = False,
    download: bool = False,
    prepare: bool = False,
    force: bool = False,
    retry: bool = False,
    checkpoint: str | None = None,
    benchmark_runner: Callable[..., object] | None = None,
) -> object:
    """Run the official REVE Age experiment through NeuralBench.

    ``benchmark_runner`` is injectable for tests.  In normal use this imports
    NeuralBench's public ``run_benchmark`` function, which resolves the
    upstream ``NtReve`` model, official channel mapping, pretrained checkpoint,
    preprocessing, and linear probe from NeuralBench's YAML configuration.
    """

    if benchmark_runner is None:
        try:
            from neuralbench import run_benchmark
        except Exception as exc:
            raise ReveDependencyError(
                "The official REVE runner could not be imported. Install the "
                "project's [reve] extra and check the dependency error: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        benchmark_runner = run_benchmark

    request = build_reve_run_request(
        debug=debug,
        download=download,
        prepare=prepare,
        force=force,
        retry=retry,
        checkpoint=checkpoint,
    )
    return benchmark_runner(**request)


def build_official_commands() -> dict[str, str]:
    """Return the commands for the later download/prepare/train milestone."""

    return {
        "download": "neuralbench eeg age --download",
        "prepare": "neuralbench eeg age --prepare",
        "train": "neuralbench eeg age -m reve",
        "dry_run": "python -m neurobench_age.reve_baseline --dry-run",
        "official_run": "python -m neurobench_age.reve_baseline --official-run",
        "debug_run": "python -m neurobench_age.reve_baseline --official-run --debug",
    }


def dry_run(
    batch_size: int = 2,
    recording_duration_s: float = 181.0,
    seed: int = 33,
) -> dict[str, object]:
    """Validate the baseline contract with synthetic tensors only.

    The generated array represents raw HBN windows.  The report intentionally
    labels model execution as ``contract-only``: it checks the input/output
    interface without claiming that the REVE checkpoint has been run.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    starts = build_window_starts(recording_duration_s, AGE_TASK)
    rng = np.random.default_rng(seed)
    raw_input = rng.standard_normal(
        (batch_size, AGE_TASK.raw_channel_count, AGE_TASK.window_sample_count)
    )
    contract_predictions = raw_input.mean(axis=(1, 2), keepdims=False)[:, None]

    validate_split_assignments(
        {"R1": "train", "R2": "valid", AGE_TASK.test_release: "test"},
        AGE_TASK,
    )
    dependency_state = dependency_status()
    return {
        "status": "ok",
        "model_execution": "contract-only",
        "dependency_status": dependency_state,
        "neuralbench_installed": dependency_state["neuralbench"],
        "raw_input_shape": list(raw_input.shape),
        "reve_mapped_channel_count": REVE_MODEL.mapped_channel_count,
        "prediction_shape": list(contract_predictions.shape),
        "window_count_per_recording": int(starts.size),
        "window_start_range_s": [float(starts[0]), float(starts[-1])],
        "split_contract_valid": True,
        "config": {
            "age_task": asdict(AGE_TASK),
            "reve_model": asdict(REVE_MODEL),
        },
        "official_commands": build_official_commands(),
        "official_api_request": build_reve_run_request(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the synthetic input/output contract without downloading data",
    )
    mode.add_argument(
        "--official-run",
        action="store_true",
        help="delegate to NeuralBench's official REVE Age runner",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="use NeuralBench's reduced local debug mode for an official run",
    )
    parser.add_argument("--download", action="store_true", help="download the Age study")
    parser.add_argument("--prepare", action="store_true", help="prepare the Age study")
    parser.add_argument("--force", action="store_true", help="force an official rerun")
    parser.add_argument("--retry", action="store_true", help="retry failed official runs")
    parser.add_argument("--checkpoint", help="optional NeuralBench checkpoint path")
    args = parser.parse_args(argv)
    if args.dry_run:
        run_only_options = {
            "--debug": args.debug,
            "--download": args.download,
            "--prepare": args.prepare,
            "--force": args.force,
            "--retry": args.retry,
            "--checkpoint": args.checkpoint is not None,
        }
        selected_options = [
            option for option, selected in run_only_options.items() if selected
        ]
        if selected_options:
            parser.error(
                "the following options require --official-run: "
                + ", ".join(selected_options)
            )
        print(json.dumps(dry_run(), indent=2))
        return 0
    if args.official_run:
        if args.download and args.prepare:
            parser.error("--download and --prepare cannot be used together")
        if args.force and args.retry:
            parser.error("--force and --retry cannot be used together")
        try:
            result = run_official_reve(
                debug=args.debug,
                download=args.download,
                prepare=args.prepare,
                force=args.force,
                retry=args.retry,
                checkpoint=args.checkpoint,
            )
        except ReveDependencyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, default=str))
        return 0
    parser.error("choose --dry-run or --official-run")


if __name__ == "__main__":
    raise SystemExit(main())
