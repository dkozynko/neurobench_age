"""Shared block-safe preprocessing for REVE-compatible resting EEG."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Iterable

import numpy as np

from ..research.protocol import PreprocessingContract


class PreprocessingError(ValueError):
    """Raised when a recording violates the shared preprocessing contract."""


def preprocess_rest_blocks(
    blocks: Iterable[np.ndarray],
    *,
    original_frequency_hz: float,
    channel_labels: tuple[str, ...],
    contract: PreprocessingContract,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the common REVE input contract without crossing block boundaries."""

    from scipy.signal import butter, resample_poly, sosfiltfilt

    source_frequency = float(original_frequency_hz)
    if not math.isfinite(source_frequency) or source_frequency <= 0:
        raise PreprocessingError("original frequency must be finite and positive")
    if not channel_labels or any(not label.strip() for label in channel_labels):
        raise PreprocessingError("channel labels must be non-empty")
    normalized_labels = [label.strip().casefold() for label in channel_labels]
    if len(set(normalized_labels)) != len(normalized_labels):
        raise PreprocessingError("duplicate channel labels are not allowed")

    arrays = [np.asarray(block, dtype=np.float64) for block in blocks]
    if not arrays:
        raise PreprocessingError("at least one resting block is required")
    for index, block in enumerate(arrays):
        if block.ndim != 2:
            raise PreprocessingError(f"block {index} must have channels x samples shape")
        if block.shape[0] != len(channel_labels):
            raise PreprocessingError(f"block {index} channel count does not match labels")
        if block.shape[1] < 1:
            raise PreprocessingError(f"block {index} has no samples")
        if not np.isfinite(block).all():
            raise PreprocessingError(f"block {index} contains non-finite samples")

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
            raise PreprocessingError(
                f"block {index} is too short for the declared band-pass filter"
            ) from error
        if contract.notch_hz:
            raise PreprocessingError(
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
        raise PreprocessingError("no usable samples remain after acquisition-order selection")

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
        raise PreprocessingError("no complete windows remain within individual blocks")
    stacked = np.stack(windows).astype(np.float32, copy=False)
    if not np.isfinite(stacked).all():
        raise PreprocessingError("preprocessing produced non-finite samples")
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
