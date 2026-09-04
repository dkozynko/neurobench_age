from __future__ import annotations

import numpy as np
import pytest

from neurobench_age.data.mipdb import MipdbPreprocessingError, preprocess_rest_blocks
from neurobench_age.research.protocol import load_study_protocol
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING = load_study_protocol(
    ROOT / "configs" / "research" / "external_frozen_probe.json"
).preprocessing


def test_preprocessing_resamples_windows_and_records_qc() -> None:
    rng = np.random.default_rng(7)
    block = rng.normal(size=(3, 200)).astype(np.float64)

    windows, qc = preprocess_rest_blocks(
        [block],
        original_frequency_hz=100.0,
        channel_labels=("Cz", "Fz", "Pz"),
        contract=PREPROCESSING,
    )

    assert windows.shape == (1, 3, 400)
    assert windows.dtype == np.float32
    assert np.isfinite(windows).all()
    assert np.abs(windows).max() <= 15.0
    assert qc["original_frequency_hz"] == 100.0
    assert qc["output_frequency_hz"] == 200.0
    assert qc["window_count"] == 1
    assert qc["spatial_interpolation"] is False


def test_preprocessing_never_builds_a_window_across_blocks() -> None:
    rng = np.random.default_rng(8)
    blocks = [rng.normal(size=(2, 300)), rng.normal(size=(2, 300))]

    with pytest.raises(MipdbPreprocessingError, match="no complete windows"):
        preprocess_rest_blocks(
            blocks,
            original_frequency_hz=200.0,
            channel_labels=("Cz", "Fz"),
            contract=PREPROCESSING,
        )


def test_preprocessing_caps_acquisition_order_at_120_seconds() -> None:
    rng = np.random.default_rng(9)
    blocks = [
        rng.normal(size=(2, 80 * 200)),
        rng.normal(size=(2, 80 * 200)),
    ]

    windows, qc = preprocess_rest_blocks(
        blocks,
        original_frequency_hz=200.0,
        channel_labels=("Cz", "Fz"),
        contract=PREPROCESSING,
    )

    assert windows.shape == (60, 2, 400)
    assert qc["selected_duration_seconds"] == 120.0
    assert qc["included_blocks"] == [
        {"block_index": 0, "selected_samples": 16000},
        {"block_index": 1, "selected_samples": 8000},
    ]


@pytest.mark.parametrize(
    ("blocks", "labels", "message"),
    [
        ([np.ones((2, 400))], ("Cz", "Cz"), "duplicate channel"),
        ([np.ones((2, 400))], ("Cz",), "channel count"),
        ([np.array([[np.nan] * 400, [1.0] * 400])], ("Cz", "Fz"), "non-finite"),
    ],
)
def test_preprocessing_rejects_invalid_signal_contract(
    blocks: list[np.ndarray], labels: tuple[str, ...], message: str
) -> None:
    with pytest.raises(MipdbPreprocessingError, match=message):
        preprocess_rest_blocks(
            blocks,
            original_frequency_hz=200.0,
            channel_labels=labels,
            contract=PREPROCESSING,
        )
