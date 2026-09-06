from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from neurobench_age.data.mipdb import (
    MipdbPreprocessingError,
    load_mipdb_resting_subject,
    parse_mipdb_rest_segments,
    preprocess_rest_blocks,
)
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


def _events(path: Path, rows: list[tuple[float, str]]) -> None:
    path.write_text(
        "onset\tduration\ttrial_type\n"
        + "".join(f"{onset}\t0\t{marker}\n" for onset, marker in rows),
        encoding="utf-8",
    )


def test_rest_segments_discard_precondition_and_preserve_eo_ec_boundaries(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events.tsv"
    _events(
        events,
        [(3.75, "90"), (113.5, "20"), (133.5, "30"), (173.5, "20")],
    )

    segments = parse_mipdb_rest_segments(
        events, recording_duration_s=200.0, contract=PREPROCESSING
    )

    assert [
        (segment.condition, segment.start_s, segment.stop_s)
        for segment in segments
    ] == [
        ("eyes_open", 113.5, 133.5),
        ("eyes_closed", 133.5, 173.5),
        ("eyes_open", 173.5, 200.0),
    ]


def test_rest_segments_use_last_start_after_an_incomplete_attempt(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events.tsv"
    _events(
        events,
        [
            (1.0, "90"),
            (2.0, "20"),
            (3.0, "30"),
            (5.0, "90"),
            (6.0, "20"),
            (7.0, "30"),
            (8.0, "20"),
        ],
    )

    segments = parse_mipdb_rest_segments(
        events, recording_duration_s=10.0, contract=PREPROCESSING
    )

    assert [(segment.condition, segment.start_s) for segment in segments] == [
        ("eyes_open", 6.0),
        ("eyes_closed", 7.0),
        ("eyes_open", 8.0),
    ]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([(1.0, "20"), (3.0, "30")], "paradigm-start"),
        ([(1.0, "90"), (2.0, "20"), (3.0, "20")], "alternate"),
        ([(1.0, "90"), (3.0, "30"), (2.0, "20")], "increasing"),
        ([(1.0, "90"), (2.0, "20")], "both eyes-open and eyes-closed"),
    ],
)
def test_rest_segments_reject_ambiguous_marker_streams(
    tmp_path: Path, rows: list[tuple[float, str]], message: str
) -> None:
    events = tmp_path / "events.tsv"
    _events(events, rows)

    with pytest.raises(MipdbPreprocessingError, match=message):
        parse_mipdb_rest_segments(
            events, recording_duration_s=10.0, contract=PREPROCESSING
        )


class _FakeRaw:
    def __init__(self, data: np.ndarray, *, sfreq: float, bads: list[str] | None = None):
        self._data = data
        self.ch_names = [f"E{index}" for index in range(1, 129)]
        self.info = {"sfreq": sfreq, "bads": list(bads or [])}
        self.n_times = data.shape[1]

    def get_channel_types(self) -> list[str]:
        return ["eeg"] * len(self.ch_names)

    def get_data(self) -> np.ndarray:
        return self._data


def test_subject_loader_uses_only_declared_block01_conditions(tmp_path: Path) -> None:
    subject_id = "sub-A00000001"
    eeg = tmp_path / subject_id / "eeg"
    eeg.mkdir(parents=True)
    recording = eeg / f"{subject_id}_task-block01_eeg.vhdr"
    recording.write_text("header\n")
    _events(
        eeg / f"{subject_id}_task-block01_events.tsv",
        [(0.5, "90"), (1.0, "20"), (3.0, "30"), (5.0, "20")],
    )
    rng = np.random.default_rng(10)
    raw = _FakeRaw(rng.normal(size=(128, 1000)), sfreq=100.0)
    contract = replace(PREPROCESSING, max_seconds_per_subject=4.0)

    windows, qc = load_mipdb_resting_subject(
        tmp_path,
        {
            "subject_id": subject_id,
            "age": 12.0,
            "recordings": [str(recording.relative_to(tmp_path))],
        },
        contract=contract,
        raw_loader=lambda path: raw,
    )

    assert windows.shape == (2, 128, 400)
    assert qc["resting_task"] == "block01"
    assert qc["condition_sequence"] == ["eyes_open", "eyes_closed", "eyes_open"]
    assert qc["mapped_channel_count"] == 128
    assert qc["selected_duration_seconds"] == 4.0
    assert "age" not in qc and "prediction" not in qc and "metrics" not in qc


def test_subject_loader_rejects_bad_or_missing_egi_channels(tmp_path: Path) -> None:
    subject_id = "sub-A00000001"
    eeg = tmp_path / subject_id / "eeg"
    eeg.mkdir(parents=True)
    recording = eeg / f"{subject_id}_task-block01_eeg.vhdr"
    recording.write_text("header\n")
    _events(
        eeg / f"{subject_id}_task-block01_events.tsv",
        [(0.5, "90"), (1.0, "20"), (4.0, "30")],
    )
    data = np.ones((128, 1000))

    with pytest.raises(MipdbPreprocessingError, match="bad mapped channels"):
        load_mipdb_resting_subject(
            tmp_path,
            {"subject_id": subject_id, "age": 12.0, "recordings": [str(recording.relative_to(tmp_path))]},
            contract=PREPROCESSING,
            raw_loader=lambda path: _FakeRaw(data, sfreq=100.0, bads=["E7"]),
        )
