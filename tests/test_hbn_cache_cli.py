from __future__ import annotations

import csv
import importlib.util
import json
import sys
from types import ModuleType
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from neurobench_age.pipelines.frozen_probe import (
    RepresentationCacheIdentity,
    load_cached_representations,
)
from neurobench_age.pipelines.independent import (
    HbnRecording,
    PreparedRecording,
    PreprocessedRecordingStore,
)
import neurobench_age.pipelines.independent as independent
from neurobench_age.pipelines.representation_materialization import (
    materialize_hbn_representations,
)
from neurobench_age.research.protocol import load_study_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/research/external_frozen_probe.json"
PROTOCOL = load_study_protocol(PROTOCOL_PATH)


def test_independent_eeglab_reader_simplifies_nested_matlab_cells(monkeypatch) -> None:
    """The HBN reader must avoid MNE's ambiguous ndarray truth-value path."""

    fake_mne = ModuleType("mne")
    fake_mne_io = ModuleType("mne.io")
    fake_mne_eeglab = ModuleType("mne.io.eeglab")
    fake_eeglab = ModuleType("mne.io.eeglab.eeglab")
    fake_scipy = ModuleType("scipy")
    fake_scipy_io = ModuleType("scipy.io")
    original_readmat = object()
    calls: dict[str, object] = {}

    def loadmat(*args: object, **kwargs: object) -> object:
        calls["args"] = args
        calls["kwargs"] = kwargs
        return {"EEG": "simplified"}

    fake_eeglab._readmat = original_readmat
    fake_scipy_io.loadmat = loadmat
    fake_scipy.io = fake_scipy_io
    fake_mne_eeglab.eeglab = fake_eeglab
    fake_mne_io.eeglab = fake_mne_eeglab
    fake_mne.io = fake_mne_io
    for name, module in {
        "mne": fake_mne,
        "mne.io": fake_mne_io,
        "mne.io.eeglab": fake_mne_eeglab,
        "mne.io.eeglab.eeglab": fake_eeglab,
        "scipy": fake_scipy,
        "scipy.io": fake_scipy_io,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    with independent._simplified_eeglab_mat_reader():
        assert fake_eeglab._readmat(
            "record.set",
            uint16_codec="latin1",
            preload=True,
        ) == {"EEG": "simplified"}

    assert calls == {
        "args": ("record.set",),
        "kwargs": {
            "struct_as_record": False,
            "squeeze_me": True,
            "simplify_cells": True,
            "uint16_codec": "latin1",
        },
    }
    assert fake_eeglab._readmat is original_readmat


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    def forward(self, value: torch.Tensor, *, return_output: bool = False):
        value = value[:, :2, :2].reshape(value.shape[0], 1, 4)
        outputs = []
        for layer in self.layers:
            value = torch.tanh(layer(value))
            outputs.append(value)
        return outputs if return_output else outputs[-1]


def _canonical_manifest(path: Path, data_root: Path) -> None:
    rows = (
        ("R1", "sub-train", "10", "R1/download/sub-train/eeg/sub-train_task-RestingState_eeg.set", "125", "train"),
        ("R8", "sub-val", "14", "R8/download/sub-val/eeg/sub-val_task-RestingState_eeg.set", "125", "val"),
        ("R5", "sub-sealed", "12", "R5/download/sub-sealed/eeg/sub-sealed_task-RestingState_eeg.set", "125", "test"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("release", "subject", "age", "recording_relpath", "duration_s", "split"))
        writer.writerows(rows)
    for row in rows[:2]:
        recording = data_root / row[3]
        recording.parent.mkdir(parents=True, exist_ok=True)
        recording.touch()


def test_hbn_materializer_caches_only_train_validation_and_exactly_resumes(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "hbn"
    source_manifest = tmp_path / "canonical.csv"
    _canonical_manifest(source_manifest, data_root)
    cache_root = tmp_path / "representations"
    output = tmp_path / "training_manifest.json"
    loaded: list[str] = []
    encoder = _TinyEncoder()

    def prepared_loader(recording):
        loaded.append(recording.subject)
        value = 1.0 if recording.subject == "sub-train" else 2.0
        return PreparedRecording(
            np.full((2, 24_000), value, dtype=np.float32),
            ("E1", "E2"),
        )

    kwargs = {
        "protocol": PROTOCOL,
        "subject_manifest_path": source_manifest,
        "data_root": data_root,
        "preprocessing_cache_root": tmp_path / "preprocessed",
        "representation_cache_root": cache_root,
        "training_manifest_path": output,
        "mapping_path": tmp_path / "reve.json",
        "repository_root": ROOT,
        "device": "cpu",
        "extraction_batch_size": 7,
        "prepared_loader": prepared_loader,
        "encoder_loader": lambda checkpoint, **kwargs: encoder,
    }
    report = materialize_hbn_representations(**kwargs)

    assert set(loaded) == {"sub-train", "sub-val"}
    assert "sub-sealed" not in loaded
    assert [row["split"] for row in report["subjects"]] == [
        "train",
        "validation",
    ]
    assert output.is_file()
    for row in report["subjects"]:
        identity = RepresentationCacheIdentity(
            protocol_sha256=report["protocol_sha256"],
            checkpoint=report["checkpoint"],
            checkpoint_sha256=report["checkpoint_sha256"],
            dataset_manifest_sha256=report["dataset_manifest_sha256"],
            preprocessing_sha256=report["preprocessing_sha256"],
            subject_id=row["subject_id"],
            source_tree_sha256=report["source_tree_sha256"],
        )
        cached = load_cached_representations(cache_root, identity)
        assert cached[-1].shape == (60, 1, 4)

    first_bytes = output.read_bytes()
    resumed = materialize_hbn_representations(**kwargs)
    assert resumed == report
    assert output.read_bytes() == first_bytes

    train_recording = (
        data_root
        / "R1/download/sub-train/eeg/sub-train_task-RestingState_eeg.set"
    )
    train_recording.write_bytes(b"changed HBN acquisition")
    with pytest.raises(Exception, match="training manifest does not match"):
        materialize_hbn_representations(**kwargs)


def test_hbn_preprocessing_cache_key_changes_with_recording_bytes(
    tmp_path: Path,
) -> None:
    recording_path = tmp_path / "sub-test_task-RestingState_eeg.set"
    recording_path.write_bytes(b"first recording")
    companion_path = recording_path.with_suffix(".fdt")
    companion_path.write_bytes(b"first samples")
    recording = HbnRecording(
        path=recording_path,
        release="R1",
        subject="sub-test",
        task="task-RestingState",
        age=12.0,
        duration_s=120.0,
    )
    store = PreprocessedRecordingStore(tmp_path / "cache")

    first = store._cache_paths(recording)
    companion_path.write_bytes(b"second samples")
    second = store._cache_paths(recording)

    assert second != first


def test_hbn_cache_cli_has_no_test_head_or_seed_controls(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    data_root = tmp_path / "hbn"
    source_manifest = tmp_path / "canonical.csv"
    _canonical_manifest(source_manifest, data_root)
    output = tmp_path / "training_manifest.json"
    script_path = ROOT / "scripts/cache_hbn_representations.py"
    spec = importlib.util.spec_from_file_location("cache_hbn_representations", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "materialize_hbn_representations",
        lambda **kwargs: {
            "protocol_sha256": PROTOCOL.sha256,
            "checkpoint_sha256": "a" * 64,
            "subjects": [
                {"subject_id": "sub-train", "split": "train", "age": 10.0},
                {"subject_id": "sub-val", "split": "validation", "age": 14.0},
            ],
        },
    )

    assert module.main(
        [
            "--protocol", str(PROTOCOL_PATH),
            "--subject-manifest", str(source_manifest.resolve()),
            "--data-root", str(data_root.resolve()),
            "--preprocessing-cache-root", str((tmp_path / "preprocessed").resolve()),
            "--representation-cache-root", str((tmp_path / "representations").resolve()),
            "--training-manifest", str(output.resolve()),
            "--mapping", str((tmp_path / "reve.json").resolve()),
            "--device", "cpu",
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["train_subjects"] == 1
    assert summary["validation_subjects"] == 1
    source = script_path.read_text(encoding="utf-8")
    assert 'add_argument("--test' not in source
    assert 'add_argument("--head' not in source
    assert 'add_argument("--seed' not in source
