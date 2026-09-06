from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from neurobench_age.pipelines.frozen_probe import (
    FrozenEncoderError,
    RepresentationCacheIdentity,
    write_cached_representations,
)
from neurobench_age.pipelines.frozen_probe_training import (
    load_frozen_probe_training_manifest,
)
from neurobench_age.research.protocol import load_study_protocol
from neurobench_age.research.study_lock import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]


def _protocol(tmp_path: Path):
    payload = json.loads(
        (ROOT / "configs/research/external_frozen_probe.json").read_text()
    )
    payload["training"].update(
        {"batch_size": 2, "max_epochs": 1, "patience": 1}
    )
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, load_study_protocol(path)


def _write_inputs(tmp_path: Path):
    protocol_path, protocol = _protocol(tmp_path)
    cache_root = tmp_path / "cache"
    acquisition_files = [
        {
            "subject_id": "sub-acquisition",
            "path": "R1/sub-acquisition.set",
            "size_bytes": 10,
            "sha256": "a" * 64,
        }
    ]
    subject_manifest_sha256 = "9" * 64
    common = {
        "checkpoint": "brain-bzh/reve-base",
        "checkpoint_sha256": "b" * 64,
        "dataset_manifest_sha256": canonical_sha256(
            {
                "subject_manifest_sha256": subject_manifest_sha256,
                "acquisition_files": acquisition_files,
            }
        ),
        "preprocessing_sha256": "d" * 64,
        "source_tree_sha256": "e" * 64,
    }
    subjects = []
    for index, (split, age) in enumerate(
        (
            ("train", 8.0),
            ("train", 11.0),
            ("train", 14.0),
            ("validation", 9.0),
            ("validation", 12.0),
            ("validation", 15.0),
        ),
        start=1,
    ):
        subject_id = f"sub-{index:03d}"
        identity = RepresentationCacheIdentity(
            protocol_sha256=protocol.sha256,
            subject_id=subject_id,
            **common,
        )
        signal = torch.full((3, 2, 2), age / 10.0)
        write_cached_representations(
            cache_root,
            identity,
            {-2: signal + 0.5, -1: signal},
            evidence={
                "encoder_frozen": True,
                "encoder_eval_mode": True,
                "inference_mode": True,
                "layer_indices": [-2, -1],
                "state_sha256_before": "f" * 64,
                "state_sha256_after": "f" * 64,
            },
        )
        subjects.append({"subject_id": subject_id, "split": split, "age": age})
    manifest = {
        "schema_version": 1,
        "protocol_sha256": protocol.sha256,
        **common,
        "subject_manifest_sha256": subject_manifest_sha256,
        "acquisition_files": acquisition_files,
        "subjects": subjects,
    }
    manifest_path = tmp_path / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return protocol_path, protocol, manifest_path, cache_root


def test_training_manifest_rejects_protocol_drift_and_test_split(
    tmp_path: Path,
) -> None:
    _, protocol, manifest_path, _ = _write_inputs(tmp_path)
    payload = json.loads(manifest_path.read_text())
    payload["protocol_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(FrozenEncoderError, match="protocol_sha256"):
        load_frozen_probe_training_manifest(manifest_path, protocol=protocol)

    payload["protocol_sha256"] = protocol.sha256
    payload["subjects"][0]["split"] = "test"
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(FrozenEncoderError, match="train or validation"):
        load_frozen_probe_training_manifest(manifest_path, protocol=protocol)


def test_training_manifest_rejects_acquisition_identity_drift(tmp_path: Path) -> None:
    _, protocol, manifest_path, _ = _write_inputs(tmp_path)
    payload = json.loads(manifest_path.read_text())
    payload["acquisition_files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(FrozenEncoderError, match="dataset.*identity"):
        load_frozen_probe_training_manifest(manifest_path, protocol=protocol)


def test_frozen_probe_cli_runs_exact_matrix_and_exact_resume(
    tmp_path: Path, capsys
) -> None:
    protocol_path, _, manifest_path, cache_root = _write_inputs(tmp_path)
    output_root = tmp_path / "runs"
    script_path = ROOT / "scripts/run_frozen_probe.py"
    spec = importlib.util.spec_from_file_location("run_frozen_probe", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arguments = [
        "--protocol",
        str(protocol_path),
        "--training-manifest",
        str(manifest_path),
        "--cache-root",
        str(cache_root),
        "--output-root",
        str(output_root),
        "--device",
        "cpu",
    ]

    assert module.main(arguments) == 0
    first_summary = json.loads(capsys.readouterr().out)
    assert first_summary["run_count"] == 40
    first_inventory = (output_root / "checkpoint_inventory.json").read_bytes()

    assert module.main(arguments) == 0
    assert (output_root / "checkpoint_inventory.json").read_bytes() == first_inventory
    summary = json.loads(capsys.readouterr().out)
    assert summary["run_count"] == 40
    assert summary["protocol_sha256"]

    source = script_path.read_text(encoding="utf-8")
    assert 'add_argument("--test' not in source
    assert 'add_argument("--seed' not in source
    assert 'add_argument("--head' not in source
