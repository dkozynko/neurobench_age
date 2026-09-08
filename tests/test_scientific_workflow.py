from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import neurobench_age.research.sealing as sealing_module
from neurobench_age.analysis.confirmatory import analyze_confirmatory_study
from neurobench_age.core.evidence import source_tree_sha256
from neurobench_age.data.medium_subset import manifest_sha256
from neurobench_age.data.mipdb import build_mipdb_inventory, finalize_mipdb_cohort
from neurobench_age.pipelines.external_holdout import (
    ExternalSubjectMaterial,
    RuntimeProvenance,
    run_external_holdout,
)
from neurobench_age.pipelines.frozen_probe_training import (
    load_frozen_probe_training_manifest,
    train_frozen_probe_study,
)
from neurobench_age.pipelines.independent import PreparedRecording
from neurobench_age.pipelines.representation_materialization import (
    materialize_hbn_representations,
    run_mipdb_pilot,
)
from neurobench_age.research.protocol import load_study_protocol
from neurobench_age.research.study_lock import seal_study


ROOT = Path(__file__).resolve().parents[1]


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _hbn_manifest(path: Path, data_root: Path) -> None:
    rows = [
        ("R1", "sub-train-a", 8.0, "R1/sub-train-a_task-RestingState_eeg.set", 120.0, "train"),
        ("R2", "sub-train-b", 10.0, "R2/sub-train-b_task-RestingState_eeg.set", 120.0, "train"),
        ("R8", "sub-val-a", 9.0, "R8/sub-val-a_task-RestingState_eeg.set", 120.0, "val"),
        ("R9", "sub-val-b", 11.0, "R9/sub-val-b_task-RestingState_eeg.set", 120.0, "val"),
        ("R5", "sub-sealed", 12.0, "R5/sub-sealed_task-RestingState_eeg.set", 120.0, "test"),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ("release", "subject", "age", "recording_relpath", "duration_s", "split")
        )
        writer.writerows(rows)
    for row in rows[:-1]:
        recording = data_root / row[3]
        recording.parent.mkdir(parents=True, exist_ok=True)
        recording.write_bytes(f"raw:{row[1]}".encode())


def _mipdb_dataset(root: Path) -> None:
    root.mkdir(parents=True)
    subjects = [f"sub-{index:03d}" for index in range(60)]
    (root / "dataset_description.json").write_text(
        '{"Name":"synthetic MIPDB","DatasetType":"raw"}\n'
    )
    (root / "participants.tsv").write_text(
        "participant_id\tage\n"
        + "".join(
            f"{subject}\t{8.0 + (index % 30) * 0.1:.1f}\n"
            for index, subject in enumerate(subjects)
        )
    )
    for subject in subjects:
        eeg = root / subject / "eeg"
        eeg.mkdir(parents=True)
        (eeg / f"{subject}_task-block01_eeg.set").write_bytes(
            f"raw:{subject}".encode()
        )


def _mipdb_subject_loader(bids_root, subject, *, contract):
    return np.ones((2, 128, 400), dtype=np.float32), {
        "channel_labels": [f"E{index}" for index in range(1, 129)],
        "mapped_channel_count": 128,
        "window_count": 2,
        "cross_block_windows": False,
        "spatial_interpolation": False,
        "qc_reasons": [],
    }


def test_tiny_artifact_derived_sealed_study_runs_end_to_end(
    tmp_path: Path, monkeypatch
) -> None:
    protocol_path, protocol = _protocol(tmp_path)
    encoder = _TinyEncoder()
    hbn_root = tmp_path / "hbn"
    hbn_manifest_path = tmp_path / "hbn.csv"
    _hbn_manifest(hbn_manifest_path, hbn_root)
    hbn_training_manifest = tmp_path / "hbn_training.json"
    representation_root = tmp_path / "hbn_representations"

    materialize_hbn_representations(
        protocol=protocol,
        subject_manifest_path=hbn_manifest_path,
        data_root=hbn_root,
        preprocessing_cache_root=tmp_path / "hbn_preprocessed",
        representation_cache_root=representation_root,
        training_manifest_path=hbn_training_manifest,
        mapping_path=tmp_path / "mapping.json",
        repository_root=ROOT,
        device="cpu",
        extraction_batch_size=2,
            prepared_loader=lambda recording: PreparedRecording(
                np.full(
                    (128, 24_000),
                    float(recording.age or 0.0) / 10.0,
                    dtype=np.float32,
                ),
                tuple(f"E{index}" for index in range(1, 129)),
            ),
        encoder_loader=lambda checkpoint, **kwargs: encoder,
    )
    records = load_frozen_probe_training_manifest(
        hbn_training_manifest, protocol=protocol
    )
    checkpoint_root = tmp_path / "head_runs"
    checkpoint_inventory = train_frozen_probe_study(
        records=records,
        cache_root=representation_root,
        output_root=checkpoint_root,
        training=protocol.training,
        device="cpu",
        training_source_sha256=source_tree_sha256(ROOT),
        available_memory_bytes=16 * 1024**3,
    )
    checkpoint_inventory_path = checkpoint_root / "checkpoint_inventory.json"

    mipdb_root = tmp_path / "mipdb"
    _mipdb_dataset(mipdb_root)
    draft = build_mipdb_inventory(
        mipdb_root,
        protocol=protocol,
        hbn_age_support=(8.0, 11.0),
        hbn_age_support_manifest_sha256=manifest_sha256(hbn_manifest_path),
    )
    draft_path = tmp_path / "mipdb_draft.json"
    draft_path.write_text(json.dumps(draft, sort_keys=True) + "\n")
    pilot_path = tmp_path / "mipdb_pilot_qc.json"
    run_mipdb_pilot(
        protocol=protocol,
        bids_root=mipdb_root,
        manifest_path=draft_path,
        mapping_path=tmp_path / "mapping.json",
        output_path=pilot_path,
        device="cpu",
        extraction_batch_size=2,
        subject_loader=_mipdb_subject_loader,
        encoder_loader=lambda checkpoint, **kwargs: encoder,
    )
    cohort_qc_path = tmp_path / "mipdb_cohort_qc.json"
    final_manifest_path = tmp_path / "mipdb_final.json"
    finalized = finalize_mipdb_cohort(
        bids_root=mipdb_root,
        draft_manifest_path=draft_path,
        protocol=protocol,
        qc_output_path=cohort_qc_path,
        output_path=final_manifest_path,
        subject_loader=_mipdb_subject_loader,
    )
    assert len(finalized["cohorts"]["primary"]) == 50
    assert finalized["underpowered"] is False

    tree_sha256 = source_tree_sha256(ROOT)
    monkeypatch.setattr(
        sealing_module,
        "_repository_provenance",
        lambda root: (tree_sha256, "synthetic-clean-revision", False),
    )
    environment_path = tmp_path / "environment.lock"
    environment_path.write_text("synthetic environment\n")
    external_root = (tmp_path / "external").resolve()
    payload = sealing_module.derive_study_payload(
        protocol=protocol,
        repository_root=ROOT,
        environment_path=environment_path,
        hbn_subject_manifest_path=hbn_manifest_path,
        hbn_training_manifest_path=hbn_training_manifest,
        hbn_data_root=hbn_root,
        checkpoint_root=checkpoint_root,
        checkpoint_inventory_path=checkpoint_inventory_path,
        mipdb_manifest_path=final_manifest_path,
        mipdb_bids_root=mipdb_root,
        mipdb_pilot_qc_path=pilot_path,
        mipdb_cohort_qc_path=cohort_qc_path,
        output_root=external_root,
    )
    lock_path = tmp_path / "study_lock.json"
    lock = seal_study(lock_path, payload)

    ages = {item["subject_id"]: float(item["age"]) for item in finalized["subjects"]}

    def provider(subject_id, identity):
        value = ages[subject_id] / 10.0
        representation = torch.full((2, 1, 4), value)
        return ExternalSubjectMaterial(
            representations={-2: representation + 0.1, -1: representation},
            cache_identity=identity,
            qc={"status": "passed", "subject_id": subject_id, "window_count": 2},
        )

    prediction_inventory = run_external_holdout(
        lock_path=lock_path,
        checkpoint_root=checkpoint_root,
        inventory_path=checkpoint_inventory_path,
        mipdb_manifest_path=final_manifest_path,
        environment_path=environment_path,
        output_root=external_root,
        runtime=RuntimeProvenance(
            training_source_sha256=tree_sha256,
            git_revision="synthetic-clean-revision",
            git_dirty=False,
            environment_sha256=_sha256_file(environment_path),
        ),
        representation_provider=provider,
        device="cpu",
    )
    assert prediction_inventory["prediction_count"] == 2_000
    report = analyze_confirmatory_study(
        protocol_path=protocol_path,
        lock_path=lock_path,
        checkpoint_root=checkpoint_root,
        checkpoint_inventory_path=checkpoint_inventory_path,
        mipdb_manifest_path=final_manifest_path,
        prediction_root=external_root,
        analysis_output_root=tmp_path / "analysis",
    )
    assert report["status"] == "complete"
    assert report["cohort"]["underpowered"] is False
    assert report["lock_sha256"] == lock["lock_sha256"]
    assert checkpoint_inventory["run_count"] == 40
