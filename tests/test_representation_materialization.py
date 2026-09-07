from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from neurobench_age.pipelines.frozen_probe import (
    FrozenEncoderError,
    RepresentationCacheIdentity,
    encoder_state_sha256,
)
from neurobench_age.pipelines.representation_materialization import (
    LazyMipdbRepresentationProvider,
    preprocessing_contract_sha256,
)
from neurobench_age.research.protocol import load_study_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_study_protocol(
    ROOT / "configs/research/external_frozen_probe.json"
)


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    def forward(self, value: torch.Tensor, *, return_output: bool = False):
        # Convert [batch, channels, samples] into a tiny token representation.
        value = value[:, :2, :2].reshape(value.shape[0], 1, 4)
        outputs = []
        for layer in self.layers:
            value = torch.tanh(layer(value))
            outputs.append(value)
        return outputs if return_output else outputs[-1]


def _manifest(path: Path, dataset_sha256: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "finalized",
                "dataset": "MIPDB",
                "protocol_sha256": PROTOCOL.sha256,
                "dataset_manifest_sha256": dataset_sha256,
                "subjects": [
                    {
                        "subject_id": "sub-001",
                        "age": 12.0,
                        "recordings": [
                            "sub-001/eeg/sub-001_task-block01_eeg.vhdr"
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_lazy_external_provider_reads_no_eeg_before_started_marker(
    tmp_path: Path,
) -> None:
    encoder = _TinyEncoder()
    checkpoint_sha256 = encoder_state_sha256(encoder)
    dataset_sha256 = "c" * 64
    manifest = tmp_path / "mipdb.json"
    _manifest(manifest, dataset_sha256)
    calls = {"subject": 0, "encoder": 0}

    def subject_loader(*args, **kwargs):
        calls["subject"] += 1
        return np.ones((3, 128, 400), dtype=np.float32), {
            "channel_labels": [f"E{index}" for index in range(1, 129)],
            "mapped_channel_count": 128,
            "window_count": 3,
            "cross_block_windows": False,
            "spatial_interpolation": False,
            "qc_reasons": [],
        }

    def encoder_loader(*args, **kwargs):
        calls["encoder"] += 1
        assert kwargs["initialization_seed"] == PROTOCOL.encoder.initialization_seed
        return encoder

    marker = tmp_path / "output" / "evaluation_started.json"
    provider = LazyMipdbRepresentationProvider(
        protocol=PROTOCOL,
        bids_root=tmp_path / "bids",
        manifest_path=manifest,
        cache_root=tmp_path / "cache",
        mapping_path=tmp_path / "reve.json",
        started_marker_path=marker,
        expected_lock_sha256="f" * 64,
        device="cpu",
        extraction_batch_size=2,
        subject_loader=subject_loader,
        encoder_loader=encoder_loader,
    )
    identity = RepresentationCacheIdentity(
        protocol_sha256=PROTOCOL.sha256,
        checkpoint=PROTOCOL.encoder.checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        dataset_manifest_sha256=dataset_sha256,
        preprocessing_sha256=preprocessing_contract_sha256(PROTOCOL.preprocessing),
        subject_id="sub-001",
        source_tree_sha256="e" * 64,
    )

    assert calls == {"subject": 0, "encoder": 0}
    with pytest.raises(FrozenEncoderError, match="started marker"):
        provider("sub-001", identity)
    assert calls == {"subject": 0, "encoder": 0}

    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps({"lock_sha256": "f" * 64, "state": "started"}) + "\n"
    )
    material = provider("sub-001", identity)

    assert calls == {"subject": 1, "encoder": 1}
    assert material.cache_identity == identity
    assert material.representations[-2].shape[0] == 3
    assert material.representations[-1].shape[0] == 3
    assert material.qc["mapped_channel_count"] == 128


def test_lazy_external_provider_exactly_resumes_complete_cache(tmp_path: Path) -> None:
    encoder = _TinyEncoder()
    dataset_sha256 = "c" * 64
    manifest = tmp_path / "mipdb.json"
    _manifest(manifest, dataset_sha256)
    marker = tmp_path / "output" / "evaluation_started.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps({"lock_sha256": "f" * 64, "state": "started"}) + "\n"
    )
    calls = {"subject": 0}

    def subject_loader(*args, **kwargs):
        calls["subject"] += 1
        return np.ones((2, 128, 400), dtype=np.float32), {
            "channel_labels": [f"E{index}" for index in range(1, 129)],
            "mapped_channel_count": 128,
            "window_count": 2,
            "cross_block_windows": False,
            "spatial_interpolation": False,
            "qc_reasons": [],
        }

    kwargs = {
        "protocol": PROTOCOL,
        "bids_root": tmp_path / "bids",
        "manifest_path": manifest,
        "cache_root": tmp_path / "cache",
        "mapping_path": tmp_path / "reve.json",
        "started_marker_path": marker,
        "expected_lock_sha256": "f" * 64,
        "device": "cpu",
        "extraction_batch_size": 1,
    }
    identity = RepresentationCacheIdentity(
        protocol_sha256=PROTOCOL.sha256,
        checkpoint=PROTOCOL.encoder.checkpoint,
        checkpoint_sha256=encoder_state_sha256(encoder),
        dataset_manifest_sha256=dataset_sha256,
        preprocessing_sha256=preprocessing_contract_sha256(PROTOCOL.preprocessing),
        subject_id="sub-001",
        source_tree_sha256="e" * 64,
    )
    first = LazyMipdbRepresentationProvider(
        **kwargs,
        subject_loader=subject_loader,
        encoder_loader=lambda checkpoint, **unused: encoder,
    )
    expected = first("sub-001", identity)

    resumed = LazyMipdbRepresentationProvider(
        **kwargs,
        subject_loader=lambda *args, **kwargs: pytest.fail("raw EEG was reread"),
        encoder_loader=lambda **kwargs: pytest.fail("encoder was rebuilt"),
    )
    actual = resumed("sub-001", identity)

    assert calls["subject"] == 1
    assert torch.equal(actual.representations[-1], expected.representations[-1])
    assert actual.qc == expected.qc


def test_lazy_external_provider_reuses_one_verified_encoder_for_subjects(
    tmp_path: Path,
) -> None:
    encoder = _TinyEncoder()
    dataset_sha256 = "c" * 64
    manifest = tmp_path / "mipdb.json"
    _manifest(manifest, dataset_sha256)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["subjects"].append(
        {
            "subject_id": "sub-002",
            "age": 13.0,
            "recordings": ["sub-002/eeg/sub-002_task-block01_eeg.vhdr"],
        }
    )
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    marker = tmp_path / "output/evaluation_started.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps({"lock_sha256": "f" * 64, "state": "started"}) + "\n"
    )
    encoder_loads = 0

    def encoder_loader(*args, **kwargs):
        nonlocal encoder_loads
        encoder_loads += 1
        return encoder

    def subject_loader(*args, **kwargs):
        return np.ones((2, 128, 400), dtype=np.float32), {
            "channel_labels": [f"E{index}" for index in range(1, 129)],
            "mapped_channel_count": 128,
            "window_count": 2,
            "cross_block_windows": False,
            "spatial_interpolation": False,
            "qc_reasons": [],
        }

    provider = LazyMipdbRepresentationProvider(
        protocol=PROTOCOL,
        bids_root=tmp_path / "bids",
        manifest_path=manifest,
        cache_root=tmp_path / "cache",
        mapping_path=tmp_path / "reve.json",
        started_marker_path=marker,
        expected_lock_sha256="f" * 64,
        device="cpu",
        extraction_batch_size=1,
        subject_loader=subject_loader,
        encoder_loader=encoder_loader,
    )
    common = {
        "protocol_sha256": PROTOCOL.sha256,
        "checkpoint": PROTOCOL.encoder.checkpoint,
        "checkpoint_sha256": encoder_state_sha256(encoder),
        "dataset_manifest_sha256": dataset_sha256,
        "preprocessing_sha256": preprocessing_contract_sha256(PROTOCOL.preprocessing),
        "source_tree_sha256": "e" * 64,
    }
    for subject_id in ("sub-001", "sub-002"):
        provider(
            subject_id,
            RepresentationCacheIdentity(subject_id=subject_id, **common),
        )

    assert encoder_loads == 1
