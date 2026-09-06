from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from neurobench_age.pipelines.representation_materialization import run_mipdb_pilot
from neurobench_age.research.protocol import load_study_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/research/external_frozen_probe.json"
PROTOCOL = load_study_protocol(PROTOCOL_PATH)


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


def _manifest(path: Path) -> list[str]:
    pilot = [f"sub-{index:03d}" for index in range(10)]
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "draft",
                "dataset": "MIPDB",
                "protocol_sha256": PROTOCOL.sha256,
                "dataset_manifest_sha256": "c" * 64,
                "cohorts": {"pilot": pilot, "primary": ["sub-999"], "extrapolation": []},
                "subjects": [
                    {
                        "subject_id": subject_id,
                        "age": 10.0 + index,
                        "recordings": [
                            f"{subject_id}/eeg/{subject_id}_task-block01_eeg.vhdr"
                        ],
                    }
                    for index, subject_id in enumerate([*pilot, "sub-999"])
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return pilot


def _assert_no_target_or_prediction_fields(value: object) -> None:
    forbidden = {"age", "target", "targets", "prediction", "predictions", "metrics"}
    if isinstance(value, dict):
        assert not (set(value) & forbidden)
        for item in value.values():
            _assert_no_target_or_prediction_fields(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_target_or_prediction_fields(item)


def test_pilot_runs_exact_ten_without_retaining_targets_or_predictions(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "mipdb.json"
    pilot = _manifest(manifest)
    output = tmp_path / "pilot_qc.json"
    loaded: list[str] = []

    def subject_loader(bids_root, subject, *, contract):
        loaded.append(subject["subject_id"])
        return np.ones((2, 128, 400), dtype=np.float32), {
            "channel_labels": [f"E{index}" for index in range(1, 129)],
            "mapped_channel_count": 128,
            "window_count": 2,
            "cross_block_windows": False,
            "spatial_interpolation": False,
            "qc_reasons": [],
        }

    report = run_mipdb_pilot(
        protocol=PROTOCOL,
        bids_root=tmp_path / "bids",
        manifest_path=manifest,
        mapping_path=tmp_path / "reve.json",
        output_path=output,
        device="cpu",
        extraction_batch_size=1,
        subject_loader=subject_loader,
        encoder_loader=lambda checkpoint, **kwargs: _TinyEncoder(),
    )

    assert loaded == pilot
    assert report["status"] == "passed"
    assert report["pilot_subject_count"] == 10
    assert len(report["draft_manifest_sha256"]) == 64
    assert [row["subject_id"] for row in report["subjects"]] == pilot
    assert all(row["representation_shapes"]["-1"] == [2, 1, 4] for row in report["subjects"])
    _assert_no_target_or_prediction_fields(report)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    with pytest.raises(Exception, match="exists"):
        run_mipdb_pilot(
            protocol=PROTOCOL,
            bids_root=tmp_path / "bids",
            manifest_path=manifest,
            mapping_path=tmp_path / "reve.json",
            output_path=output,
            device="cpu",
            extraction_batch_size=1,
            subject_loader=subject_loader,
            encoder_loader=lambda checkpoint, **kwargs: _TinyEncoder(),
        )


def test_pilot_cli_exposes_no_head_seed_or_metric_override(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest = tmp_path / "mipdb.json"
    _manifest(manifest)
    output = tmp_path / "pilot.json"
    script_path = ROOT / "scripts/run_mipdb_pilot.py"
    spec = importlib.util.spec_from_file_location("run_mipdb_pilot", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "run_mipdb_pilot",
        lambda **kwargs: {
            "status": "passed",
            "pilot_subject_count": 10,
            "protocol_sha256": PROTOCOL.sha256,
        },
    )

    assert module.main(
        [
            "--protocol", str(PROTOCOL_PATH),
            "--bids-root", str((tmp_path / "bids").resolve()),
            "--mipdb-manifest", str(manifest.resolve()),
            "--mapping", str((tmp_path / "reve.json").resolve()),
            "--output", str(output.resolve()),
            "--device", "cpu",
        ]
    ) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["pilot_subject_count"] == 10
    source = script_path.read_text(encoding="utf-8")
    assert 'add_argument("--seed' not in source
    assert 'add_argument("--head' not in source
    assert 'add_argument("--metric' not in source
