from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neurobench_age.pipelines.ds006780_external import (
    Ds006780ExternalError,
    _infer_summary_embed_dim,
    _eeg_channel_names,
    checkpoint_path_for_run,
    feature_tensor_for_head,
    _write_or_verify,
)


def test_checkpoint_path_for_capacity_run_is_unambiguous() -> None:
    path = checkpoint_path_for_run(
        Path("/outputs/checkpoints"),
        {"training_size": 800, "head": "mean_rich_stats_residual", "seed": 42},
    )

    assert path == Path(
        "/outputs/checkpoints/n-800/mean_rich_stats_residual/seed-42/head_checkpoint.pt"
    )


def test_feature_tensor_for_head_uses_one_fixed_summary_per_head() -> None:
    tokens = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)

    linear = feature_tensor_for_head(tokens, "mean_linear")
    rich = feature_tensor_for_head(tokens, "mean_rich_stats_residual")

    assert tuple(linear.shape) == (2, 1, 4)
    assert tuple(rich.shape) == (2, 1, 20)
    assert torch.equal(linear[:, 0], tokens.mean(dim=1))


def test_feature_tensor_for_head_rejects_unplanned_head() -> None:
    with pytest.raises(Ds006780ExternalError, match="selected head"):
        feature_tensor_for_head(torch.ones(2, 3, 4), "multi_query_rich_stats")


def test_rich_summary_checkpoint_uses_mean_width_and_four_stats_blocks() -> None:
    state_dict = {
        "linear.weight": torch.zeros(1, 512),
        "correction.weight": torch.zeros(1, 2048),
    }

    assert _infer_summary_embed_dim(state_dict, "mean_rich_stats_residual") == 512


def test_encoder_channel_inventory_filters_non_eeg_channels() -> None:
    candidate = {
        "channel_inventory": [
            *({"name": f"EEG{index}", "type": "EEG"} for index in range(64)),
            {"name": "EXG1", "type": "EMG"},
            {"name": "Status", "type": "TRIG"},
        ]
    }

    assert _eeg_channel_names(candidate) == tuple(f"EEG{index}" for index in range(64))


def test_write_or_verify_creates_missing_immutable_artifact(tmp_path: Path) -> None:
    path = tmp_path / "marker.json"
    payload = {"status": "complete"}

    _write_or_verify(path, payload, "marker")
    _write_or_verify(path, payload, "marker")

    assert path.exists()
    assert json.loads(path.read_text()) == payload
