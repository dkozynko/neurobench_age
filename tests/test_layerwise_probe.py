from __future__ import annotations

from pathlib import Path
import json

import pytest
import torch

from neurobench_age.pipelines.frozen_probe import FrozenEncoderError
from neurobench_age.pipelines.layerwise_external import (
    LayerwiseExternalError,
    _discover_primary_cache_source_tree_sha256,
)
from neurobench_age.research.layerwise_probe import (
    build_layerwise_probe_head,
    layerwise_head_specs,
    required_layer_for_layerwise_head,
)
from neurobench_age.research.protocol import load_study_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_study_protocol(
    ROOT / "configs/research/layerwise_probe.json", profile="layerwise"
)


def test_layerwise_protocol_maps_one_mean_linear_head_to_each_layer() -> None:
    specs = layerwise_head_specs(PROTOCOL)

    assert tuple(spec.layer_index for spec in specs) == (-4, -3, -2, -1)
    assert required_layer_for_layerwise_head(
        "mean_linear_layer_m3", specs=specs
    ) == -3

    head = build_layerwise_probe_head(
        "mean_linear_layer_m3", embed_dim=4, specs=specs
    )
    prediction = head(torch.randn(3, 5, 4))

    assert tuple(prediction.shape) == (3, 1)
    assert not hasattr(head, "encoder")


def test_layerwise_head_mapping_rejects_unknown_head() -> None:
    specs = layerwise_head_specs(PROTOCOL)

    with pytest.raises(FrozenEncoderError, match="unknown layer-wise head"):
        required_layer_for_layerwise_head("mean_linear", specs=specs)


def test_primary_cache_source_identity_is_read_from_all_entries(tmp_path: Path) -> None:
    cache_root = tmp_path / "mipdb-cache"
    source_sha = "a" * 64
    identity = {
        "protocol_sha256": "b" * 64,
        "checkpoint": "brain-bzh/reve-base",
        "checkpoint_sha256": "c" * 64,
        "dataset_manifest_sha256": "d" * 64,
        "preprocessing_sha256": "e" * 64,
        "subject_id": "sub-001",
        "source_tree_sha256": source_sha,
    }
    entry = cache_root / ("f" * 64)
    entry.mkdir(parents=True)
    (entry / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "cache_key": entry.name,
                "identity": identity,
            }
        ),
        encoding="utf-8",
    )

    assert (
        _discover_primary_cache_source_tree_sha256(
            cache_root,
            protocol_sha256=identity["protocol_sha256"],
            checkpoint=identity["checkpoint"],
            checkpoint_sha256=identity["checkpoint_sha256"],
            dataset_manifest_sha256=identity["dataset_manifest_sha256"],
            preprocessing_sha256=identity["preprocessing_sha256"],
        )
        == source_sha
    )

    identity["source_tree_sha256"] = "1" * 64
    second = cache_root / ("0" * 64)
    second.mkdir()
    second_metadata = json.loads((entry / "metadata.json").read_text())
    second_metadata["cache_key"] = second.name
    second_metadata["identity"] = identity
    (second / "metadata.json").write_text(
        json.dumps(second_metadata),
        encoding="utf-8",
    )
    with pytest.raises(LayerwiseExternalError, match="inconsistent source identities"):
        _discover_primary_cache_source_tree_sha256(
            cache_root,
            protocol_sha256="b" * 64,
            checkpoint="brain-bzh/reve-base",
            checkpoint_sha256="c" * 64,
            dataset_manifest_sha256="d" * 64,
            preprocessing_sha256="e" * 64,
        )
