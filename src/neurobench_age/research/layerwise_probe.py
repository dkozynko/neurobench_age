"""Secondary layer-wise linear probing with the primary frozen-head trainer."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from neurobench_age.heads.math import MeanLinearCopyHead
from neurobench_age.pipelines.frozen_probe import FrozenEncoderError, _canonical_sha256
from neurobench_age.pipelines.frozen_probe_training import (
    CachedSubjectRecord,
    GlobalWindowBatchPlan,
    ValidatedRepresentationStore,
    _load_completed_run,
    _prepare_training_store_device,
    _run_identity,
    train_frozen_probe_run,
    validate_training_records,
)
from neurobench_age.research.protocol import StudyProtocol
from neurobench_age.research.training_protocol import FrozenProbeTrainingProtocol


@dataclass(frozen=True)
class LayerwiseHeadSpec:
    """One fixed mean-linear probe assigned to one frozen encoder layer."""

    name: str
    layer_index: int
    aggregation: str


def layerwise_head_specs(protocol: StudyProtocol) -> tuple[LayerwiseHeadSpec, ...]:
    """Return and validate the layer-to-head mapping declared in the protocol."""

    if protocol.study_id != "reve_age_layerwise_probe_v1":
        raise FrozenEncoderError("layer-wise runner requires its dedicated protocol")
    expected_layers = tuple(protocol.encoder.layer_indices)
    specs = tuple(
        LayerwiseHeadSpec(head.name, head.layer_index, head.aggregation)
        for head in protocol.heads
    )
    if (
        not specs
        or tuple(spec.layer_index for spec in specs) != expected_layers
        or len({spec.name for spec in specs}) != len(specs)
        or any(spec.aggregation != "mean" for spec in specs)
    ):
        raise FrozenEncoderError("layer-wise heads do not match the declared layer inventory")
    return specs


def required_layer_for_layerwise_head(
    head_name: str, *, specs: Sequence[LayerwiseHeadSpec]
) -> int:
    for spec in specs:
        if spec.name == head_name:
            return spec.layer_index
    raise FrozenEncoderError(f"unknown layer-wise head: {head_name}")


def build_layerwise_probe_head(
    head_name: str,
    *,
    embed_dim: int,
    specs: Sequence[LayerwiseHeadSpec],
) -> nn.Module:
    """Build the same trainable mean-linear head for every selected layer."""

    required_layer_for_layerwise_head(head_name, specs=specs)
    return MeanLinearCopyHead(embed_dim=embed_dim, n_outputs=1)


def _expected_run_paths(
    output_root: Path, *, specs: Sequence[LayerwiseHeadSpec], seeds: Sequence[int]
) -> dict[tuple[str, int], Path]:
    return {
        (spec.name, int(seed)): output_root / spec.name / f"seed-{int(seed)}"
        for spec in specs
        for seed in seeds
    }


def _audit_inventory(
    *,
    output_root: Path,
    records: Sequence[CachedSubjectRecord],
    protocol: StudyProtocol,
    training: FrozenProbeTrainingProtocol,
    specs: Sequence[LayerwiseHeadSpec],
    training_source_sha256: str,
) -> Mapping[str, Any]:
    expected_paths = _expected_run_paths(
        Path(output_root), specs=specs, seeds=training.seeds
    )
    runs: list[dict[str, Any]] = []
    for (head_name, seed), run_dir in expected_paths.items():
        identity = _run_identity(
            head_name=head_name,
            seed=seed,
            records=records,
            training=training,
            training_source_sha256=training_source_sha256,
            run_metadata={
                "analysis": "layerwise_mean_linear",
                "layer_index": required_layer_for_layerwise_head(
                    head_name, specs=specs
                ),
            },
        )
        manifest = _load_completed_run(
            run_dir,
            expected_identity_sha256=_canonical_sha256(identity),
            training_source_sha256=training_source_sha256,
        )
        runs.append(
            {
                "head_name": head_name,
                "layer_index": required_layer_for_layerwise_head(
                    head_name, specs=specs
                ),
                "seed": seed,
                "run_identity_sha256": _canonical_sha256(identity),
                "run_manifest_sha256": manifest["run_manifest_sha256"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "selected_epoch": manifest["selected_epoch"],
                "head_parameter_count": manifest["head_parameters"]["trainable"],
            }
        )
    body: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "study_id": protocol.study_id,
        "protocol_sha256": protocol.sha256,
        "training_protocol_sha256": training.sha256,
        "training_source_sha256": training_source_sha256,
        "layer_indices": list(protocol.encoder.layer_indices),
        "heads": [spec.name for spec in specs],
        "head_layers": {spec.name: spec.layer_index for spec in specs},
        "seeds": list(training.seeds),
        "run_count": len(runs),
        "runs": runs,
    }
    inventory = {**body, "checkpoint_inventory_sha256": _canonical_sha256(body)}
    path = Path(output_root) / "checkpoint_inventory.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != inventory:
            raise FrozenEncoderError("layer-wise checkpoint inventory does not match runs")
    else:
        path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return inventory


def train_layerwise_probe_study(
    *,
    records: Sequence[CachedSubjectRecord],
    cache_root: Path,
    output_root: Path,
    protocol: StudyProtocol,
    training: FrozenProbeTrainingProtocol,
    device: str,
    training_source_sha256: str,
    progress_sink: Callable[[Mapping[str, Any]], None] | None = None,
    available_memory_bytes: int | None = None,
) -> Mapping[str, Any]:
    """Train or resume the exact layer-by-seed matrix without touching primary runs."""

    records = validate_training_records(records)
    specs = layerwise_head_specs(protocol)
    if training.seeds != tuple(range(33, 43)):
        raise FrozenEncoderError("layer-wise study requires seeds 33 through 42")
    output_root = Path(output_root)
    if not output_root.is_absolute():
        raise FrozenEncoderError("layer-wise output_root must be absolute")
    expected_paths = _expected_run_paths(
        output_root, specs=specs, seeds=training.seeds
    )
    allowed_heads = {spec.name for spec in specs}
    actual_heads = {
        path.name for path in output_root.iterdir() if path.is_dir()
    } if output_root.is_dir() else set()
    if actual_heads - allowed_heads:
        raise FrozenEncoderError("layer-wise output contains unexpected head directories")
    for (head_name, seed), run_dir in expected_paths.items():
        if run_dir.exists():
            _load_completed_run(
                run_dir,
                expected_identity_sha256=_canonical_sha256(
                    _run_identity(
                        head_name=head_name,
                        seed=seed,
                        records=records,
                        training=training,
                        training_source_sha256=training_source_sha256,
                        run_metadata={
                            "analysis": "layerwise_mean_linear",
                            "layer_index": required_layer_for_layerwise_head(
                                head_name, specs=specs
                            ),
                        },
                    )
                ),
                training_source_sha256=training_source_sha256,
            )

    store = ValidatedRepresentationStore.build(
        records=records,
        cache_root=cache_root,
        required_layers=protocol.encoder.layer_indices,
        available_memory_bytes=available_memory_bytes,
    )
    if progress_sink is not None:
        progress_sink(
            {
                "event": "layerwise_store_load_complete",
                "subject_count": store.subject_count,
                "projected_bytes": store.projected_bytes,
                "actual_retained_bytes": store.actual_retained_bytes,
            }
        )

    train_records = tuple(record for record in records if record.split == "train")
    shared_cpu_stores: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    active_layer: int | None = None
    active_store: tuple[torch.Tensor, torch.Tensor, str] | None = None
    completed = 0
    for spec in specs:
        for seed in training.seeds:
            if spec.layer_index not in shared_cpu_stores:
                tensors = {
                    record.subject_id: store.tensor(record.subject_id, spec.layer_index)
                    for record in train_records
                }
                plan = GlobalWindowBatchPlan.build(
                    train_records, tensors=tensors, batch_size=training.batch_size
                )
                shared_cpu_stores[spec.layer_index] = plan.flatten(tensors)
            if active_layer != spec.layer_index:
                if active_store is not None and active_store[0].device.type == "cuda":
                    del active_store
                    torch.cuda.empty_cache()
                cpu_features, cpu_targets = shared_cpu_stores[spec.layer_index]
                active_store = (
                    *_prepare_training_store_device(
                        cpu_features, cpu_targets, device=device
                    ),
                )
                active_layer = spec.layer_index
            result = train_frozen_probe_run(
                head_name=spec.name,
                seed=seed,
                records=records,
                cache_root=cache_root,
                run_dir=output_root / spec.name / f"seed-{seed}",
                training=training,
                device=device,
                training_source_sha256=training_source_sha256,
                representation_store=store,
                flat_training_store=active_store,
                head_builder=lambda name, *, embed_dim, specs=specs: build_layerwise_probe_head(
                    name, embed_dim=embed_dim, specs=specs
                ),
                required_layer_resolver=lambda name, specs=specs: required_layer_for_layerwise_head(
                    name, specs=specs
                ),
                run_metadata={
                    "analysis": "layerwise_mean_linear",
                    "layer_index": spec.layer_index,
                },
                progress_sink=progress_sink,
            )
            completed += 1
            if progress_sink is not None:
                progress_sink(
                    {
                        "event": "layerwise_progress",
                        "completed_runs": completed,
                        "total_runs": len(expected_paths),
                        "head_name": spec.name,
                        "layer_index": spec.layer_index,
                        "seed": seed,
                        "reused": result.reused,
                    }
                )
    return _audit_inventory(
        output_root=output_root,
        records=records,
        protocol=protocol,
        training=training,
        specs=specs,
        training_source_sha256=training_source_sha256,
    )
