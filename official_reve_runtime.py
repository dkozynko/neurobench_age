"""Scoped NeuralBench patch lifecycle for the official REVE runner.

This module owns temporary monkeypatches and cleanup. It receives the facade as
the hooks object so tests and callers can keep replacing the public seams.
"""

from __future__ import annotations

import json
import math
import os
import sys
import types
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_CONFIGURE_OPTIMIZERS_ABSENT = object()


def _last_tuned_configure_optimizers(brain_module: Any, *, hooks: Any) -> dict[str, Any]:
    """Build the tuned optimizer from one prepared BrainModule instance."""

    model = getattr(brain_module, "model", None)
    trainer = getattr(brain_module, "trainer", None)
    return hooks._load_reve_helpers().build_last_tuned_optimizer_config(model, trainer=trainer)


def _patch_last_tuned_configure_optimizers(
    brain_module: Any,
    patched_modules: list[dict[str, Any]],
    *,
    hooks: Any,
) -> None:
    """Install one instance-only optimizer override and record restoration state."""

    if brain_module is None:
        raise RuntimeError("last_tuned prepare_pl_module created no _brain_module")
    if not isinstance(patched_modules, list):
        raise TypeError("patched_modules must be a per-run list")
    if any(record["module"] is brain_module for record in patched_modules):
        raise RuntimeError("last_tuned BrainModule was already patched in this run")

    instance_attributes = getattr(brain_module, "__dict__", None)
    if not isinstance(instance_attributes, dict):
        raise TypeError("last_tuned BrainModule must expose instance attributes")
    previous = instance_attributes.get("configure_optimizers", _CONFIGURE_OPTIMIZERS_ABSENT)
    record = {
        "module": brain_module,
        "previous": previous,
    }
    patched_modules.append(record)
    try:
        def configure_optimizers(module: Any) -> dict[str, Any]:
            return _last_tuned_configure_optimizers(module, hooks=hooks)

        brain_module.configure_optimizers = types.MethodType(configure_optimizers, brain_module)
    except BaseException:
        patched_modules.pop()
        raise


def _restore_last_tuned_configure_optimizers(
    patched_modules: list[dict[str, Any]],
) -> None:
    """Restore every patched instance in reverse installation order."""

    restoration_errors: list[BaseException] = []
    while patched_modules:
        record = patched_modules.pop()
        brain_module = record["module"]
        previous = record["previous"]
        try:
            if previous is _CONFIGURE_OPTIMIZERS_ABSENT:
                instance_attributes = getattr(brain_module, "__dict__", {})
                if "configure_optimizers" in instance_attributes:
                    delattr(brain_module, "configure_optimizers")
            else:
                brain_module.configure_optimizers = previous
        except BaseException as error:
            restoration_errors.append(error)
    if restoration_errors:
        error = RuntimeError("failed to restore one or more last_tuned configure_optimizers patches")
        for restoration_error in restoration_errors:
            error.add_note(repr(restoration_error))
        raise error


def _last_tuned_report_metadata(
    *,
    query_metadata: Mapping[str, Any],
    optimizer_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Flatten resolved tuning state into the stable report representation."""

    scheduler = optimizer_config["scheduler"]
    scheduler_kwargs = scheduler["kwargs"]
    return {
        **dict(query_metadata),
        "optimizer": optimizer_config["optimizer"]["name"],
        "base_learning_rate": optimizer_config["optimizer"]["lr"],
        "query_learning_rate": optimizer_config["param_groups"][1]["learning_rate"],
        "weight_decay": optimizer_config["optimizer"]["kwargs"]["weight_decay"],
        "scheduler": scheduler["name"],
        "scheduler_max_lr": list(scheduler_kwargs["max_lr"]),
        "scheduler_pct_start": scheduler_kwargs["pct_start"],
        "scheduler_anneal_strategy": scheduler_kwargs["anneal_strategy"],
        "scheduler_div_factor": scheduler_kwargs["div_factor"],
        "scheduler_final_div_factor": scheduler_kwargs["final_div_factor"],
        "scheduler_interval": scheduler["interval"],
        "scheduler_frequency": scheduler["frequency"],
        "optimizer_param_groups": list(optimizer_config["param_groups"]),
        "monitor": "val/pearsonr",
        "checkpoint_selection_monitor": "val/pearsonr",
        "test_pearsonr_role": "diagnostic_only",
    }


def _merge_last_tuned_result_metadata(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the resolved tuning metadata captured from the official test pass."""

    metadata: dict[str, Any] = {}
    for result in results:
        candidate = result.get("tuning_metadata")
        if isinstance(candidate, Mapping):
            metadata.update(candidate)
    return metadata


def _capture_test_result(
    result: Mapping[str, Any],
    *,
    head_variant: str,
    experiment_id: int,
    tuning_metadata_by_experiment: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Copy one official result and attach late-bound tuned metadata."""

    captured = dict(result)
    if head_variant != "last_tuned":
        return captured
    metadata = captured.get("tuning_metadata")
    merged = dict(metadata) if isinstance(metadata, Mapping) else {}
    merged.update(tuning_metadata_by_experiment.get(experiment_id, {}))
    captured["tuning_metadata"] = merged
    return captured


def _read_validation_history(path: Path, *, seed: int) -> list[dict[str, Any]]:
    """Read and validate strict one-based validation history records."""

    if not path.is_file():
        raise RuntimeError(f"strict validation history is missing: {path}")

    records: list[dict[str, Any]] = []
    seen_epochs: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid strict validation JSON at line {line_number}") from error
        if not isinstance(row, Mapping):
            raise RuntimeError(f"strict validation record at line {line_number} is not an object")
        row_seed = row.get("seed")
        epoch = row.get("epoch")
        metric = row.get("val/pearsonr")
        if row_seed != seed or isinstance(row_seed, bool):
            raise RuntimeError(f"strict validation seed mismatch at line {line_number}")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise RuntimeError(f"strict validation epoch is invalid at line {line_number}")
        if epoch in seen_epochs:
            raise RuntimeError(f"duplicate epoch in strict validation history: {epoch}")
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise RuntimeError(f"strict validation val/pearsonr is invalid at line {line_number}")
        metric_value = float(metric)
        if not math.isfinite(metric_value):
            raise RuntimeError(f"strict validation val/pearsonr is non-finite at line {line_number}")
        seen_epochs.add(epoch)
        records.append({"seed": seed, "epoch": epoch, "val/pearsonr": metric_value})

    if not records:
        raise RuntimeError(f"strict validation history is empty: {path}")
    return records


def _resolve_selected_validation(
    checkpoint_path: Path,
    validation_history_path: Path,
    *,
    seed: int,
) -> dict[str, int | float]:
    """Bind the checkpoint's raw epoch to exactly one validation record."""

    import torch

    if not checkpoint_path.is_file():
        raise RuntimeError(f"strict selected checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("strict checkpoint payload is not a mapping")
    raw_epoch = checkpoint.get("epoch")
    if isinstance(raw_epoch, bool) or not isinstance(raw_epoch, int) or raw_epoch < 0:
        raise RuntimeError("strict checkpoint does not contain a valid integer epoch")

    selected_epoch = raw_epoch + 1
    records = _read_validation_history(validation_history_path, seed=seed)
    matches = [row for row in records if row["epoch"] == selected_epoch]
    if len(matches) != 1:
        raise RuntimeError("strict selected epoch must match exactly one validation record")
    return {
        "checkpoint_epoch_zero_based": raw_epoch,
        "selected_epoch": selected_epoch,
        "selected_val_pearsonr": matches[0]["val/pearsonr"],
    }


def _freeze_provenance_snapshot(source_path: Path) -> Path:
    """Create an immutable per-run snapshot of a mutable upstream artifact."""

    if not source_path.is_file():
        raise RuntimeError(f"strict provenance file is missing: {source_path}")
    snapshot_path = source_path.with_name("strict_provenance_config.yaml")
    data = source_path.read_bytes()
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path.is_file():
        if snapshot_path.read_bytes() != data:
            raise RuntimeError(f"strict provenance snapshot differs: {snapshot_path}")
        return snapshot_path

    temporary_path = snapshot_path.with_name(
        f".{snapshot_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, snapshot_path)
            _fsync_directory(snapshot_path.parent)
        except FileExistsError as error:
            if not snapshot_path.is_file() or snapshot_path.read_bytes() != data:
                raise RuntimeError(f"strict provenance snapshot differs: {snapshot_path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)
    return snapshot_path


def _build_strict_selection_record(
    *,
    checkpoint_path: Path,
    official_config_path: Path,
    manifest_path: Path | None = None,
    provenance_path: Path | None = None,
    data_mode: str = "manifest",
    timeline_count: int | None = None,
    validation_history_path: Path,
    selection_monitor: str,
    selection_mode: str,
    seed: int,
    head_variant: str,
    strict_final_test: bool,
    sha256_file: Any,
) -> dict[str, Any]:
    """Build the immutable provenance record without writing it."""

    if selection_monitor != "val/pearsonr" or selection_mode != "max":
        raise RuntimeError("strict selection must monitor val/pearsonr in max mode")
    if data_mode not in {"manifest", "full"}:
        raise RuntimeError(f"unsupported strict data mode: {data_mode!r}")
    selected = _resolve_selected_validation(checkpoint_path, validation_history_path, seed=seed)
    immutable_config_path = _freeze_provenance_snapshot(official_config_path)
    if not immutable_config_path.is_file():
        raise RuntimeError(f"strict provenance file is missing: {immutable_config_path}")

    if data_mode == "manifest":
        if manifest_path is None:
            raise RuntimeError("manifest strict selection requires manifest_path")
        if provenance_path is None:
            provenance_path = manifest_path
        if not manifest_path.is_file():
            raise RuntimeError(f"strict provenance file is missing: {manifest_path}")
    else:
        if manifest_path is not None:
            raise RuntimeError("full-data strict selection must not include manifest_path")
        if provenance_path is None:
            raise RuntimeError("full-data strict selection requires provenance_path")
        if isinstance(timeline_count, bool) or not isinstance(timeline_count, int) or timeline_count < 1:
            raise RuntimeError("full-data strict selection requires a positive timeline_count")
    if not provenance_path.is_file():
        raise RuntimeError(f"strict provenance file is missing: {provenance_path}")

    manifest_sha256 = sha256_file(manifest_path) if manifest_path is not None else None
    return {
        "evaluation_protocol": "strict",
        "data_mode": data_mode,
        "timeline_count": timeline_count,
        "selection_monitor": selection_monitor,
        "selection_mode": selection_mode,
        "seed": int(seed),
        "head_variant": head_variant,
        **selected,
        "strict_final_test": bool(strict_final_test),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "official_config_path": str(immutable_config_path.resolve()),
        "official_config_sha256": sha256_file(immutable_config_path),
        "manifest_path": str(manifest_path.resolve()) if manifest_path is not None else None,
        "manifest_sha256": manifest_sha256,
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": sha256_file(provenance_path),
        "validation_history_path": str(validation_history_path.resolve()),
        "validation_history_sha256": sha256_file(validation_history_path),
        "test_status": "sealed" if strict_final_test else "withheld",
    }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_bytes_atomic(
    path: Path,
    data: bytes,
    *,
    remove_final_on_failure: bool = False,
) -> None:
    """Replace one file atomically and clean up incomplete evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        if remove_final_on_failure:
            path.unlink(missing_ok=True)
        raise


def _replace_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON object used as late-bound run metadata."""

    data = (json.dumps(payload, indent=2, default=str) + "\n").encode("utf-8")
    _replace_bytes_atomic(path, data)


def _publish_json_create_if_absent(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish JSON atomically without replacing an existing evidence file."""

    data = _canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
            _fsync_directory(path.parent)
        except FileExistsError as error:
            if not path.is_file() or path.read_bytes() != data:
                raise RuntimeError(f"existing strict evidence differs: {path}") from error
            return
    finally:
        temp_path.unlink(missing_ok=True)


def _create_test_start_marker(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a durable, exclusive marker before consuming test data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_json_bytes(payload)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f"strict test marker already exists: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        # The marker intentionally remains when a write is interrupted: its
        # existence conservatively consumes the one permitted test access.
        raise


def _create_test_completed_marker(path: Path, payload: Mapping[str, Any]) -> None:
    """Record successful post-test integrity verification exactly once."""

    _publish_json_create_if_absent(path, payload)


def _extract_official_test_pearson(results: Sequence[Mapping[str, Any]]) -> float:
    """Extract only the pinned official ``test/pearsonr`` result."""

    if len(results) != 1:
        raise RuntimeError("strict final test requires exactly one official result")
    result = results[0]
    if "test/pearsonr" not in result:
        raise RuntimeError("official strict result is missing test/pearsonr")
    value = result["test/pearsonr"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("official test/pearsonr must be numeric")
    pearson = float(value)
    if not math.isfinite(pearson):
        raise RuntimeError("official test/pearsonr must be finite")
    return pearson


def _run_strict_test_phase(
    *,
    original_test: Any,
    experiment: Any,
    loaders: Mapping[str, Any],
    best_model_path: str,
    selection_path: Path,
    test_started_path: Path,
    test_completed_path: Path,
    selection_record: Mapping[str, Any],
    strict_final_test: bool,
    invocation_key: int,
    test_invocations: set[int],
    sha256_file: Any,
) -> dict[str, Any]:
    """Seal selection and optionally consume the one permitted test pass."""

    run_dir = selection_path.parent
    legacy_metrics_path = run_dir / "epoch_test_metrics.jsonl"
    if legacy_metrics_path.exists():
        raise RuntimeError(
            "strict evaluation cannot reuse a directory with legacy test metrics: "
            f"{legacy_metrics_path}"
        )
    prior_prediction_dirs = [
        path for path in run_dir.rglob("test_predictions") if path.is_dir()
    ]
    if prior_prediction_dirs:
        raise RuntimeError(
            "strict evaluation cannot reuse a directory with prior test predictions: "
            f"{prior_prediction_dirs[0]}"
        )
    _publish_json_create_if_absent(selection_path, selection_record)
    if not strict_final_test:
        return {}
    if invocation_key in test_invocations:
        raise RuntimeError("strict experiment was already evaluated")

    checkpoint_path = Path(best_model_path)
    expected_checkpoint_sha256 = selection_record.get("checkpoint_sha256")
    if not isinstance(expected_checkpoint_sha256, str):
        raise RuntimeError("strict selection is missing checkpoint_sha256")
    test_started_payload = {
        "selection_sha256": sha256_file(selection_path),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "test_evaluations": 1,
    }
    _create_test_start_marker(test_started_path, test_started_payload)
    test_invocations.add(invocation_key)

    result = original_test(experiment, loaders, best_model_path)
    if not isinstance(result, Mapping):
        raise RuntimeError("official strict test result is not a mapping")
    test_pearson = _extract_official_test_pearson([result])
    actual_checkpoint_sha256 = sha256_file(checkpoint_path)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise RuntimeError("strict checkpoint hash changed during test")

    _create_test_completed_marker(
        test_completed_path,
        {
            "selection_sha256": sha256_file(selection_path),
            "checkpoint_sha256_after_test": actual_checkpoint_sha256,
            "test_pearsonr": test_pearson,
            "test_evaluations": 1,
        },
    )
    return dict(result)


def _selected_validation_checkpoint_epoch(
    results: Sequence[Mapping[str, Any]],
) -> int | None:
    """Select an epoch only from explicitly recorded validation Pearson values."""

    candidates: list[tuple[float, int]] = []
    for result in results:
        records = result.get("epoch_metrics")
        if not isinstance(records, (list, tuple)):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            metric = record.get("val/pearsonr", record.get("pearsonr"))
            epoch = record.get("epoch")
            if isinstance(metric, bool) or isinstance(epoch, bool):
                continue
            if not isinstance(metric, (int, float)) or not isinstance(epoch, int):
                continue
            if math.isfinite(float(metric)):
                candidates.append((float(metric), epoch))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _head_metadata(
    reve: Any,
    *,
    head_variant: str,
    data_mode: str = "manifest",
    manifest_path: Path | None = None,
    manifest_digest: str | None = None,
    provenance_path: Path | None = None,
    provenance_digest: str | None = None,
    timeline_count: int | None = None,
    rows: int | None = None,
    seeds: Sequence[int],
    launch_command: str,
) -> dict[str, Any]:
    """Build stable run metadata shared by every head variant."""

    query_initialization = {
        "mean_linear": "neuralbench_default",
        "mean_linear_copy": "not_applicable",
        "mean_linear_detached": "not_applicable",
        "mean_linear_warmup": "not_applicable",
        "mean_linear_gradient_scaled": "not_applicable",
        "mean_linear_probe_scaled": "not_applicable",
        "mean_anchor": "train_dummy_final_token_mean",
        "mean_residual": "train_dummy_final_token_mean",
        "mean_vector_anchor": "train_dummy_final_token_mean",
        "mean_mlp_residual": "not_applicable",
        "mean_stats_residual": "not_applicable",
        "mean_stats_residual_detached": "not_applicable",
        "mean_stats_residual_gradient_scaled": "not_applicable",
        "mean_stats_probe_scaled": "not_applicable",
        "mean_stats_attention_residual": "train_dummy_final_token_mean",
        "mean_attention_gated": "train_dummy_final_token_mean",
        "global_stats_residual": "not_applicable",
        "mean_rich_stats_residual": "not_applicable",
        "grouped_rich_stats_shrinkage": "not_applicable",
        "grouped_stats_shared_gate": "not_applicable",
        "temporal_pyramid_stats": "not_applicable",
        "mean_covariance_residual": "not_applicable",
        "last_avg": "upstream_random_unused",
        "last_tuned": "train_dummy_final_token_mean",
        "last": "upstream_random",
        "all": "upstream_random",
    }[head_variant]
    is_default_head = head_variant == "mean_linear"
    is_local_head = head_variant in {"mean_linear_copy", "mean_linear_detached", "mean_linear_warmup", "mean_linear_gradient_scaled", "mean_linear_probe_scaled", "mean_anchor", "mean_residual", "mean_vector_anchor", "mean_mlp_residual", "mean_stats_residual", "mean_stats_residual_detached", "mean_stats_residual_gradient_scaled", "mean_stats_probe_scaled", "mean_stats_attention_residual", "mean_attention_gated", "global_stats_residual", "mean_rich_stats_residual", "grouped_rich_stats_shrinkage", "grouped_stats_shared_gate", "temporal_pyramid_stats", "mean_covariance_residual"}
    if data_mode not in {"manifest", "full"}:
        raise ValueError(f"unsupported data mode: {data_mode!r}")
    if data_mode == "manifest" and manifest_path is not None and provenance_path is None:
        provenance_path = manifest_path
        provenance_digest = manifest_digest
    metadata: dict[str, Any] = {
        "head_variant": head_variant,
        "head_source": (
            "neuralbench_default"
            if is_default_head
            else "local_mean_anchor"
            if head_variant == "mean_anchor"
            else "local_mean_residual"
            if head_variant == "mean_residual"
            else "local_mean_vector_anchor"
            if head_variant == "mean_vector_anchor"
            else "local_mean_mlp_residual"
            if head_variant == "mean_mlp_residual"
            else "local_mean_stats_residual"
            if head_variant == "mean_stats_residual"
            else "local_mean_stats_residual_detached"
            if head_variant == "mean_stats_residual_detached"
            else "local_mean_stats_residual_gradient_scaled"
            if head_variant == "mean_stats_residual_gradient_scaled"
            else "local_mean_linear_detached"
            if head_variant == "mean_linear_detached"
            else "local_mean_linear_warmup"
            if head_variant == "mean_linear_warmup"
            else "local_mean_linear_gradient_scaled"
            if head_variant == "mean_linear_gradient_scaled"
            else "local_mean_linear_probe_scaled"
            if head_variant == "mean_linear_probe_scaled"
            else "local_mean_stats_probe_scaled"
            if head_variant == "mean_stats_probe_scaled"
            else "local_mean_stats_attention_residual"
            if head_variant == "mean_stats_attention_residual"
            else "local_mean_attention_gated"
            if head_variant == "mean_attention_gated"
            else "local_global_stats_residual"
            if head_variant == "global_stats_residual"
            else "local_mean_rich_stats_residual"
            if head_variant == "mean_rich_stats_residual"
            else "local_grouped_rich_stats_shrinkage"
            if head_variant == "grouped_rich_stats_shrinkage"
            else "local_grouped_stats_shared_gate"
            if head_variant == "grouped_stats_shared_gate"
            else "local_temporal_pyramid_stats"
            if head_variant == "temporal_pyramid_stats"
            else "local_mean_covariance_residual"
            if head_variant == "mean_covariance_residual"
            else "local_mean_linear_copy"
            if is_local_head
            else "upstream_reve"
        ),
        "head_dropout": 0.0,
        "head_query_initialization": query_initialization,
        "head_linear_initialization": (
            "neuralbench_default"
            if is_default_head
            else "torch_nn_linear_default"
            if is_local_head
            else {
                "distribution": "truncated_normal",
                "std": reve.UPSTREAM_HEAD_INIT_STD,
                "cutoff": reve.UPSTREAM_HEAD_INIT_CUTOFF,
                "bias": 0.0,
            }
        ),
        "data_mode": data_mode,
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_sha256": manifest_digest,
        "provenance_path": str(provenance_path) if provenance_path is not None else None,
        "provenance_sha256": provenance_digest,
        "timeline_count": timeline_count if timeline_count is not None else rows,
        "rows": rows,
        "device": "eeg",
        "seeds": list(seeds),
        "launch_command": launch_command,
        "protocol": reve.PROTOCOL_CONTRACT,
        "runtime": reve.runtime_metadata(),
    }
    if head_variant == "mean_linear_detached":
        metadata.update(
            {
                "head_architecture": "mean_linear_detached_encoder",
                "query_initialization": "not_applicable",
                "encoder_gradient": "detached",
                "normalization": "none",
            }
        )
    if head_variant == "mean_linear_warmup":
        metadata.update(
            {
                "head_architecture": "mean_linear_zero_gate_residual_warmup",
                "query_initialization": "not_applicable",
                "residual_initialization": "torch_nn_linear_default",
                "gate_initialization": 0.0,
                "baseline_encoder_gradient": "detached",
                "residual_encoder_gradient": "enabled_after_gate_update",
                "normalization": "none",
            }
        )
    if head_variant == "mean_linear_gradient_scaled":
        metadata.update(
            {
                "head_architecture": "mean_linear_gradient_scaled",
                "query_initialization": "not_applicable",
                "encoder_gradient_scale": 0.1,
                "normalization": "none",
            }
        )
    if head_variant == "mean_linear_probe_scaled":
        metadata.update(
            {
                "head_architecture": "mean_linear_probe_gradient_scaled",
                "query_initialization": "not_applicable",
                "encoder_gradient_scale": 0.1,
                "probe_gradient_scale": 10.0,
                "normalization": "none",
            }
        )
    if head_variant == "mean_anchor":
        metadata.update(
            {
                "head_architecture": "mean_anchor_train_dummy_query_residual",
                "query_initialization": "train_dummy_final_token_mean",
                "gamma_initialization": 0.0,
                "normalization": "none",
            }
        )
    if head_variant == "mean_residual":
        metadata.update(
            {
                "head_architecture": "mean_residual_zero_correction_query_attention",
                "query_initialization": "train_dummy_final_token_mean",
                "correction_initialization": "zero",
                "normalization": "none",
            }
        )
    if head_variant == "mean_vector_anchor":
        metadata.update(
            {
                "head_architecture": "mean_vector_anchor_train_dummy_query_residual",
                "query_initialization": "train_dummy_final_token_mean",
                "gamma_initialization": 0.0,
                "normalization": "none",
            }
        )
    if head_variant == "mean_mlp_residual":
        metadata.update(
            {
                "head_architecture": "mean_mlp_zero_correction",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero",
                "normalization": "none",
            }
        )
    if head_variant == "mean_stats_residual":
        metadata.update(
            {
                "head_architecture": "mean_stats_zero_correction",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero",
                "correction_features": "per_feature_std_and_range",
                "correction_scale": 0.5,
                "normalization": "none",
            }
        )
    if head_variant == "mean_stats_residual_detached":
        metadata.update(
            {
                "head_architecture": "mean_stats_zero_correction_detached_statistics",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero",
                "correction_features": "per_feature_std_and_range",
                "correction_scale": 0.5,
                "correction_backbone_gradient": "detached",
                "normalization": "none",
            }
        )
    if head_variant == "mean_stats_residual_gradient_scaled":
        metadata.update(
            {
                "head_architecture": "mean_stats_zero_correction_gradient_scaled",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero",
                "correction_features": "per_feature_std_and_range",
                "correction_scale": 0.5,
                "encoder_gradient_scale": 0.5,
                "correction_backbone_gradient": "detached",
                "normalization": "none",
            }
        )
    if head_variant == "mean_stats_probe_scaled":
        metadata.update(
            {
                "head_architecture": "mean_stats_zero_correction_probe_gradient_scaled",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero",
                "correction_features": "per_feature_std_and_range",
                "correction_scale": 0.5,
                "encoder_gradient_scale": 1.0,
                "probe_gradient_scale": 2.0,
                "correction_backbone_gradient": "detached",
                "normalization": "none",
            }
        )
    if head_variant == "mean_stats_attention_residual":
        metadata.update(
            {
                "head_architecture": "mean_stats_attention_zero_correction",
                "query_initialization": "train_dummy_final_token_mean",
                "correction_initialization": "zero",
                "attention_correction_scale": 0.25,
                "stats_correction_scale": 0.5,
                "correction_features": "query_attention_residual_plus_per_feature_std_and_range",
                "normalization": "none",
            }
        )
    if head_variant == "mean_attention_gated":
        metadata.update(
            {
                "head_architecture": "mean_linear_detached_attention_scalar_gate",
                "query_initialization": "train_dummy_final_token_mean",
                "correction_initialization": "small_normal",
                "correction_scale": 0.25,
                "gamma_initialization": 0.0,
                "correction_encoder_gradient": "detached",
                "normalization": "none",
            }
        )
    if head_variant == "global_stats_residual":
        metadata.update(
            {
                "head_architecture": "mean_global_stats_zero_correction",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero",
                "correction_features": "global_std_range_mad_and_mean_abs",
                "correction_scale": 0.5,
                "correction_backbone_gradient": "enabled",
                "normalization": "none",
            }
        )
    if head_variant == "mean_rich_stats_residual":
        metadata.update(
            {
                "head_architecture": "mean_rich_stats_zero_correction",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero",
                "correction_features": "per_feature_std_range_mad_and_mean_abs",
                "correction_scale": 0.5,
                "correction_backbone_gradient": "enabled",
                "normalization": "none",
            }
        )
    if head_variant == "grouped_rich_stats_shrinkage":
        metadata.update(
            {
                "head_architecture": "mean_grouped_rich_stats_zero_gate_shrinkage",
                "query_initialization": "not_applicable",
                "statistic_groups": ["std", "range", "mad", "mean_abs"],
                "gate_parameterization": "direct_scalar",
                "gate_initialization": 0.0,
                "projection_initialization": (
                    "linspace_-1_1_roll_group_plus_row_alternating_sign_l2_normalized_zero_bias"
                ),
                "projection_shape": "D_to_D",
                "parameter_count_formula": "D*n_outputs+n_outputs+4*(D*D+D)+4",
                "correction_scale": 0.5,
                "correction_backbone_gradient": "enabled",
                "normalization": "none",
            }
        )
    if head_variant == "grouped_stats_shared_gate":
        metadata.update(
            {
                "head_architecture": "mean_grouped_stats_shared_gate",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero",
                "statistic_groups": ["std", "range", "mad", "mean_abs"],
                "gate_parameterization": "shared_scalar",
                "gate_initialization": 0.0,
                "projection_initialization": (
                    "linspace_-1_1_roll_group_plus_row_alternating_sign_l2_normalized_zero_bias"
                ),
                "projection_shape": "D_to_D",
                "parameter_count_formula": "D*n_outputs+n_outputs+4*(D*D+D)+1",
                "correction_scale": 0.5,
                "correction_backbone_gradient": "enabled",
                "normalization": "none",
            }
        )
    if head_variant == "temporal_pyramid_stats":
        metadata.update(
            {
                "head_architecture": "mean_temporal_pyramid_stats_low_rank_residual",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero_via_up_factor",
                "segments": 2,
                "statistics": ["std", "range", "mad", "mean_abs"],
                "correction_rank": 8,
                "low_rank_parameterization": "down_then_up",
                "parameter_count_formula": "D*n_outputs+n_outputs+(8*D)*8+D*8",
                "correction_scale": 0.5,
                "correction_backbone_gradient": "enabled",
                "token_order_contract": "contiguous_ordered_segments",
                "normalization": "none",
            }
        )
    if head_variant == "mean_covariance_residual":
        metadata.update(
            {
                "head_architecture": "mean_diagonal_covariance_low_rank_residual",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero_via_up_factor",
                "covariance_mode": "diagonal",
                "covariance_features": "diagonal_sample_variance",
                "projection_rank": 4,
                "low_rank_parameterization": "down_then_up",
                "parameter_count_formula": "D*n_outputs+n_outputs+2*D*4",
                "correction_scale": 0.5,
                "correction_backbone_gradient": "enabled",
                "normalization": "none",
            }
        )
    if head_variant == "last_tuned":
        metadata.update(
            {
                "head_source": reve.LAST_TUNED_HEAD_SOURCE,
                "head_architecture": reve.LAST_TUNED_HEAD_ARCHITECTURE,
                "protocol_class": reve.LAST_TUNED_PROTOCOL_CLASS,
                "residual_initial_alpha": reve.LAST_TUNED_INITIAL_ALPHA,
                "query_initialization": query_initialization,
                "base_learning_rate": reve.LAST_TUNED_BASE_LR,
                "query_learning_rate": reve.LAST_TUNED_QUERY_LR,
                "optimizer": "AdamW",
                "weight_decay": reve.LAST_TUNED_WEIGHT_DECAY,
                "scheduler": "OneCycleLR",
                "scheduler_max_lr": list(reve.LAST_TUNED_SCHEDULER_MAX_LR),
                "scheduler_pct_start": reve.LAST_TUNED_SCHEDULER_PCT_START,
                "scheduler_anneal_strategy": "cos",
                "scheduler_div_factor": reve.LAST_TUNED_SCHEDULER_DIV_FACTOR,
                "scheduler_final_div_factor": reve.LAST_TUNED_SCHEDULER_FINAL_DIV_FACTOR,
                "scheduler_interval": "step",
                "scheduler_frequency": 1,
                "monitor": "val/pearsonr",
                "checkpoint_selection_monitor": "val/pearsonr",
                "test_pearsonr_role": "diagnostic_only",
            }
        )
    if not is_default_head and not is_local_head:
        metadata["head_source_lock"] = reve.source_lock_metadata()
    return metadata


def _append_evaluation_callback(
    trainer: Any,
    *,
    evaluation_protocol: str,
    epoch_metrics_path: Path,
    seed: int | None,
    loaders: Mapping[str, Any],
    hooks: Any,
) -> None:
    """Attach exactly one training-time metric callback for the protocol."""

    if evaluation_protocol == "strict":
        trainer.callbacks.append(
            hooks.EpochValidationMetrics(epoch_metrics_path, seed=seed)
        )
        return
    if evaluation_protocol != "legacy":
        raise ValueError(f"unsupported evaluation protocol: {evaluation_protocol!r}")
    if "test" not in loaders:
        raise RuntimeError("official test loader was not captured")
    trainer.callbacks.append(
        hooks.EpochTestPearson(loaders["test"], epoch_metrics_path, seed=seed)
    )


# ---------------------------------------------------------------------------
# Official run patch lifecycle
# ---------------------------------------------------------------------------


def _patch_official_components(
    manifest_path: Path | None,
    data_root: Path,
    epoch_metrics_path: Path,
    selection_path: Path,
    *,
    head_variant: str = "mean_linear",
    head_dropout: float = 0.0,
    seeds: Sequence[int] = (33,),
    evaluation_protocol: str = "strict",
    strict_final_test: bool = False,
    data_mode: str = "manifest",
    provenance_path: Path | None = None,
    timeline_count: int | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    final_results: list[dict[str, Any]] | None = None,
    hooks: Any,
) -> dict[str, Any]:
    """Install the fixed-manifest and optional upstream-head patches.

    The patch is deliberately scoped to one ``run_benchmark`` call.  The
    official package remains untouched after :func:`_restore_official_components`
    runs, which is important when several variants are evaluated in one Python
    process.
    """

    evaluation_protocol, strict_final_test = hooks.validate_evaluation_options(
        evaluation_protocol,
        strict_final_test=strict_final_test,
    )
    reve = hooks._load_reve_helpers()

    reve.validate_head_variant(head_variant)
    if head_dropout != 0.0:
        raise ValueError(f"the upstream REVE comparison fixes head dropout at 0.0; got {head_dropout}")
    resolved_seeds = hooks.validate_seeds(seeds)
    if data_mode not in {"manifest", "full"}:
        raise ValueError(f"unsupported data mode: {data_mode!r}")
    if data_mode == "manifest" and manifest_path is None:
        raise ValueError("manifest data mode requires manifest_path")
    if data_mode == "full" and manifest_path is not None:
        raise ValueError("full data mode must not receive manifest_path")
    if data_mode == "full" and provenance_path is None:
        raise ValueError("full data mode requires provenance_path")
    if data_mode == "full":
        assert provenance_path is not None
        provenance_path.unlink(missing_ok=True)

    from neuralbench.data import Data
    from neuralbench.main import Experiment
    from neuralfetch.studies import shirazi2024hbn

    timelines: list[dict[str, Any]] | None = None
    if data_mode == "manifest":
        assert manifest_path is not None
        timelines = hooks.load_manifest_timelines(manifest_path, data_root)

    original_iter_timelines = shirazi2024hbn.Shirazi2024Hbn.iter_timelines
    original_info = shirazi2024hbn.Shirazi2024Hbn._info
    original_prepare = Data.prepare
    original_test = Experiment._test
    original_setup_run = Experiment.setup_run
    original_prepare_pl_module = Experiment.prepare_pl_module
    original_setup_trainer = Experiment.setup_trainer

    def iter_manifest_timelines(_study: Any) -> Iterable[dict[str, Any]]:
        assert timelines is not None
        return iter(timelines)

    captured_loaders: dict[int, dict[str, Any]] = {}
    provenance_state_by_data: dict[int, dict[str, Any]] = {}
    patched_brain_modules: list[dict[str, Any]] = []
    tuning_metadata_by_experiment: dict[int, dict[str, Any]] = {}
    test_invocations: set[int] = set()

    def prepare_and_capture(data: Any) -> dict[str, Any]:
        loaders = original_prepare(data)
        captured_loaders[id(data)] = loaders
        if data_mode == "full":
            study = getattr(data, "study", None)
            actual_timelines = getattr(study, "_timelines", None)
            if actual_timelines is None:
                raise RuntimeError("full-data Data.prepare did not expose study._timelines")
            normalized = hooks._canonical_full_data_timelines(actual_timelines)
            payload, raw = hooks._full_data_provenance_payload(
                data_root=data_root,
                timelines=normalized,
            )
            assert provenance_path is not None
            try:
                _replace_bytes_atomic(
                    provenance_path,
                    raw,
                    remove_final_on_failure=True,
                )
                final_bytes = provenance_path.read_bytes()
                provenance_digest = hooks._sha256_bytes(final_bytes)
            except BaseException:
                provenance_path.unlink(missing_ok=True)
                raise
            state = {
                "data_mode": "full",
                "timeline_count": payload["timeline_count"],
                "provenance_path": str(provenance_path.resolve()),
                "provenance_sha256": provenance_digest,
            }
            provenance_state_by_data[id(data)] = state
        return loaders

    def test_and_capture(
        self: Any,
        loaders: dict[str, Any],
        best_model_path: str | None,
    ) -> dict[str, Any]:
        if evaluation_protocol == "legacy":
            result = original_test(self, loaders, best_model_path)
            if final_results is not None:
                final_results.append(
                    hooks._capture_test_result(
                        result,
                        head_variant=head_variant,
                        experiment_id=id(self),
                        tuning_metadata_by_experiment=tuning_metadata_by_experiment,
                    )
                )
            return result

        uid_folder = self.infra.uid_folder()
        if uid_folder is None:
            raise RuntimeError("strict evaluation requires an official uid folder")
        if best_model_path is None:
            raise RuntimeError("strict evaluation requires a selected checkpoint")
        source_state = provenance_state_by_data.get(id(self.data))
        if data_mode == "full" and source_state is None:
            raise RuntimeError("full-data strict evaluation has no post-prepare provenance")
        selection_manifest_path = manifest_path if data_mode == "manifest" else None
        selection_provenance_path = (
            Path(source_state["provenance_path"])
            if source_state is not None
            else manifest_path
        )
        selection_timeline_count = (
            int(source_state["timeline_count"])
            if source_state is not None
            else len(timelines or [])
        )
        selection_record = hooks._build_strict_selection_record(
            checkpoint_path=Path(best_model_path),
            official_config_path=uid_folder / "config.yaml",
            manifest_path=selection_manifest_path,
            provenance_path=selection_provenance_path,
            data_mode=data_mode,
            timeline_count=selection_timeline_count,
            validation_history_path=epoch_metrics_path,
            selection_monitor="val/pearsonr",
            selection_mode="max",
            seed=int(getattr(self, "seed", 0)),
            head_variant=head_variant,
            strict_final_test=strict_final_test,
            sha256_file=hooks._sha256_file,
        )
        result = hooks._run_strict_test_phase(
            original_test=original_test,
            experiment=self,
            loaders=loaders,
            best_model_path=best_model_path,
            selection_path=selection_path,
            test_started_path=selection_path.parent / "test_started.json",
            test_completed_path=selection_path.parent / "test_completed.json",
            selection_record=selection_record,
            strict_final_test=strict_final_test,
            invocation_key=id(self),
            test_invocations=test_invocations,
            sha256_file=hooks._sha256_file,
        )
        if strict_final_test and final_results is not None:
            final_results.append(
                hooks._capture_test_result(
                    result,
                    head_variant=head_variant,
                    experiment_id=id(self),
                    tuning_metadata_by_experiment=tuning_metadata_by_experiment,
                )
            )
        return result

    def setup_with_evaluation_callbacks(self: Any, is_test: bool = False) -> Any:
        trainer = original_setup_trainer(self, is_test=is_test)
        if not is_test:
            loaders = captured_loaders.get(id(self.data))
            if loaders is None:
                raise RuntimeError("official loaders were not captured")
            _append_evaluation_callback(
                trainer,
                evaluation_protocol=evaluation_protocol,
                epoch_metrics_path=epoch_metrics_path,
                seed=getattr(self, "seed", None),
                loaders=loaders,
                hooks=hooks,
            )
        return trainer

    def setup_with_metadata(self: Any) -> Any:
        # The standard REVE YAML wrapper is mean-pooling plus a linear probe.
        # For upstream variants, replace only that downstream config; the
        # already-built official NtReve encoder still comes from NeuralBench.
        if head_variant != "mean_linear":
            hooks._set_frozen_experiment_field(
                self,
                "downstream_model_wrapper",
                reve.make_upstream_reve_wrapper(variant=head_variant, dropout=head_dropout),
            )
        if head_variant == "last_tuned":
            # NeuralBench expresses the actual checkpoint criterion through
            # ``trainer_config``.  Record its resolved tuning counterpart on
            # this run instance so the separate tuning validator can reject a
            # relabeling of the diagnostic test callback as a selector.
            hooks._set_frozen_experiment_field(
                self,
                "checkpoint_selection",
                {
                    "monitor": "val/pearsonr",
                    "mode": "max",
                    "test_pearsonr_role": "diagnostic_only",
                },
            )
        hooks._set_frozen_experiment_field(self, "save_test_predictions", True)
        # Keep the selected checkpoint and raw prediction cache available for
        # post-run hashing/export. This does not affect training or selection.
        hooks._set_frozen_experiment_field(self, "delete_checkpoints_on_exit", False)
        result = original_setup_run(self)
        uid_folder = self.infra.uid_folder()
        if uid_folder is not None:
            payload = dict(run_metadata or {})
            payload.update(
                {
                    "head_variant": head_variant,
                    "head_dropout": float(head_dropout),
                    "seed": int(self.seed),
                    "data_seed": hooks._get_attr_or_key(self.data, "seed"),
                    "protocol": reve.PROTOCOL_CONTRACT,
                    "evaluation_protocol": evaluation_protocol,
                    "strict_final_test": bool(strict_final_test),
                    "test_access_policy": (
                        "single_use_predeclared"
                        if strict_final_test
                        else "withheld"
                        if evaluation_protocol == "strict"
                        else "epoch_diagnostic"
                    ),
                }
            )
            _replace_json_atomic(uid_folder / "run_metadata.json", payload)
        return result

    def persist_tuning_metadata(self: Any, metadata: Mapping[str, Any]) -> None:
        """Merge late-bound query and optimizer details into run metadata."""

        uid_folder = self.infra.uid_folder()
        if uid_folder is None:
            return
        path = uid_folder / "run_metadata.json"
        payload: dict[str, Any] = {}
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, Mapping):
                raise ValueError("run_metadata.json must contain a JSON object")
            payload.update(loaded)
        payload.update(metadata)
        _replace_json_atomic(path, payload)

    def persist_provenance_metadata(self: Any) -> None:
        if data_mode != "full":
            return
        state = provenance_state_by_data.get(id(self.data))
        if state is None:
            raise RuntimeError("full-data experiment has no post-prepare provenance")
        uid_folder = self.infra.uid_folder()
        if uid_folder is None:
            raise RuntimeError("full-data run has no official uid folder")
        path = uid_folder / "run_metadata.json"
        payload: dict[str, Any] = {}
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, Mapping):
                raise ValueError("run_metadata.json must contain a JSON object")
            payload.update(loaded)
        payload.update(state)
        _replace_json_atomic(path, payload)

    def prepare_with_protocol(
        self: Any,
        train_loader: Any,
        val_loader: Any = None,
    ) -> Any:
        result = original_prepare_pl_module(self, train_loader, val_loader)
        persist_provenance_metadata(self)
        if head_variant in {"grouped_rich_stats_shrinkage", "grouped_stats_shared_gate", "temporal_pyramid_stats", "mean_covariance_residual"}:
            brain_module = getattr(self, "_brain_module", None)
            grouped_head = None
            expected_class_name = {
                "grouped_rich_stats_shrinkage": "GroupedRichStatsShrinkageHead",
                "grouped_stats_shared_gate": "GroupedStatsSharedGateHead",
                "temporal_pyramid_stats": "TemporalPyramidStatsResidualHead",
                "mean_covariance_residual": "MeanCovarianceResidualHead",
            }[head_variant]
            if brain_module is not None and hasattr(brain_module, "modules"):
                grouped_head = next(
                    (
                        module
                        for module in brain_module.modules()
                        if module.__class__.__name__ == expected_class_name
                    ),
                    None,
                )
            if grouped_head is None or not callable(getattr(grouped_head, "metadata", None)):
                raise RuntimeError(f"{head_variant} model did not expose its head metadata")
            head_metadata = dict(grouped_head.metadata())
            head_metadata["parameter_count"] = sum(
                parameter.numel() for parameter in grouped_head.parameters()
            )
            persist_tuning_metadata(
                self,
                {
                    "head_metadata": head_metadata,
                    "head_parameter_count": head_metadata["parameter_count"],
                },
            )
        if head_variant == "last_tuned":
            brain_module = getattr(self, "_brain_module", None)
            model = getattr(brain_module, "model", None)
            tuning_model = reve._resolve_last_tuned_model(model)
            query_metadata = getattr(getattr(tuning_model, "head", None), "tuning_metadata", None)
            if not isinstance(query_metadata, Mapping):
                raise RuntimeError("last_tuned prepared model did not expose tuning metadata")
            optimizer_metadata = reve.last_tuned_optimizer_metadata(tuning_model)
            _patch_last_tuned_configure_optimizers(brain_module, patched_brain_modules, hooks=hooks)
            reve.validate_last_tuned_protocol(head_variant, experiment=self, optimizer_config=optimizer_metadata)
            tuning_metadata = hooks._last_tuned_report_metadata(query_metadata=query_metadata, optimizer_config=optimizer_metadata)
            tuning_metadata_by_experiment[id(self)] = tuning_metadata
            persist_tuning_metadata(self, tuning_metadata)
        else:
            loaders = captured_loaders.get(id(self.data))
            reve.validate_official_protocol(self, loaders=loaders, n_total_params=self._n_total_params, n_trainable_params=self._n_trainable_params)
        return result

    # NeuralBench's CLI and experiment_config modules each keep a local alias
    # to the YAML loader. Patch both so a task-specific grid cannot silently
    # reintroduce the default (33, 34, 35) seed expansion.
    import neuralbench.cli as cli
    import neuralbench.experiment_config as experiment_config

    original_cli_load_yaml_config = cli.load_yaml_config
    original_experiment_load_yaml_config = experiment_config.load_yaml_config

    originals = {
        "data_mode": data_mode,
        "patched_study_source": data_mode == "manifest",
        "iter_timelines": original_iter_timelines,
        "info": original_info,
        "prepare": original_prepare,
        "test": original_test,
        "setup_run": original_setup_run,
        "prepare_pl_module": original_prepare_pl_module,
        "patched_brain_modules": patched_brain_modules,
        "setup_trainer": original_setup_trainer,
        "cli_loader": (cli, original_cli_load_yaml_config),
        "experiment_loader": (
            experiment_config,
            original_experiment_load_yaml_config,
        ),
    }

    def load_seed_grid(path: Path, *args: Any, **kwargs: Any) -> Any:
        if Path(path).name == "grid.yaml":
            return {"seed": list(resolved_seeds)}
        return original_cli_load_yaml_config(path, *args, **kwargs)

    def load_seed_grid_for_experiment_config(
        path: Path, *args: Any, **kwargs: Any
    ) -> Any:
        if Path(path).name == "grid.yaml":
            return {"seed": list(resolved_seeds)}
        return original_experiment_load_yaml_config(path, *args, **kwargs)

    try:
        if data_mode == "manifest":
            assert timelines is not None
            shirazi2024hbn.Shirazi2024Hbn.iter_timelines = iter_manifest_timelines
            if original_info is not None:
                shirazi2024hbn.Shirazi2024Hbn._info = original_info.model_copy(update={"num_timelines": len(timelines)})
        Data.prepare = prepare_and_capture
        Experiment._test = test_and_capture
        Experiment.setup_trainer = setup_with_evaluation_callbacks
        Experiment.setup_run = setup_with_metadata
        Experiment.prepare_pl_module = prepare_with_protocol
        cli.load_yaml_config = load_seed_grid
        experiment_config.load_yaml_config = load_seed_grid_for_experiment_config
    except BaseException as active_error:
        try:
            hooks._restore_official_components(originals)
        except BaseException as cleanup_error:
            active_error.add_note(f"patch setup cleanup failure: {cleanup_error!r}")
        raise
    return originals


def _restore_official_components(originals: Mapping[str, Any], *, restore_tuned: Any) -> None:
    from neuralbench.data import Data
    from neuralbench.main import Experiment
    from neuralfetch.studies import shirazi2024hbn

    cli, original_cli_loader = originals["cli_loader"]
    experiment_config, original_experiment_loader = originals["experiment_loader"]
    restoration_errors: list[tuple[str, BaseException]] = []

    def attempt(label: str, restore: Any) -> None:
        try:
            restore()
        except BaseException as error:
            restoration_errors.append((label, error))

    if originals.get("patched_study_source", True):
        attempt("Shirazi2024Hbn.iter_timelines", lambda: setattr(shirazi2024hbn.Shirazi2024Hbn, "iter_timelines", originals["iter_timelines"]))
        attempt("Shirazi2024Hbn._info", lambda: setattr(shirazi2024hbn.Shirazi2024Hbn, "_info", originals["info"]))
    attempt("Data.prepare", lambda: setattr(Data, "prepare", originals["prepare"]))
    attempt("Experiment.setup_run", lambda: setattr(Experiment, "setup_run", originals["setup_run"]))
    attempt("Experiment._test", lambda: setattr(Experiment, "_test", originals["test"]))
    attempt("Experiment.prepare_pl_module", lambda: setattr(Experiment, "prepare_pl_module", originals["prepare_pl_module"]))
    attempt("Experiment.setup_trainer", lambda: setattr(Experiment, "setup_trainer", originals["setup_trainer"]))
    attempt("neuralbench.cli.load_yaml_config", lambda: setattr(cli, "load_yaml_config", original_cli_loader))
    attempt("neuralbench.experiment_config.load_yaml_config", lambda: setattr(experiment_config, "load_yaml_config", original_experiment_loader))
    attempt("last_tuned.configure_optimizers", lambda: restore_tuned(originals.get("patched_brain_modules", [])))

    if restoration_errors:
        error = RuntimeError("official component restoration failed")
        for label, restoration_error in restoration_errors:
            error.add_note(f"{label}: {restoration_error!r}")
        raise error


def run_official_subset(
    *,
    manifest_path: Path | None = None,
    data_root: Path,
    epoch_metrics_path: Path,
    selection_path: Path,
    config_path: Path,
    head_variant: str = "mean_linear",
    head_dropout: float = 0.0,
    seeds: Sequence[int] = (33,),
    evaluation_protocol: str = "strict",
    strict_final_test: bool = False,
    data_mode: str = "manifest",
    provenance_path: Path | None = None,
    timeline_count: int | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    hooks: Any,
) -> list[dict[str, Any]]:
    """Run official REVE on the selected source and collect cached results."""

    os.environ["NEURALBENCH_CONFIG"] = str(config_path)
    final_results: list[dict[str, Any]] = []
    originals: Mapping[str, Any] | None = None
    benchmark_aggregator: Any = None
    original_aggregator_prepare: Any = None
    try:
        originals = hooks._patch_official_components(
            manifest_path,
            data_root,
            epoch_metrics_path,
            selection_path,
            head_variant=head_variant,
            head_dropout=head_dropout,
            seeds=seeds,
            evaluation_protocol=evaluation_protocol,
            strict_final_test=strict_final_test,
            data_mode=data_mode,
            provenance_path=provenance_path,
            timeline_count=timeline_count,
            run_metadata=run_metadata,
            final_results=final_results,
        )
        from neuralbench.main import BenchmarkAggregator

        benchmark_aggregator = BenchmarkAggregator
        original_aggregator_prepare = BenchmarkAggregator.prepare
        BenchmarkAggregator.prepare = _run_experiments_synchronously
        from neuralbench import run_benchmark

        run_benchmark(device="eeg", task="age", model="reve", force=True)
        # The public runner returns no result for a non-debug local run. The
        # official Experiment._test result is captured above instead. Avoid a
        # second ``plot_cached`` call: it reconstructs the canonical mean head
        # and collides with this run's custom upstream head UID.
        return final_results
    finally:
        active_error = sys.exc_info()[1]
        cleanup_errors: list[BaseException] = []
        if active_error is not None and data_mode == "full" and provenance_path is not None:
            try:
                provenance_path.unlink(missing_ok=True)
            except BaseException as error:
                cleanup_errors.append(error)
        if benchmark_aggregator is not None and original_aggregator_prepare is not None:
            try:
                benchmark_aggregator.prepare = original_aggregator_prepare
            except BaseException as error:
                cleanup_errors.append(error)
        if originals is not None:
            try:
                hooks._restore_official_components(originals)
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            if active_error is not None:
                for cleanup_error in cleanup_errors:
                    active_error.add_note(f"cleanup failure: {cleanup_error!r}")
            else:
                cleanup_error = RuntimeError("official REVE cleanup failed")
                for error in cleanup_errors:
                    cleanup_error.add_note(repr(error))
                raise cleanup_error


def _run_experiments_synchronously(aggregator: Any) -> None:
    """Run prepared NeuralBench experiments in the current process.

    NeuralBench's public non-debug runner submits experiments to an ``exca``
    job array and returns before those workers finish when no scheduler is
    available. The fixed-manifest harness needs the worker-local monkeypatches
    above to survive into the actual experiment, so execute each prepared
    experiment directly while retaining the canonical (non-debug) config.
    """

    for experiment in aggregator.experiments:
        experiment.run()
