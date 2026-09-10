"""Lock-gated external inference for the ds006780 transfer experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from neurobench_age.core.evidence import source_tree_sha256
from neurobench_age.data.ds006780 import (
    Ds006780Error,
    build_target_free_qc,
    verify_ds006780_manifest,
)
from neurobench_age.pipelines.capacity_data_regime import (
    _head_directory_name,
    build_capacity_data_regime_summary_head,
    capacity_summary_from_tokens,
)
from neurobench_age.pipelines.frozen_probe import (
    encoder_state_sha256,
    load_reve_encoder,
)
from neurobench_age.pipelines.representation_materialization import (
    extract_frozen_representations_batched,
)
from neurobench_age.research.ds006780_external_lock import (
    SELECTED_HEADS,
    Ds006780ExternalLockError,
    load_ds006780_external_lock,
    select_ds006780_runs,
)
from neurobench_age.research.strict_json import (
    canonical_sha256,
    load_json_strict,
    reject_target_fields,
)
from neurobench_age.data.ds006780 import validate_external_config


class Ds006780ExternalError(RuntimeError):
    """Raised when ds006780 inference cannot satisfy the sealed lock."""


@dataclass(frozen=True)
class _LoadedHead:
    training_size: int
    head: str
    seed: int
    checkpoint_sha256: str
    model: nn.Module


def checkpoint_path_for_run(checkpoint_root: Path, run: Mapping[str, Any]) -> Path:
    """Resolve a capacity checkpoint using all three immutable run coordinates."""

    try:
        size = int(run["training_size"])
        head = str(run["head"])
        seed = int(run["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise Ds006780ExternalError("selected checkpoint run identity is invalid") from error
    return (
        Path(checkpoint_root)
        / f"n-{size}"
        / _head_directory_name(head)
        / f"seed-{seed}"
        / "head_checkpoint.pt"
    )


def feature_tensor_for_head(tokens: torch.Tensor, head: str) -> torch.Tensor:
    """Build the fixed feature summary consumed by one selected head."""

    if head not in SELECTED_HEADS:
        raise Ds006780ExternalError(f"selected head is not in the locked matrix: {head}")
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
        raise Ds006780ExternalError("REVE tokens must have [windows, tokens, features] shape")
    if tokens.shape[0] <= 0 or tokens.shape[1] <= 0 or tokens.shape[2] <= 0:
        raise Ds006780ExternalError("REVE tokens must be non-empty")
    if not torch.isfinite(tokens).all():
        raise Ds006780ExternalError("REVE tokens contain non-finite values")
    try:
        return capacity_summary_from_tokens(
            tokens.detach().cpu().contiguous(),
            head_name=head,
            embed_dim=int(tokens.shape[-1]),
        )
    except Exception as error:
        raise Ds006780ExternalError(f"could not build features for selected head {head}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise Ds006780ExternalError(f"could not hash artifact: {path}") from error
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = load_json_strict(path)
    except Exception as error:
        raise Ds006780ExternalError(f"could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise Ds006780ExternalError(f"{label} must be a JSON object")
    return value


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise Ds006780ExternalError(f"immutable artifact already exists: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_or_verify(path: Path, payload: Mapping[str, Any], label: str) -> None:
    if path.exists():
        existing = _load_json(path, label)
        if existing != dict(payload):
            raise Ds006780ExternalError(f"existing {label} conflicts with the locked run")
        return
    _write_create_only(path, payload)


def _infer_summary_embed_dim(state_dict: Mapping[str, Any], head: str) -> int:
    """Infer the token width from the summary-head state, not its rich input width."""

    linear_weight = state_dict.get("linear.weight")
    if not isinstance(linear_weight, torch.Tensor) or linear_weight.ndim != 2:
        raise Ds006780ExternalError("selected checkpoint has no linear.weight")
    embed_dim = int(linear_weight.shape[1])
    if embed_dim <= 0:
        raise Ds006780ExternalError("selected checkpoint has an invalid linear width")
    if head == "mean_rich_stats_residual":
        correction_weight = state_dict.get("correction.weight")
        if (
            not isinstance(correction_weight, torch.Tensor)
            or correction_weight.ndim != 2
            or int(correction_weight.shape[1]) != 4 * embed_dim
        ):
            raise Ds006780ExternalError(
                "rich-statistics checkpoint correction width is invalid"
            )
    return embed_dim


def _eeg_channel_names(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the ordered EEG-only channel inventory used by REVE."""

    inventory = candidate.get("channel_inventory")
    if not isinstance(inventory, list):
        raise Ds006780ExternalError("target-free manifest channel inventory is invalid")
    names = tuple(
        str(row["name"])
        for row in inventory
        if isinstance(row, Mapping) and row.get("type") == "EEG" and isinstance(row.get("name"), str)
    )
    if len(names) != 64 or len(set(names)) != len(names):
        raise Ds006780ExternalError("target-free manifest channel inventory is invalid")
    return names


def _load_heads(
    lock: Mapping[str, Any],
    inventory: Mapping[str, Any],
    checkpoint_root: Path,
    device: str,
) -> tuple[_LoadedHead, ...]:
    try:
        selected = select_ds006780_runs(inventory)
    except Ds006780ExternalLockError as error:
        raise Ds006780ExternalError(str(error)) from error
    if inventory.get("checkpoint_inventory_body_sha256") != lock["checkpoint_inventory_sha256"]:
        raise Ds006780ExternalError("checkpoint inventory hash differs from external lock")
    if list(selected) != lock["selected_runs"]:
        raise Ds006780ExternalError("selected checkpoint records differ from external lock")

    loaded: list[_LoadedHead] = []
    for run in selected:
        path = checkpoint_path_for_run(checkpoint_root, run)
        if not path.is_file():
            raise Ds006780ExternalError(f"selected checkpoint is missing: {path}")
        if _sha256_file(path) != run["selected_checkpoint_sha256"]:
            raise Ds006780ExternalError(f"selected checkpoint hash differs: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise Ds006780ExternalError(f"selected checkpoint is unreadable: {path}") from error
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 3:
            raise Ds006780ExternalError(f"selected checkpoint payload is invalid: {path}")
        for field, expected in (
            ("head_name", run["head"]),
            ("seed", run["seed"]),
            ("selected_epoch", run["selected_epoch"]),
            ("training_source_sha256", lock["training_source_sha256"]),
            ("representation_protocol_sha256", lock["representation_protocol_sha256"]),
            ("training_protocol_sha256", lock["training_protocol_sha256"]),
        ):
            if payload.get(field) != expected:
                raise Ds006780ExternalError(f"selected checkpoint identity differs: {path}")
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise Ds006780ExternalError(f"selected checkpoint state is invalid: {path}")
        try:
            embed_dim = _infer_summary_embed_dim(state_dict, str(run["head"]))
        except Ds006780ExternalError as error:
            raise Ds006780ExternalError(f"{error}: {path}") from error
        model = build_capacity_data_regime_summary_head(
            str(run["head"]),
            embed_dim=embed_dim,
            n_outputs=int(state_dict["linear.weight"].shape[0]),
        )
        try:
            model.load_state_dict(state_dict, strict=True)
        except (RuntimeError, ValueError) as error:
            raise Ds006780ExternalError(f"selected checkpoint state does not match head: {path}") from error
        model.to(device)
        model.eval()
        loaded.append(
            _LoadedHead(
                training_size=int(run["training_size"]),
                head=str(run["head"]),
                seed=int(run["seed"]),
                checkpoint_sha256=str(run["selected_checkpoint_sha256"]),
                model=model,
            )
        )
    return tuple(loaded)


def _predict(model: nn.Module, features: torch.Tensor, *, device: str) -> float:
    values: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, int(features.shape[0]), 64):
            batch = features[start : start + 64].to(device)
            output = model(batch).reshape(-1)
            if output.numel() != batch.shape[0] or not torch.isfinite(output).all():
                raise Ds006780ExternalError("selected head emitted invalid predictions")
            values.append(output.detach().cpu())
    prediction = float(torch.cat(values).mean())
    if not math.isfinite(prediction):
        raise Ds006780ExternalError("subject prediction is non-finite")
    return prediction


def _prediction_static(
    *,
    lock: Mapping[str, Any],
    run: _LoadedHead,
    subject_id: str,
    age: float,
    qc_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study_id": lock["study_id"],
        "lock_sha256": lock["lock_sha256"],
        "external_config_sha256": lock["external_config_sha256"],
        "target_free_manifest_sha256": lock["target_free_manifest_sha256"],
        "target_manifest_sha256": lock["target_manifest_sha256"],
        "encoder_checkpoint_sha256": lock["encoder_checkpoint_sha256"],
        "head_checkpoint_sha256": run.checkpoint_sha256,
        "training_size": run.training_size,
        "head": run.head,
        "seed": run.seed,
        "subject_id": subject_id,
        "target_age": age,
        "signal_qc_sha256": qc_sha256,
    }


def _load_existing_prediction(path: Path, expected_static: Mapping[str, Any]) -> dict[str, Any]:
    record = _load_json(path, "existing subject prediction")
    expected_fields = {"prediction_sha256", "prediction", "qc_sha256", *expected_static}
    if set(record) != expected_fields:
        raise Ds006780ExternalError(f"existing prediction schema differs: {path}")
    body = {key: value for key, value in record.items() if key != "prediction_sha256"}
    if record.get("prediction_sha256") != canonical_sha256(body):
        raise Ds006780ExternalError(f"existing prediction hash differs: {path}")
    if any(record.get(key) != value for key, value in expected_static.items()):
        raise Ds006780ExternalError(f"existing prediction identity differs: {path}")
    if record.get("qc_sha256") != expected_static["signal_qc_sha256"]:
        raise Ds006780ExternalError(f"existing prediction QC identity differs: {path}")
    if not isinstance(record.get("prediction"), (int, float)) or not math.isfinite(float(record["prediction"])):
        raise Ds006780ExternalError(f"existing prediction value is invalid: {path}")
    return record


def _qc_path(qc_root: Path, subject_id: str, run_id: str) -> Path:
    return Path(qc_root) / f"{subject_id}_{run_id}_qc.json"


def _load_subject_qc(
    *,
    bids_root: Path,
    config: Mapping[str, Any],
    target_free_manifest: Mapping[str, Any],
    candidate: Mapping[str, Any],
    expected_qc_sha256: str,
    qc_root: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    stored_path = _qc_path(qc_root, str(candidate["subject_id"]), str(candidate["run_id"]))
    stored = _load_json(stored_path, "stored target-free QC")
    if stored.get("qc_sha256") != expected_qc_sha256:
        raise Ds006780ExternalError("stored QC hash differs from target manifest")
    stored_body = {key: value for key, value in stored.items() if key != "qc_sha256"}
    if canonical_sha256(stored_body) != expected_qc_sha256:
        raise Ds006780ExternalError("stored QC payload is not self-consistent")
    reject_target_fields(stored_body)
    if stored.get("manifest_sha256") != target_free_manifest["manifest_sha256"]:
        raise Ds006780ExternalError("stored QC is bound to a different target-free manifest")
    try:
        windows, computed = build_target_free_qc(
            bids_root,
            candidate,
            manifest_sha256=str(target_free_manifest["manifest_sha256"]),
            config=config,
        )
    except (Ds006780Error, OSError) as error:
        raise Ds006780ExternalError(
            f"target-free QC recomputation failed for {candidate['subject_id']}"
        ) from error
    if computed != stored_body:
        raise Ds006780ExternalError(
            f"recomputed target-free QC differs from stored QC: {candidate['subject_id']}"
        )
    return windows, stored_body


def _prediction_inventory(
    *, lock: Mapping[str, Any], output_root: Path, subjects: Sequence[str], heads: Sequence[_LoadedHead]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    expected_paths: set[Path] = set()
    for run in heads:
        for subject_id in subjects:
            path = (
                output_root
                / "predictions"
                / f"n-{run.training_size}"
                / run.head
                / f"seed-{run.seed}"
                / f"{subject_id}.json"
            )
            expected_paths.add(path)
            record = _load_json(path, "subject prediction")
            row = {
                "training_size": run.training_size,
                "head": run.head,
                "seed": run.seed,
                "subject_id": subject_id,
                "path": str(path.relative_to(output_root)),
                "prediction_sha256": record.get("prediction_sha256"),
            }
            if not isinstance(row["prediction_sha256"], str):
                raise Ds006780ExternalError(f"prediction hash is missing: {path}")
            rows.append(row)
    actual_paths = set((output_root / "predictions").rglob("*.json"))
    if actual_paths != expected_paths:
        raise Ds006780ExternalError("prediction file inventory is not exact")
    rows.sort(key=lambda row: (row["training_size"], row["head"], row["seed"], row["subject_id"]))
    body = {
        "schema_version": 1,
        "status": "complete",
        "study_id": lock["study_id"],
        "lock_sha256": lock["lock_sha256"],
        "prediction_count": len(rows),
        "predictions": rows,
    }
    return {**body, "prediction_inventory_sha256": canonical_sha256(body)}


def run_ds006780_external(
    *,
    lock_path: Path,
    external_config_path: Path,
    target_free_manifest_path: Path,
    target_manifest_path: Path,
    checkpoint_inventory_path: Path,
    checkpoint_root: Path,
    bids_root: Path,
    qc_root: Path,
    mapping_path: Path | None,
    output_root: Path,
    project_root: Path,
    device: str,
    extraction_batch_size: int,
) -> Mapping[str, Any]:
    """Extract once per subject and evaluate all 40 locked heads without metrics."""

    lock = load_ds006780_external_lock(lock_path)
    output_root = Path(output_root).resolve()
    if output_root != Path(lock["output_root"]).resolve():
        raise Ds006780ExternalError("output root differs from the external lock")
    if source_tree_sha256(project_root) != lock["execution_source_sha256"]:
        raise Ds006780ExternalError("execution source differs from the external lock")
    if device not in {"cpu", "cuda", "mps"}:
        raise Ds006780ExternalError("device must be cpu, cuda, or mps")
    if device == "cuda" and not torch.cuda.is_available():
        raise Ds006780ExternalError("CUDA requested but unavailable")
    if isinstance(extraction_batch_size, bool) or extraction_batch_size <= 0:
        raise Ds006780ExternalError("extraction_batch_size must be positive")

    config = _load_json(external_config_path, "external config")
    validate_external_config(config, require_approved_precision=True)
    if canonical_sha256(config) != lock["external_config_sha256"]:
        raise Ds006780ExternalError("external config differs from the external lock")
    target_free = _load_json(target_free_manifest_path, "target-free manifest")
    target = _load_json(target_manifest_path, "target manifest")
    if target_free.get("manifest_sha256") != lock["target_free_manifest_sha256"]:
        raise Ds006780ExternalError("target-free manifest digest differs from lock")
    if target.get("manifest_sha256") != lock["target_manifest_sha256"]:
        raise Ds006780ExternalError("target manifest digest differs from lock")
    try:
        verify_ds006780_manifest(
            bids_root,
            target_free,
            config_path=external_config_path,
            project_root=project_root,
        )
    except Ds006780Error as error:
        raise Ds006780ExternalError(str(error)) from error
    candidates = {
        str(row["subject_id"]): row
        for row in target_free.get("candidate_runs", [])
        if isinstance(row, Mapping)
    }
    subjects = target.get("subjects")
    if not isinstance(subjects, Mapping) or list(sorted(subjects)) != lock["subject_ids"]:
        raise Ds006780ExternalError("target-bearing subject inventory differs from lock")
    inventory = _load_json(checkpoint_inventory_path, "capacity checkpoint inventory")
    loaded_heads = _load_heads(lock, inventory, checkpoint_root, device)
    if not loaded_heads:
        raise Ds006780ExternalError("no selected checkpoints were loaded")

    first_candidate = candidates.get(lock["subject_ids"][0])
    if first_candidate is None:
        raise Ds006780ExternalError("first locked subject is absent from target-free manifest")
    channel_names = _eeg_channel_names(first_candidate)
    encoder = load_reve_encoder(
        lock["encoder_checkpoint"],
        channel_names=channel_names,
        mapping_path=mapping_path,
        initialization_seed=0,
    )
    if encoder_state_sha256(encoder) != lock["encoder_checkpoint_sha256"]:
        raise Ds006780ExternalError("loaded REVE encoder differs from external lock")
    encoder_hash = encoder_state_sha256(encoder)
    output_root.mkdir(parents=True, exist_ok=True)
    start_marker_body = {
        "schema_version": 1,
        "study_id": lock["study_id"],
        "lock_sha256": lock["lock_sha256"],
        "state": "started",
        "subject_count": len(lock["subject_ids"]),
        "selected_run_count": len(loaded_heads),
    }
    _write_or_verify(
        output_root / "evaluation_started.json",
        {**start_marker_body, "marker_sha256": canonical_sha256(start_marker_body)},
        "evaluation started marker",
    )

    for subject_id in lock["subject_ids"]:
        candidate = candidates.get(subject_id)
        subject_record = subjects.get(subject_id)
        if candidate is None or not isinstance(subject_record, Mapping):
            raise Ds006780ExternalError(f"locked subject is absent from manifests: {subject_id}")
        age = float(subject_record["age_years"])
        qc_sha256 = str(subject_record["signal_qc_sha256"])
        missing = []
        for run in loaded_heads:
            path = (
                output_root
                / "predictions"
                / f"n-{run.training_size}"
                / run.head
                / f"seed-{run.seed}"
                / f"{subject_id}.json"
            )
            static = _prediction_static(
                lock=lock, run=run, subject_id=subject_id, age=age, qc_sha256=qc_sha256
            )
            if path.exists():
                _load_existing_prediction(path, static)
            else:
                missing.append((run, path, static))
        if not missing:
            continue
        windows, _ = _load_subject_qc(
            bids_root=bids_root,
            config=config,
            target_free_manifest=target_free,
            candidate=candidate,
            expected_qc_sha256=qc_sha256,
            qc_root=qc_root,
        )
        representations, evidence = extract_frozen_representations_batched(
            encoder,
            windows,
            batch_size=extraction_batch_size,
            device=device,
            layer_indices=(-1,),
            pool_tokens=False,
        )
        if evidence["state_sha256_before"] != encoder_hash or evidence["state_sha256_after"] != encoder_hash:
            raise Ds006780ExternalError("REVE encoder state changed during external extraction")
        tokens = representations[-1].detach().cpu().contiguous()
        feature_cache = {
            head: feature_tensor_for_head(tokens, head) for head in SELECTED_HEADS
        }
        for run, path, static in missing:
            feature = feature_cache[run.head]
            prediction = _predict(run.model, feature, device=device)
            body = {**static, "qc_sha256": qc_sha256, "prediction": prediction}
            payload = {**body, "prediction_sha256": canonical_sha256(body)}
            _write_create_only(path, payload)

    inventory_payload = _prediction_inventory(
        lock=lock,
        output_root=output_root,
        subjects=lock["subject_ids"],
        heads=loaded_heads,
    )
    if inventory_payload["prediction_count"] != lock["expected_prediction_count"]:
        raise Ds006780ExternalError("prediction inventory count differs from lock")
    _write_or_verify(
        output_root / "prediction_inventory.json",
        inventory_payload,
        "prediction inventory",
    )
    completion_body = {
        "schema_version": 1,
        "study_id": lock["study_id"],
        "lock_sha256": lock["lock_sha256"],
        "prediction_inventory_sha256": inventory_payload["prediction_inventory_sha256"],
        "status": "complete",
    }
    _write_or_verify(
        output_root / "evaluation_completed.json",
        {**completion_body, "marker_sha256": canonical_sha256(completion_body)},
        "evaluation completed marker",
    )
    return inventory_payload
