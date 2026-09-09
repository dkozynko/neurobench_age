"""Lock-gated external evaluation for the capacity--data extension.

This adapter is intentionally separate from the primary external evaluator.
It consumes only a checkpoint-sealed extension lock, evaluates the declared
75-subject MIPDB primary cohort for every one of the 90 finalized runs, and
creates the final extension lock only after the complete prediction inventory
has been validated.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from neurobench_age.pipelines.external_holdout import ExternalHoldoutError
from neurobench_age.pipelines.frozen_probe_training import FrozenProbeRunResult
from neurobench_age.research.capacity_data_regime_lock import (
    CapacityDataRegimeLockError,
    build_final_lock,
    build_prediction_inventory,
    load_checkpoint_inventory,
    load_checkpoint_sealed_lock,
    transition_lifecycle,
)

from .capacity_data_regime import (
    CapacityDataRegimePipelineError,
    _head_directory_name,
    build_capacity_data_regime_head,
    required_capacity_layer_for_head,
    snapshot_primary_artifacts,
)


class CapacityDataRegimeExternalError(RuntimeError):
    """Raised when secondary external evidence is missing or inconsistent."""


_HEAD_NAMES = (
    "mean_linear",
    "mean_rich_stats_residual",
    "mean_mlp_residual_matched(hidden_dim=4)",
)
_SIZES = (200, 400, 800)
_SEEDS = tuple(range(33, 43))
_PREDICTION_SPLIT = "mipdb_primary"
_INFERENCE_BATCH_SIZE = 64
_SAFE_SUBJECT_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise CapacityDataRegimeExternalError(f"cannot hash artifact: {path}") from error
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CapacityDataRegimeExternalError(f"cannot read {description}: {path}") from error
    if not isinstance(value, dict):
        raise CapacityDataRegimeExternalError(f"{description} must be a JSON object")
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
            raise CapacityDataRegimeExternalError(
                f"external artifact already exists: {path}"
            ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_external_output_root(output_root: Path, checkpoint_root: Path) -> None:
    resolved_output = Path(output_root).resolve()
    resolved_checkpoint = Path(checkpoint_root).resolve()
    if resolved_output == resolved_checkpoint or resolved_output.is_relative_to(resolved_checkpoint):
        raise CapacityDataRegimeExternalError(
            "secondary external output must be separate from checkpoint output"
        )
    allowed = {
        "run_predictions",
        "prediction_inventory.json",
        "final_lock.json",
        "external_metrics.json",
    }
    if resolved_output.exists():
        for child in resolved_output.iterdir():
            if child.name not in allowed:
                raise CapacityDataRegimeExternalError(
                    f"unexpected secondary external artifact: {child}"
                )


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CapacityDataRegimeExternalError(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise CapacityDataRegimeExternalError(f"{field} must be finite")
    return value


def _validate_subjects(subjects: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    if len(subjects) != 75:
        raise CapacityDataRegimeExternalError(
            "secondary MIPDB evaluation requires exactly 75 primary subjects"
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, subject in enumerate(subjects):
        if not isinstance(subject, Mapping):
            raise CapacityDataRegimeExternalError(f"MIPDB subject {index} is invalid")
        subject_id = subject.get("subject_id")
        if (
            not isinstance(subject_id, str)
            or not _SAFE_SUBJECT_ID.fullmatch(subject_id)
            or subject_id in seen
        ):
            raise CapacityDataRegimeExternalError(
                f"MIPDB subject {index} has an invalid or duplicate ID"
            )
        split = subject.get("split", _PREDICTION_SPLIT)
        if split != _PREDICTION_SPLIT:
            raise CapacityDataRegimeExternalError(
                "secondary evaluation refuses non-primary or extrapolation subjects"
            )
        seen.add(subject_id)
        normalized.append(
            {
                "subject_id": subject_id,
                "true_age": _finite_float(subject.get("true_age", subject.get("age")), "true_age"),
                "split": split,
            }
        )
    return tuple(normalized)


def _validate_representation(value: object, *, subject_id: str) -> torch.Tensor:
    if isinstance(value, Mapping):
        if set(value) != {-2, -1}:
            raise CapacityDataRegimeExternalError(
                f"MIPDB representations for {subject_id} must contain exactly layers -2 and -1"
            )
        tensor = value[-1]
    else:
        tensor = value
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.ndim != 3
        or any(int(dimension) <= 0 for dimension in tensor.shape)
        or tensor.requires_grad
        or not torch.isfinite(tensor).all()
    ):
        raise CapacityDataRegimeExternalError(
            f"MIPDB representation for {subject_id} is invalid"
        )
    return tensor.detach().cpu().contiguous()


def _load_extension_head(
    checkpoint_root: Path,
    run: Mapping[str, Any],
    core: Mapping[str, Any],
    device: str,
) -> nn.Module:
    run_dir = (
        Path(checkpoint_root)
        / f"n-{run['training_size']}"
        / _head_directory_name(str(run["head"]))
        / f"seed-{run['seed']}"
    )
    checkpoint_path = run_dir / "head_checkpoint.pt"
    if not checkpoint_path.is_file():
        raise CapacityDataRegimeExternalError(f"checkpoint is missing: {checkpoint_path}")
    actual_sha = _sha256_file(checkpoint_path)
    if actual_sha != run["selected_checkpoint_sha256"]:
        raise CapacityDataRegimeExternalError(
            f"checkpoint hash does not match inventory: {checkpoint_path}"
        )
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CapacityDataRegimeExternalError(
            f"checkpoint is unreadable: {checkpoint_path}"
        ) from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 3:
        raise CapacityDataRegimeExternalError(f"checkpoint payload is invalid: {checkpoint_path}")
    expected = {
        "head_name": run["head"],
        "seed": run["seed"],
        "training_source_sha256": core["training_source_sha256"],
        "representation_protocol_sha256": core["representation_protocol_sha256"],
        "training_protocol_sha256": core["training_protocol_sha256"],
    }
    if any(payload.get(field) != value for field, value in expected.items()):
        raise CapacityDataRegimeExternalError(
            f"checkpoint provenance does not match extension lock: {checkpoint_path}"
        )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not isinstance(state_dict.get("linear.weight"), torch.Tensor):
        raise CapacityDataRegimeExternalError(
            f"checkpoint state does not expose linear.weight: {checkpoint_path}"
        )
    linear_weight = state_dict["linear.weight"]
    if linear_weight.ndim != 2:
        raise CapacityDataRegimeExternalError(f"checkpoint linear weight is invalid: {checkpoint_path}")
    model = build_capacity_data_regime_head(
        str(run["head"]), embed_dim=int(linear_weight.shape[1]), n_outputs=int(linear_weight.shape[0])
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, ValueError) as error:
        raise CapacityDataRegimeExternalError(
            f"checkpoint state does not match the declared extension head: {checkpoint_path}"
        ) from error
    model.to(device)
    model.eval()
    return model


def _predict(model: nn.Module, representations: torch.Tensor, *, device: str) -> float:
    values: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, representations.shape[0], _INFERENCE_BATCH_SIZE):
            output = model(representations[start : start + _INFERENCE_BATCH_SIZE].to(device)).reshape(-1)
            if output.numel() != min(_INFERENCE_BATCH_SIZE, representations.shape[0] - start):
                raise CapacityDataRegimeExternalError("extension head emitted the wrong prediction count")
            if not torch.isfinite(output).all():
                raise CapacityDataRegimeExternalError("extension head emitted a non-finite prediction")
            values.append(output.detach().cpu())
    prediction = float(torch.cat(values).mean())
    if not math.isfinite(prediction):
        raise CapacityDataRegimeExternalError("subject prediction is non-finite")
    return prediction


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    targets = torch.tensor([float(row["true_age"]) for row in rows], dtype=torch.float64)
    predictions = torch.tensor([float(row["prediction"]) for row in rows], dtype=torch.float64)
    if targets.numel() < 2 or not torch.isfinite(targets).all() or not torch.isfinite(predictions).all():
        raise CapacityDataRegimeExternalError("external metric inputs are invalid")
    centered_targets = targets - targets.mean()
    centered_predictions = predictions - predictions.mean()
    denominator = float(torch.sqrt(centered_targets.square().sum() * centered_predictions.square().sum()))
    if denominator <= 0.0:
        raise CapacityDataRegimeExternalError("external Pearson is undefined")
    slope_denominator = float(centered_predictions.square().sum())
    if slope_denominator <= 0.0:
        raise CapacityDataRegimeExternalError("external calibration is undefined")
    slope = float(torch.dot(centered_predictions, centered_targets) / slope_denominator)
    intercept = float(targets.mean() - slope * predictions.mean())
    return {
        "subject_count": len(rows),
        "pearson": float(torch.dot(centered_targets, centered_predictions) / denominator),
        "mae": float((predictions - targets).abs().mean()),
        "rmse": float(torch.sqrt((predictions - targets).square().mean())),
        "calibration": {"intercept": intercept, "slope": slope},
    }


def run_capacity_data_regime_external(
    *,
    checkpoint_sealed_lock_path: Path,
    checkpoint_inventory_path: Path,
    checkpoint_root: Path,
    primary_study_lock: Path,
    primary_prediction_inventory: Path,
    subjects: Sequence[Mapping[str, Any]],
    output_root: Path,
    representation_provider: Callable[[str], object],
    device: str = "cpu",
) -> Mapping[str, Any]:
    """Evaluate the exact 90-run extension inventory on 75 MIPDB subjects."""

    if device not in {"cpu", "cuda", "mps"}:
        raise CapacityDataRegimeExternalError("device must be cpu, cuda, or mps")
    if device == "cuda" and not torch.cuda.is_available():
        raise CapacityDataRegimeExternalError("CUDA was requested but is unavailable")
    try:
        sealed = load_checkpoint_sealed_lock(
            _load_json(checkpoint_sealed_lock_path, "checkpoint-sealed lock")
        )
        inventory = load_checkpoint_inventory(
            _load_json(checkpoint_inventory_path, "checkpoint inventory"),
            expected_core_digest=sealed["lock_core_sha256"],
        )
    except CapacityDataRegimeLockError as error:
        raise CapacityDataRegimeExternalError(str(error)) from error
    if inventory["checkpoint_inventory_body_sha256"] != sealed["checkpoint_inventory_body_sha256"]:
        raise CapacityDataRegimeExternalError(
            "checkpoint inventory does not match checkpoint-sealed lock"
        )
    core = sealed["lock_core"]
    before = snapshot_primary_artifacts(
        primary_study_lock=primary_study_lock,
        primary_prediction_inventory=primary_prediction_inventory,
    )
    if before["primary_study_lock_sha256"] != core["parent_primary_study_lock_sha256"] or before[
        "primary_prediction_inventory_sha256"
    ] != core["parent_primary_prediction_inventory_sha256"]:
        raise CapacityDataRegimeExternalError(
            "primary evidence snapshot differs from extension lock core"
        )
    normalized_subjects = _validate_subjects(subjects)
    output_root = Path(output_root)
    _validate_external_output_root(output_root, Path(checkpoint_root))
    output_root.mkdir(parents=True, exist_ok=True)
    lifecycle_dir = Path(checkpoint_sealed_lock_path).parent / "lifecycle"
    lifecycle_completed = (lifecycle_dir / "lifecycle-completed.json").exists()
    lifecycle_started = (lifecycle_dir / "lifecycle-external_started.json").exists()
    if not lifecycle_completed and not lifecycle_started:
        try:
            transition_lifecycle(
                lifecycle_dir,
                state="external_started",
                lock_sha256=sealed["lock_core_sha256"],
            )
        except CapacityDataRegimeLockError as error:
            raise CapacityDataRegimeExternalError(str(error)) from error

    all_predictions: list[dict[str, Any]] = []
    run_metrics: list[dict[str, Any]] = []
    for run in inventory["runs"]:
        identity = (run["training_size"], run["head"], run["seed"])
        model = _load_extension_head(Path(checkpoint_root), run, core, device)
        rows: list[dict[str, Any]] = []
        for subject in normalized_subjects:
            actual_snapshot = snapshot_primary_artifacts(
                primary_study_lock=primary_study_lock,
                primary_prediction_inventory=primary_prediction_inventory,
            )
            if actual_snapshot != before:
                raise CapacityDataRegimeExternalError(
                    "primary evidence changed before MIPDB materialization"
                )
            representation = _validate_representation(
                representation_provider(subject["subject_id"]),
                subject_id=subject["subject_id"],
            )
            row = {
                "training_size": run["training_size"],
                "head": run["head"],
                "seed": run["seed"],
                "subject_id": subject["subject_id"],
                "true_age": subject["true_age"],
                "prediction": _predict(model, representation, device=device),
                "split": _PREDICTION_SPLIT,
            }
            rows.append(row)
        metrics = _metrics(rows)
        run_file = (
            output_root
            / "run_predictions"
            / f"n-{run['training_size']}"
            / _head_directory_name(str(run["head"]))
            / f"seed-{run['seed']}.json"
        )
        body = {
            "schema_version": 1,
            "training_size": run["training_size"],
            "head": run["head"],
            "seed": run["seed"],
            "prediction_count": len(rows),
            "metrics": metrics,
            "predictions": rows,
        }
        payload = {**body, "body_sha256": _canonical_sha256(body)}
        if run_file.exists():
            existing = _load_json(run_file, "existing extension run predictions")
            if existing != payload:
                raise CapacityDataRegimeExternalError(
                    f"existing external run predictions conflict: {run_file}"
                )
        else:
            _write_create_only(run_file, payload)
        all_predictions.extend(rows)
        run_metrics.append({"identity": list(identity), **metrics})

    actual_snapshot = snapshot_primary_artifacts(
        primary_study_lock=primary_study_lock,
        primary_prediction_inventory=primary_prediction_inventory,
    )
    if actual_snapshot != before:
        raise CapacityDataRegimeExternalError(
            "primary evidence changed during MIPDB evaluation"
        )
    try:
        prediction_inventory = build_prediction_inventory(
            {"lock_core": core, "lock_core_sha256": sealed["lock_core_sha256"]},
            inventory,
            predictions=all_predictions,
        )
        final_lock = build_final_lock(sealed, prediction_inventory)
    except CapacityDataRegimeLockError as error:
        raise CapacityDataRegimeExternalError(str(error)) from error
    prediction_inventory_path = output_root / "prediction_inventory.json"
    final_lock_path = output_root / "final_lock.json"
    if prediction_inventory_path.exists():
        if _load_json(prediction_inventory_path, "prediction inventory") != prediction_inventory:
            raise CapacityDataRegimeExternalError("existing prediction inventory conflicts")
    else:
        _write_create_only(prediction_inventory_path, prediction_inventory)
    if final_lock_path.exists():
        if _load_json(final_lock_path, "final extension lock") != final_lock:
            raise CapacityDataRegimeExternalError("existing final extension lock conflicts")
    else:
        _write_create_only(final_lock_path, final_lock)
    metrics_payload = {
        "schema_version": 1,
        "status": "complete",
        "checkpoint_sealed_lock_sha256": sealed["lock_sha256"],
        "prediction_inventory_body_sha256": prediction_inventory[
            "prediction_inventory_body_sha256"
        ],
        "runs": sorted(
            run_metrics,
            key=lambda row: (row["identity"][0], row["identity"][1], row["identity"][2]),
        ),
    }
    metrics_payload = {
        **metrics_payload,
        "metrics_sha256": _canonical_sha256(metrics_payload),
    }
    metrics_path = output_root / "external_metrics.json"
    if metrics_path.exists():
        if _load_json(metrics_path, "external metrics") != metrics_payload:
            raise CapacityDataRegimeExternalError("existing external metrics conflict")
    else:
        _write_create_only(metrics_path, metrics_payload)
    if not lifecycle_completed:
        try:
            transition_lifecycle(
                lifecycle_dir,
                state="completed",
                lock_sha256=sealed["lock_core_sha256"],
            )
        except CapacityDataRegimeLockError as error:
            raise CapacityDataRegimeExternalError(str(error)) from error
    return {
        "status": "completed",
        "prediction_inventory": prediction_inventory,
        "final_lock": final_lock,
        "external_metrics": metrics_payload,
    }
