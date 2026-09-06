"""Cache-only head training and exact frozen-probe run inventory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from neurobench_age.heads.math import (
    MeanLinearCopyHead,
    MeanRichStatsResidualHead,
    MultiQueryRichStatsResidualHead,
)
from neurobench_age.research.protocol import StudyProtocol, TrainingContract

from .frozen_probe import (
    FrozenEncoderError,
    RepresentationCacheIdentity,
    _canonical_sha256,
    _is_sha256,
    _sha256_file,
    load_cached_representations,
)

APPROVED_HEADS = (
    "mean_linear",
    "mean_layer_linear",
    "mean_rich_stats_residual",
    "multi_query_rich_stats",
)
_HEAD_LAYERS = {
    "mean_linear": -1,
    "mean_layer_linear": -2,
    "mean_rich_stats_residual": -1,
    "multi_query_rich_stats": -1,
}


def required_layer_for_head(head_name: str) -> int:
    try:
        return _HEAD_LAYERS[head_name]
    except KeyError as error:
        raise FrozenEncoderError(
            f"head must be one of the approved heads: {APPROVED_HEADS}"
        ) from error


def build_frozen_probe_head(
    head_name: str, *, embed_dim: int, n_outputs: int = 1
) -> nn.Module:
    """Build one approved head that accepts cached tokens, never an encoder."""

    required_layer_for_head(head_name)
    if head_name in {"mean_linear", "mean_layer_linear"}:
        return MeanLinearCopyHead(embed_dim=embed_dim, n_outputs=n_outputs)
    if head_name == "mean_rich_stats_residual":
        return MeanRichStatsResidualHead(embed_dim=embed_dim, n_outputs=n_outputs)
    return MultiQueryRichStatsResidualHead(
        embed_dim=embed_dim, n_outputs=n_outputs
    )


@dataclass(frozen=True)
class CachedSubjectRecord:
    subject_id: str
    split: str
    age: float
    cache_identity: RepresentationCacheIdentity

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation"}:
            raise FrozenEncoderError(
                "frozen-probe records must use train or validation split"
            )
        if not math.isfinite(float(self.age)):
            raise FrozenEncoderError("frozen-probe record must contain a finite age")
        if self.subject_id != self.cache_identity.subject_id:
            raise FrozenEncoderError(
                "record subject_id does not match its cache identity"
            )


def validate_training_records(
    records: Sequence[CachedSubjectRecord],
) -> tuple[CachedSubjectRecord, ...]:
    normalized = tuple(records)
    seen: set[str] = set()
    for record in normalized:
        if not isinstance(record, CachedSubjectRecord):
            raise FrozenEncoderError("training records have an invalid item")
        if record.subject_id in seen:
            raise FrozenEncoderError(
                "training records contain duplicate or overlapping subject IDs"
            )
        seen.add(record.subject_id)
    splits = {record.split for record in normalized}
    if splits != {"train", "validation"}:
        raise FrozenEncoderError(
            "training records must contain non-empty train and validation splits"
        )
    return normalized


def load_frozen_probe_training_manifest(
    path: Path, *, protocol: StudyProtocol
) -> tuple[CachedSubjectRecord, ...]:
    """Load the strict HBN train/validation-to-cache index used by all heads."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenEncoderError(f"could not read training manifest: {path}") from error
    required_fields = {
        "schema_version",
        "protocol_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "dataset_manifest_sha256",
        "subject_manifest_sha256",
        "acquisition_files",
        "preprocessing_sha256",
        "source_tree_sha256",
        "subjects",
    }
    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise FrozenEncoderError("training manifest fields do not match the strict schema")
    if payload["schema_version"] != 1:
        raise FrozenEncoderError("training manifest schema_version must be 1")
    if payload["protocol_sha256"] != protocol.sha256:
        raise FrozenEncoderError("training manifest protocol_sha256 does not match --protocol")
    if payload["checkpoint"] != protocol.encoder.checkpoint:
        raise FrozenEncoderError("training manifest checkpoint does not match protocol")
    for field in (
        "checkpoint_sha256",
        "dataset_manifest_sha256",
        "subject_manifest_sha256",
        "preprocessing_sha256",
        "source_tree_sha256",
    ):
        value = payload[field]
        if not isinstance(value, str) or not _is_sha256(value):
            raise FrozenEncoderError(f"training manifest {field} must be a SHA-256 digest")
    acquisition_files = payload["acquisition_files"]
    if not isinstance(acquisition_files, list) or not acquisition_files:
        raise FrozenEncoderError("training manifest acquisition_files must be non-empty")
    for item in acquisition_files:
        if (
            not isinstance(item, dict)
            or set(item) != {"subject_id", "path", "size_bytes", "sha256"}
            or not isinstance(item["subject_id"], str)
            or not isinstance(item["path"], str)
            or isinstance(item["size_bytes"], bool)
            or not isinstance(item["size_bytes"], int)
            or item["size_bytes"] < 0
            or not _is_sha256(item["sha256"])
        ):
            raise FrozenEncoderError("training manifest acquisition file is invalid")
    expected_dataset_identity = _canonical_sha256(
        {
            "subject_manifest_sha256": payload["subject_manifest_sha256"],
            "acquisition_files": acquisition_files,
        }
    )
    if payload["dataset_manifest_sha256"] != expected_dataset_identity:
        raise FrozenEncoderError("training manifest dataset identity does not match acquisition")
    raw_subjects = payload["subjects"]
    if not isinstance(raw_subjects, list):
        raise FrozenEncoderError("training manifest subjects must be an array")
    records: list[CachedSubjectRecord] = []
    for index, item in enumerate(raw_subjects):
        if not isinstance(item, dict) or set(item) != {"subject_id", "split", "age"}:
            raise FrozenEncoderError(
                f"training manifest subjects[{index}] has invalid fields"
            )
        subject_id = item["subject_id"]
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise FrozenEncoderError(
                f"training manifest subjects[{index}].subject_id is invalid"
            )
        age = item["age"]
        if isinstance(age, bool) or not isinstance(age, (int, float)):
            raise FrozenEncoderError(
                f"training manifest subjects[{index}].age must be numeric"
            )
        identity = RepresentationCacheIdentity(
            protocol_sha256=protocol.sha256,
            checkpoint=payload["checkpoint"],
            checkpoint_sha256=payload["checkpoint_sha256"],
            dataset_manifest_sha256=payload["dataset_manifest_sha256"],
            preprocessing_sha256=payload["preprocessing_sha256"],
            subject_id=subject_id,
            source_tree_sha256=payload["source_tree_sha256"],
        )
        records.append(
            CachedSubjectRecord(
                subject_id=subject_id,
                split=item["split"],
                age=float(age),
                cache_identity=identity,
            )
        )
    return validate_training_records(records)


def predict_cached_subjects(
    head: nn.Module,
    records: Sequence[CachedSubjectRecord],
    *,
    cache_root: Path,
    head_name: str,
    batch_size: int,
    device: str,
) -> tuple[dict[str, float | str], ...]:
    """Predict subject ages from one declared cached layer only."""

    layer_index = required_layer_for_head(head_name)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise FrozenEncoderError("batch_size must be a positive integer")
    if not records:
        raise FrozenEncoderError("prediction records must not be empty")
    subject_ids = [record.subject_id for record in records]
    if len(subject_ids) != len(set(subject_ids)):
        raise FrozenEncoderError("prediction records contain duplicate subject IDs")
    head.to(device)
    head.eval()
    rows: list[dict[str, float | str]] = []
    with torch.inference_mode():
        for record in records:
            cached = load_cached_representations(
                cache_root,
                record.cache_identity,
                required_layers=(layer_index,),
            )[layer_index]
            if cached.ndim != 3 or cached.shape[0] == 0:
                raise FrozenEncoderError(
                    "cached representation must have shape [windows, tokens, features]"
                )
            window_predictions: list[torch.Tensor] = []
            for start in range(0, cached.shape[0], batch_size):
                batch = cached[start : start + batch_size].to(device)
                prediction = head(batch).reshape(-1)
                if prediction.numel() != batch.shape[0] or not torch.isfinite(prediction).all():
                    raise FrozenEncoderError(
                        "head must produce one finite prediction per cached window"
                    )
                window_predictions.append(prediction.cpu())
            subject_prediction = float(torch.cat(window_predictions).mean())
            rows.append(
                {
                    "subject_id": record.subject_id,
                    "age": float(record.age),
                    "prediction": subject_prediction,
                }
            )
    return tuple(rows)


@dataclass(frozen=True)
class FrozenProbeRunResult:
    manifest: Mapping[str, Any]
    reused: bool


def _strict_training_contract(training: TrainingContract) -> None:
    if training.optimizer != "AdamW":
        raise FrozenEncoderError("frozen-probe optimizer must be AdamW")
    if training.loss != "MSELoss":
        raise FrozenEncoderError("frozen-probe loss must be MSELoss")
    if training.checkpoint_metric != "validation_pearson" or training.metric_mode != "max":
        raise FrozenEncoderError(
            "frozen-probe checkpoint selection must maximize validation Pearson"
        )
    if (
        training.batch_size <= 0
        or training.max_epochs <= 0
        or training.patience <= 0
        or training.learning_rate <= 0
        or training.weight_decay < 0
    ):
        raise FrozenEncoderError("frozen-probe training settings are invalid")


def _run_identity(
    *,
    head_name: str,
    seed: int,
    records: Sequence[CachedSubjectRecord],
    training: TrainingContract,
) -> dict[str, Any]:
    training_payload = asdict(training)
    training_payload["seeds"] = list(training.seeds)
    cache_contract_fields = (
        "protocol_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "dataset_manifest_sha256",
        "preprocessing_sha256",
        "source_tree_sha256",
    )
    first_identity = records[0].cache_identity
    cache_contract = {
        field: getattr(first_identity, field) for field in cache_contract_fields
    }
    for record in records[1:]:
        actual = {
            field: getattr(record.cache_identity, field)
            for field in cache_contract_fields
        }
        if actual != cache_contract:
            raise FrozenEncoderError(
                "training records contain mixed cache provenance"
            )
    return {
        "schema_version": 1,
        "head_name": head_name,
        "seed": seed,
        "cache_contract": cache_contract,
        "training": training_payload,
        "subjects": [
            {
                "subject_id": record.subject_id,
                "split": record.split,
                "age": float(record.age),
                "cache_key": record.cache_identity.key,
            }
            for record in records
        ],
    }


def _process_peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _pearson_from_rows(rows: Sequence[Mapping[str, float | str]]) -> float:
    if len(rows) < 2:
        return float("nan")
    targets = torch.tensor([float(row["age"]) for row in rows], dtype=torch.float64)
    predictions = torch.tensor(
        [float(row["prediction"]) for row in rows], dtype=torch.float64
    )
    targets = targets - targets.mean()
    predictions = predictions - predictions.mean()
    denominator = torch.sqrt(targets.square().sum() * predictions.square().sum())
    if denominator <= 0:
        return float("nan")
    return float((targets * predictions).sum() / denominator)


def _load_completed_run(
    run_dir: Path, *, expected_identity_sha256: str
) -> Mapping[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    checkpoint_path = run_dir / "head_checkpoint.pt"
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise FrozenEncoderError(f"existing run is incomplete: {run_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenEncoderError(f"existing run manifest is invalid: {run_dir}") from error
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise FrozenEncoderError(f"existing run completion marker is invalid: {run_dir}")
    claimed_manifest_hash = manifest.get("run_manifest_sha256")
    manifest_body = {
        key: value for key, value in manifest.items() if key != "run_manifest_sha256"
    }
    if claimed_manifest_hash != _canonical_sha256(manifest_body):
        raise FrozenEncoderError(f"existing run manifest hash does not match: {run_dir}")
    if manifest.get("run_identity_sha256") != expected_identity_sha256:
        raise FrozenEncoderError(f"existing run identity does not match: {run_dir}")
    if manifest.get("checkpoint_sha256") != _sha256_file(checkpoint_path):
        raise FrozenEncoderError(f"existing run checkpoint hash does not match: {run_dir}")
    return manifest


def _configure_strict_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_frozen_probe_run(
    *,
    head_name: str,
    seed: int,
    records: Sequence[CachedSubjectRecord],
    cache_root: Path,
    run_dir: Path,
    training: TrainingContract,
    device: str,
) -> FrozenProbeRunResult:
    """Train one cache-only head and select the earliest best validation epoch."""

    required_layer = required_layer_for_head(head_name)
    records = validate_training_records(records)
    _strict_training_contract(training)
    if seed not in training.seeds:
        raise FrozenEncoderError(f"seed {seed} is outside the predeclared inventory")
    identity = _run_identity(
        head_name=head_name, seed=seed, records=records, training=training
    )
    identity_sha256 = _canonical_sha256(identity)
    run_dir = Path(run_dir)
    if run_dir.exists():
        return FrozenProbeRunResult(
            manifest=_load_completed_run(
                run_dir, expected_identity_sha256=identity_sha256
            ),
            reused=True,
        )

    train_records = tuple(record for record in records if record.split == "train")
    validation_records = tuple(
        record for record in records if record.split == "validation"
    )
    sample = load_cached_representations(
        cache_root,
        train_records[0].cache_identity,
        required_layers=(required_layer,),
    )[required_layer]
    if sample.ndim != 3 or sample.shape[-1] <= 0:
        raise FrozenEncoderError(
            "cached representation must have shape [windows, tokens, features]"
        )

    _configure_strict_determinism(seed)
    head = build_frozen_probe_head(head_name, embed_dim=int(sample.shape[-1])).to(device)
    head_parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        head_parameters,
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != {id(parameter) for parameter in head_parameters}:
        raise FrozenEncoderError("optimizer does not own exactly the trainable head parameters")
    loss_function = nn.MSELoss()
    order_generator = torch.Generator(device="cpu").manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch_index in range(training.max_epochs):
        head.train()
        epoch_loss_sum = 0.0
        epoch_windows = 0
        subject_order = torch.randperm(
            len(train_records), generator=order_generator
        ).tolist()
        for subject_offset in subject_order:
            record = train_records[subject_offset]
            cached = load_cached_representations(
                cache_root,
                record.cache_identity,
                required_layers=(required_layer,),
            )[required_layer]
            if cached.ndim != 3 or cached.shape[-1] != sample.shape[-1] or cached.shape[0] == 0:
                raise FrozenEncoderError("training cache tensors have inconsistent shapes")
            window_order = torch.randperm(
                cached.shape[0], generator=order_generator
            )
            for start in range(0, cached.shape[0], training.batch_size):
                indices = window_order[start : start + training.batch_size]
                batch = cached[indices].to(device)
                targets = torch.full(
                    (batch.shape[0],),
                    float(record.age),
                    dtype=batch.dtype,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                predictions = head(batch).reshape(-1)
                loss = loss_function(predictions, targets)
                if not torch.isfinite(loss):
                    raise FrozenEncoderError("training produced a non-finite loss")
                loss.backward()
                optimizer.step()
                epoch_loss_sum += float(loss.detach().cpu()) * batch.shape[0]
                epoch_windows += batch.shape[0]

        validation_rows = predict_cached_subjects(
            head,
            validation_records,
            cache_root=cache_root,
            head_name=head_name,
            batch_size=training.batch_size,
            device=device,
        )
        validation_pearson = _pearson_from_rows(validation_rows)
        validation_mse = sum(
            (float(row["prediction"]) - float(row["age"])) ** 2
            for row in validation_rows
        ) / len(validation_rows)
        history.append(
            {
                "epoch": epoch_index + 1,
                "training_window_mse": epoch_loss_sum / epoch_windows,
                "validation_subject_mse": validation_mse,
                "validation_subject_pearson": validation_pearson,
            }
        )
        comparable = validation_pearson if math.isfinite(validation_pearson) else -float("inf")
        if comparable > best_score:
            best_score = comparable
            best_epoch = epoch_index + 1
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in head.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training.patience:
                break

    if best_state is None:
        raise FrozenEncoderError("training produced no finite validation Pearson")
    head.load_state_dict(best_state)
    runtime_seconds = time.perf_counter() - started
    peak_accelerator_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.startswith("cuda") and torch.cuda.is_available()
        else 0
    )
    parameter_count = sum(parameter.numel() for parameter in head.parameters())

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    transaction_dir = Path(
        tempfile.mkdtemp(prefix=f".{run_dir.name}-transaction-", dir=run_dir.parent)
    )
    try:
        checkpoint_path = transaction_dir / "head_checkpoint.pt"
        torch.save(
            {
                "state_dict": best_state,
                "head_name": head_name,
                "seed": seed,
                "selected_epoch": best_epoch,
                "run_identity_sha256": identity_sha256,
            },
            checkpoint_path,
        )
        manifest_body: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "head_name": head_name,
            "layer_index": required_layer,
            "seed": seed,
            "run_identity": identity,
            "run_identity_sha256": identity_sha256,
            "head_parameters": {
                "total": parameter_count,
                "trainable": sum(
                    parameter.numel()
                    for parameter in head.parameters()
                    if parameter.requires_grad
                ),
            },
            "optimizer": {
                "name": training.optimizer,
                "learning_rate": training.learning_rate,
                "weight_decay": training.weight_decay,
                "scheduler": "none",
            },
            "loss": training.loss,
            "training_unit": "window",
            "validation_unit": "subject_arithmetic_mean_of_windows",
            "checkpoint_selection": {
                "metric": training.checkpoint_metric,
                "mode": training.metric_mode,
                "tie_break": "earliest_epoch",
            },
            "selected_epoch": best_epoch,
            "selected_validation_pearson": best_score,
            "validation_history": history,
            "runtime_seconds": runtime_seconds,
            "peak_process_rss_bytes": _process_peak_rss_bytes(),
            "peak_accelerator_memory_bytes": peak_accelerator_memory,
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_sha256": _sha256_file(checkpoint_path),
        }
        manifest = {
            **manifest_body,
            "run_manifest_sha256": _canonical_sha256(manifest_body),
        }
        (transaction_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        transaction_dir.replace(run_dir)
    except Exception:
        shutil.rmtree(transaction_dir, ignore_errors=True)
        raise
    return FrozenProbeRunResult(manifest=manifest, reused=False)


def _expected_run_paths(output_root: Path) -> dict[tuple[str, int], Path]:
    return {
        (head_name, seed): output_root / head_name / f"seed-{seed}"
        for head_name in APPROVED_HEADS
        for seed in range(33, 43)
    }


def audit_frozen_probe_inventory(
    *,
    output_root: Path,
    records: Sequence[CachedSubjectRecord],
    training: TrainingContract,
) -> Mapping[str, Any]:
    """Require and hash exactly four heads by ten predeclared seeds."""

    records = validate_training_records(records)
    _strict_training_contract(training)
    if training.seeds != tuple(range(33, 43)):
        raise FrozenEncoderError("study requires exactly seeds 33 through 42")
    output_root = Path(output_root)
    expected_paths = _expected_run_paths(output_root)
    actual_heads = {
        path.name for path in output_root.iterdir() if path.is_dir()
    } if output_root.is_dir() else set()
    if actual_heads != set(APPROVED_HEADS):
        raise FrozenEncoderError(
            "exact 40-run inventory is required: "
            f"missing_heads={sorted(set(APPROVED_HEADS) - actual_heads)} "
            f"extra_heads={sorted(actual_heads - set(APPROVED_HEADS))}"
        )
    actual_paths = {
        (head_dir.name, run_dir.name)
        for head_dir in output_root.iterdir()
        if head_dir.is_dir()
        for run_dir in head_dir.iterdir()
        if run_dir.is_dir()
    } if output_root.is_dir() else set()
    expected_names = {
        (head_name, f"seed-{seed}") for head_name, seed in expected_paths
    }
    if actual_paths != expected_names:
        raise FrozenEncoderError(
            "exact 40-run inventory is required: "
            f"missing={sorted(expected_names - actual_paths)} "
            f"extra={sorted(actual_paths - expected_names)}"
        )

    runs: list[dict[str, Any]] = []
    for (head_name, seed), run_dir in expected_paths.items():
        identity_sha256 = _canonical_sha256(
            _run_identity(
                head_name=head_name,
                seed=seed,
                records=records,
                training=training,
            )
        )
        manifest = _load_completed_run(
            run_dir, expected_identity_sha256=identity_sha256
        )
        if manifest.get("head_name") != head_name or manifest.get("seed") != seed:
            raise FrozenEncoderError(f"run identity fields do not match: {run_dir}")
        runs.append(
            {
                "head_name": head_name,
                "seed": seed,
                "run_identity_sha256": identity_sha256,
                "run_manifest_sha256": manifest["run_manifest_sha256"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "selected_epoch": manifest["selected_epoch"],
                "head_parameter_count": manifest["head_parameters"]["trainable"],
            }
        )
    inventory_body: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "heads": list(APPROVED_HEADS),
        "seeds": list(range(33, 43)),
        "run_count": len(runs),
        "runs": runs,
    }
    inventory = {
        **inventory_body,
        "checkpoint_inventory_sha256": _canonical_sha256(inventory_body),
    }
    inventory_path = output_root / "checkpoint_inventory.json"
    if inventory_path.exists():
        try:
            existing = json.loads(inventory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FrozenEncoderError("checkpoint inventory is unreadable") from error
        if existing != inventory:
            raise FrozenEncoderError("checkpoint inventory does not match the exact runs")
        return existing

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=output_root,
            prefix=".checkpoint-inventory-",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(inventory, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, inventory_path)
    except FileExistsError as error:
        raise FrozenEncoderError("checkpoint inventory already exists") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return inventory


def train_frozen_probe_study(
    *,
    records: Sequence[CachedSubjectRecord],
    cache_root: Path,
    output_root: Path,
    training: TrainingContract,
    device: str,
) -> Mapping[str, Any]:
    """Train or exactly resume the complete predeclared 4x10 run matrix."""

    records = validate_training_records(records)
    if training.seeds != tuple(range(33, 43)):
        raise FrozenEncoderError("study requires exactly seeds 33 through 42")
    output_root = Path(output_root)
    if not output_root.is_absolute():
        raise FrozenEncoderError("frozen-probe output_root must be absolute")
    for head_name in APPROVED_HEADS:
        for seed in training.seeds:
            train_frozen_probe_run(
                head_name=head_name,
                seed=seed,
                records=records,
                cache_root=cache_root,
                run_dir=output_root / head_name / f"seed-{seed}",
                training=training,
                device=device,
            )
    return audit_frozen_probe_inventory(
        output_root=output_root,
        records=records,
        training=training,
    )
