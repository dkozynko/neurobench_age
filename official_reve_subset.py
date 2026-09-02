"""Run the official NeuralBench REVE Age baseline on a manifest or full root.

The public NeuralBench CLI discovers every recording under ``DATA_DIR``. This
module keeps the official experiment unchanged: manifest mode replaces only
the HBN study's timeline iterator with rows from the canonical subset, while
full-data mode leaves official discovery intact and records the memoized
timelines for audit. Strict evaluation records validation Pearson after each
training epoch and withholds the test set until an explicit one-time finalist
gate. An explicit legacy mode retains the historical read-only test pass for
parity diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import shlex
import shutil
import sys
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from selective_hbn_download import (
    INCLUDE_PATTERNS,
    RELEASES as SELECTIVE_RELEASES,
    SELECTIVE_TASK,
    _audit_release,
    _build_provenance_payload,
    _current_provenance_paths,
)

LOGGER = logging.getLogger(__name__)

try:
    from lightning.pytorch.callbacks import Callback as LightningCallback
except ImportError:  # Keep manifest-only helpers importable without Lightning.
    class LightningCallback:  # type: ignore[no-redef]
        """Fallback base used only when the optional official stack is absent."""



# ---------------------------------------------------------------------------
# Manifest and diagnostic metric helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSource:
    """Resolved input source for one official Age run."""

    data_mode: str
    data_root: Path
    manifest_path: Path | None
    manifest_sha256: str | None
    acquisition_provenance_path: Path | None = None
    acquisition_provenance_sha256: str | None = None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a small run artifact atomically."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolve_data_source(
    *,
    manifest_path: Path | None,
    full_data: bool,
    selective_task: bool = False,
    data_root: Path,
) -> DataSource:
    """Resolve and validate the mutually exclusive manifest/full input modes."""

    source_modes = int(manifest_path is not None) + int(full_data) + int(selective_task)
    if source_modes != 1:
        raise ValueError("exactly one of --manifest, --full-data, or --selective-task is required")
    resolved_root = data_root.resolve()
    if (full_data or selective_task) and not resolved_root.exists():
        raise FileNotFoundError(f"data root does not exist: {resolved_root}")
    if (full_data or selective_task) and not resolved_root.is_dir():
        raise NotADirectoryError(f"data root is not a directory: {resolved_root}")
    if selective_task:
        acquisition_path, acquisition_digest = _validate_selective_acquisition(resolved_root)
        return DataSource(
            data_mode="selective_task",
            data_root=resolved_root,
            manifest_path=None,
            manifest_sha256=None,
            acquisition_provenance_path=acquisition_path,
            acquisition_provenance_sha256=acquisition_digest,
        )
    if full_data:
        return DataSource(
            data_mode="full",
            data_root=resolved_root,
            manifest_path=None,
            manifest_sha256=None,
        )
    assert manifest_path is not None
    resolved_manifest = manifest_path.resolve()
    if not resolved_manifest.is_file():
        raise FileNotFoundError(f"manifest does not exist: {resolved_manifest}")
    return DataSource(
        data_mode="manifest",
        data_root=resolved_root,
        manifest_path=resolved_manifest,
        manifest_sha256=manifest_sha256(resolved_manifest),
    )


def _validate_selective_acquisition(data_root: Path) -> tuple[Path, str]:
    """Validate current selective provenance and reconcile it with disk."""

    provenance_path, digest_path = _current_provenance_paths(data_root)
    if not provenance_path.is_file() or not digest_path.is_file():
        raise FileNotFoundError(
            f"selective acquisition provenance is incomplete: {provenance_path}"
        )
    raw = provenance_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    recorded_digest = digest_path.read_text(encoding="ascii").strip()
    if recorded_digest != digest:
        raise ValueError("selective acquisition provenance digest mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("selective acquisition provenance is not valid UTF-8 JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("selective acquisition provenance must be an object")
    if payload.get("schema_version") != 1:
        raise ValueError("selective acquisition provenance has an unsupported schema")
    if payload.get("data_mode") != "selective_task":
        raise ValueError("selective acquisition provenance has the wrong data_mode")
    if payload.get("task") != SELECTIVE_TASK:
        raise ValueError("selective acquisition provenance has the wrong task")
    if payload.get("study") != "Shirazi2024Hbn":
        raise ValueError("selective acquisition provenance has the wrong study")
    if payload.get("data_root") != str(data_root.resolve()):
        raise ValueError("selective acquisition provenance has the wrong data root")
    if payload.get("expected_releases") != list(SELECTIVE_RELEASES):
        raise ValueError("selective acquisition provenance has the wrong release set")
    if payload.get("include_patterns") != list(INCLUDE_PATTERNS):
        raise ValueError("selective acquisition provenance has the wrong include patterns")
    if payload.get("complete") is not True:
        raise ValueError("selective acquisition provenance is not complete")
    requested = payload.get("requested_releases")
    if not isinstance(requested, list) or tuple(requested) != SELECTIVE_RELEASES:
        raise ValueError("complete selective acquisition must request all releases")

    audits = tuple(_audit_release(data_root, release) for release in SELECTIVE_RELEASES)
    expected_payload, expected_raw = _build_provenance_payload(
        data_root=data_root,
        requested_releases=requested,
        audits=audits,
    )
    if raw != expected_raw or payload != expected_payload:
        raise ValueError("selective acquisition provenance does not match current files")
    return provenance_path.resolve(), digest


def _copy_create_only(path: Path, data: bytes) -> None:
    """Create one immutable snapshot file, or verify an existing identical file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != data:
            raise RuntimeError(f"selective acquisition snapshot differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != data:
                raise RuntimeError(f"selective acquisition snapshot differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _copy_selective_acquisition_snapshot(
    source: DataSource,
    run_dir: Path,
) -> tuple[Path, str]:
    """Copy root acquisition evidence into a create-only per-run snapshot."""

    source_path = source.acquisition_provenance_path
    source_digest = source.acquisition_provenance_sha256
    if source_path is None or source_digest is None:
        raise RuntimeError("selective source is missing acquisition provenance")
    raw = source_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != source_digest:
        raise RuntimeError("selective acquisition provenance changed after validation")
    destination = run_dir / "selective_task_provenance.json"
    digest_destination = run_dir / "selective_task_provenance.sha256"
    _copy_create_only(destination, raw)
    _copy_create_only(digest_destination, (digest + "\n").encode("ascii"))
    return destination.resolve(), digest


def _canonical_full_data_timelines(
    timelines: object,
    *,
    expected_task: str | None = None,
) -> tuple[dict[str, str | None], ...]:
    """Normalize official study identities without applying task filters."""

    if isinstance(timelines, (str, bytes)) or not isinstance(timelines, Iterable):
        raise ValueError("full-data timelines must be an iterable of mappings")
    required = {"release", "subject", "task"}
    allowed = required | {"run"}
    normalized: list[dict[str, str | None]] = []
    seen: set[tuple[str, str, str, str | None]] = set()
    for index, timeline in enumerate(timelines):
        if not isinstance(timeline, Mapping):
            raise ValueError(f"timeline {index} must be a mapping")
        keys = set(timeline)
        if not keys.issubset(allowed):
            raise ValueError(f"timeline {index} has invalid keys: {sorted(keys - allowed)}")
        missing = required - keys
        if missing:
            raise ValueError(f"timeline {index} is missing fields: {sorted(missing)}")
        values: dict[str, str | None] = {}
        for field in ("release", "subject", "task"):
            value = timeline[field]
            if not isinstance(value, str):
                raise ValueError(f"timeline {index} field {field} must be a string")
            values[field] = value
        if expected_task is not None and values["task"] != expected_task:
            raise ValueError(
                f"timeline {index} task does not match expected task {expected_task!r}"
            )
        run = timeline.get("run")
        if run is not None and not isinstance(run, str):
            raise ValueError(f"timeline {index} field run must be a string or null")
        values["run"] = run
        identity = (values["release"], values["subject"], values["task"], values["run"])
        if identity in seen:
            raise ValueError(f"duplicate full-data timeline identity: {identity}")
        seen.add(identity)
        normalized.append(values)
    if not normalized:
        raise ValueError("full-data timeline iterator is empty")
    return tuple(normalized)


def _full_data_provenance_payload(
    *,
    data_root: Path,
    timelines: Sequence[Mapping[str, str | None]],
    data_mode: str = "full",
) -> tuple[dict[str, Any], bytes]:
    """Build the versioned full-data audit payload and its exact bytes."""

    payload: dict[str, Any] = {
        "schema_version": 1,
        "data_mode": data_mode,
        "study": "Shirazi2024Hbn",
        "data_root": str(data_root.resolve()),
        "timelines": [dict(timeline) for timeline in timelines],
        "timeline_count": len(timelines),
    }
    if data_mode == "selective_task":
        payload["task"] = SELECTIVE_TASK
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return payload, raw


def _parse_timeline_name(path: Path) -> tuple[str, str, str | None]:
    """Parse an official HBN EEG filename into task and optional run fields."""

    parts = path.stem.split("_")
    if len(parts) == 3 and parts[-1] == "eeg":
        _subject, task, _eeg = parts
        return task, None, _subject
    if len(parts) == 4 and parts[-1] == "eeg":
        _subject, task, run, _eeg = parts
        return task, run, _subject
    raise ValueError(f"unsupported HBN EEG filename: {path.name}")


def load_manifest_timelines(
    manifest_path: Path,
    data_root: Path,
) -> tuple[dict[str, Any], ...]:
    """Load and validate manifest rows as ``Shirazi2024Hbn`` timelines.

    The official study loader reconstructs the recording path from
    ``release``, ``subject``, ``task`` and ``run``.  We therefore validate the
    manifest path and yield exactly those fields, while retaining all age and
    split validation in the official participant/event loaders.
    """

    data_root = data_root.resolve()
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "release",
            "subject",
            "recording_relpath",
            "age",
            "duration_s",
            "split",
        }
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f"manifest must contain exactly the canonical Age fields: {sorted(required)}")
        for row in reader:
            relative = Path(row["recording_relpath"]).as_posix()
            recording = (data_root / relative).resolve()
            try:
                recording.relative_to(data_root)
            except ValueError as exc:
                raise ValueError(f"manifest recording escapes data root: {relative}") from exc
            if not recording.is_file():
                raise FileNotFoundError(f"missing recording: {recording}")
            if relative in seen_paths:
                raise ValueError(f"duplicate manifest recording: {relative}")
            seen_paths.add(relative)

            task, run, filename_subject = _parse_timeline_name(recording)
            if filename_subject != row["subject"]:
                raise ValueError(f"manifest subject does not match filename: {relative}")
            if task != "task-RestingState":
                raise ValueError(f"manifest contains a non-resting recording: {relative}")
            rows.append({"release": row["release"], "subject": row["subject"], "task": task, "run": run})

    if not rows:
        raise ValueError(f"manifest is empty: {manifest_path}")
    return tuple(rows)


def manifest_sha256(path: Path) -> str:
    """Return the manifest digest recorded with the run."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


EVALUATION_PROTOCOLS = ("strict", "legacy")


def validate_evaluation_options(
    evaluation_protocol: str = "strict",
    *,
    strict_final_test: bool = False,
) -> tuple[str, bool]:
    """Validate the holdout-access policy before the official stack starts."""

    if evaluation_protocol not in EVALUATION_PROTOCOLS:
        raise ValueError(
            "evaluation protocol must be one of "
            f"{EVALUATION_PROTOCOLS}; got {evaluation_protocol!r}"
        )
    if evaluation_protocol == "legacy" and strict_final_test:
        raise ValueError("--strict-final-test is valid only for strict evaluation")
    return evaluation_protocol, bool(strict_final_test)


class EpochValidationMetrics(LightningCallback):
    """Persist validation Pearson without retaining or touching test data."""

    def __init__(
        self,
        output_path: Path,
        seed: int | None = None,
        validation_loader: Any | None = None,
        inverse_transform_targets: bool = False,
    ):
        self.output_path = output_path
        self.seed = seed
        self.validation_loader = validation_loader
        self.inverse_transform_targets = bool(inverse_transform_targets)
        self.training_started = False

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        self.training_started = True

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if not self.training_started or trainer.sanity_checking:
            return

        import torch

        value = getattr(trainer, "callback_metrics", {}).get("val/pearsonr")
        if value is None:
            raise RuntimeError("strict validation did not expose val/pearsonr")
        pearson = float(torch.as_tensor(value).detach().cpu())
        if self.inverse_transform_targets:
            if self.validation_loader is None:
                raise RuntimeError("original-unit validation requires a validation loader")
            from torchmetrics.regression import PearsonCorrCoef

            scaler = getattr(pl_module, "target_scaler", None)
            if scaler is None or not callable(getattr(scaler, "inverse_transform", None)):
                raise RuntimeError("original-unit validation requires a fitted target scaler")
            metric = PearsonCorrCoef()
            was_training = bool(pl_module.training)
            pl_module.eval()
            try:
                with torch.inference_mode():
                    for batch in self.validation_loader:
                        batch = trainer.strategy.batch_to_device(batch, pl_module.device, dataloader_idx=0)
                        prediction = scaler.inverse_transform(pl_module.model_forward(batch))
                        target = scaler.inverse_transform(batch.data["target"])
                        if target.ndim == 3 and target.shape[1] == 1:
                            target = target.squeeze(1)
                        metric.update(prediction.detach().cpu(), target.detach().cpu())
                pearson = float(metric.compute().detach().cpu())
            finally:
                if was_training:
                    pl_module.train()
        if not math.isfinite(pearson):
            raise RuntimeError("strict validation produced non-finite val/pearsonr")

        record = {
            "seed": self.seed,
            "epoch": int(trainer.current_epoch + 1),
            "val/pearsonr": pearson,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _select_swa_window(
    records: Sequence[Mapping[str, Any]],
    *,
    window_size: int,
) -> list[dict[str, Any]]:
    """Return the declared late validation window without test access."""

    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 2:
        raise ValueError("SWA window_size must be an integer >= 2")
    if len(records) < window_size:
        raise ValueError(
            f"SWA requires at least {window_size} validation checkpoints; got {len(records)}"
        )
    selected = [dict(record) for record in records[-window_size:]]
    epochs = [record.get("epoch") for record in selected]
    if any(isinstance(epoch, bool) or not isinstance(epoch, int) for epoch in epochs):
        raise ValueError("SWA validation records must contain integer epochs")
    if epochs != list(range(epochs[0], epochs[0] + window_size)):
        raise ValueError("SWA validation window must contain consecutive epochs")
    return selected


def _average_state_dicts(
    state_dicts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Average floating tensors and retain the final value of discrete buffers."""

    if not state_dicts:
        raise ValueError("SWA requires at least one state dict")
    keys = tuple(state_dicts[0].keys())
    if any(tuple(state.keys()) != keys for state in state_dicts[1:]):
        raise ValueError("SWA state dicts must have identical keys")
    averaged: dict[str, Any] = {}
    import torch

    for key in keys:
        values = [state[key] for state in state_dicts]
        first = values[0]
        if not isinstance(first, torch.Tensor):
            if any(value != first for value in values[1:]):
                raise ValueError(f"SWA encountered a non-tensor state mismatch for {key!r}")
            averaged[key] = first
        elif torch.is_floating_point(first) or torch.is_complex(first):
            averaged[key] = torch.stack([value.detach().cpu() for value in values]).mean(dim=0)
        else:
            averaged[key] = values[-1].detach().cpu().clone()
    return averaged


def _resolve_swa_raw_checkpoint(trainer: Any, output_path: Path) -> Path:
    """Resolve NeuralBench's official best checkpoint, including nested runs."""

    checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
    callback_path = getattr(checkpoint_callback, "best_model_path", None)
    if callback_path:
        candidate = Path(str(callback_path))
        if candidate.name == "best.ckpt" and candidate.is_file():
            return candidate

    candidates = sorted(
        {
            path.resolve()
            for path in output_path.parent.rglob("best.ckpt")
            if path.is_file()
        }
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(
            "SWA requires the official best checkpoint; NeuralBench did not expose one"
        )
    raise RuntimeError(
        "SWA found multiple official best checkpoints and cannot choose one safely: "
        + ", ".join(str(path) for path in candidates)
    )


class SWAValidationCheckpoint(LightningCallback):
    """Average the final validation checkpoints using validation only."""

    def __init__(
        self,
        output_path: Path,
        validation_loader: Any,
        *,
        seed: int | None,
        window_size: int = 3,
    ):
        self.output_path = output_path
        self.validation_loader = validation_loader
        self.seed = seed
        self.window_size = window_size
        self.training_started = False
        self._records: list[dict[str, Any]] = []
        self._snapshot_paths: list[Path] = []

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        del trainer, pl_module
        self.training_started = True

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if not self.training_started or getattr(trainer, "sanity_checking", False):
            return
        import torch

        value = getattr(trainer, "callback_metrics", {}).get("val/pearsonr")
        if value is None:
            raise RuntimeError("SWA validation did not expose val/pearsonr")
        pearson = float(torch.as_tensor(value).detach().cpu())
        if not math.isfinite(pearson):
            raise RuntimeError("SWA validation produced non-finite val/pearsonr")
        epoch = int(trainer.current_epoch + 1)
        snapshot_dir = self.output_path.parent / ".swa_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"epoch-{epoch:04d}.pt"
        state = {
            key: value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else value
            for key, value in pl_module.state_dict().items()
        }
        torch.save({"epoch": epoch, "val/pearsonr": pearson, "state_dict": state}, snapshot_path)
        self._snapshot_paths.append(snapshot_path)
        self._records.append({"epoch": epoch, "val/pearsonr": pearson})
        while len(self._snapshot_paths) > self.window_size:
            self._snapshot_paths.pop(0).unlink(missing_ok=True)

    def _evaluate_validation(self, trainer: Any, pl_module: Any) -> float:
        import torch
        from torchmetrics.regression import PearsonCorrCoef

        metric = PearsonCorrCoef()
        was_training = bool(pl_module.training)
        pl_module.eval()
        try:
            with torch.inference_mode():
                for batch in self.validation_loader:
                    batch = trainer.strategy.batch_to_device(
                        batch,
                        pl_module.device,
                        dataloader_idx=0,
                    )
                    prediction = pl_module.model_forward(batch)
                    target = batch.data["target"]
                    if pl_module.target_scaler is not None:
                        target = pl_module.target_scaler.transform(target)
                    if target.ndim == 3 and target.shape[1] == 1:
                        target = target.squeeze(1)
                    metric.update(prediction.detach().cpu(), target.detach().cpu())
            score = float(metric.compute().detach().cpu())
        finally:
            if was_training:
                pl_module.train()
        if not math.isfinite(score):
            raise RuntimeError("SWA validation produced a non-finite averaged Pearson")
        return score

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        if not self.training_started:
            return
        import torch

        selected_records = _select_swa_window(self._records, window_size=self.window_size)
        selected_epochs = [int(record["epoch"]) for record in selected_records]
        snapshots = []
        for path in self._snapshot_paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            snapshots.append(payload)
        snapshots_by_epoch = {int(payload["epoch"]): payload for payload in snapshots}
        if set(selected_epochs) != set(snapshots_by_epoch):
            raise RuntimeError("SWA snapshot window does not match validation window")
        averaged_state = _average_state_dicts(
            [snapshots_by_epoch[epoch]["state_dict"] for epoch in selected_epochs]
        )
        pl_module.load_state_dict(averaged_state, strict=True)
        averaged_score = self._evaluate_validation(trainer, pl_module)
        synthetic_epoch = max(int(record["epoch"]) for record in self._records) + 1
        history_record = {
            "seed": self.seed,
            "epoch": synthetic_epoch,
            "val/pearsonr": averaged_score,
            "checkpoint_kind": "swa_weight_average",
            "source_epochs": selected_epochs,
        }
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(history_record, sort_keys=True) + "\n")

        raw_checkpoint = _resolve_swa_raw_checkpoint(trainer, self.output_path)
        raw_copy = self.output_path.parent / "best_raw.ckpt"
        if not raw_copy.exists():
            shutil.copyfile(raw_checkpoint, raw_copy)
        checkpoint = torch.load(raw_copy, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError("SWA official checkpoint payload is not a mapping")
        checkpoint = dict(checkpoint)
        checkpoint["state_dict"] = averaged_state
        checkpoint["epoch"] = synthetic_epoch - 1
        checkpoint["swa_source_epochs"] = selected_epochs
        temporary = raw_checkpoint.with_name(".best.ckpt.swa.tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(raw_checkpoint)
        metadata = {
            "schema_version": 1,
            "method": "weight_average",
            "window_size": self.window_size,
            "seed": self.seed,
            "source_epochs": selected_epochs,
            "source_validation_pearson": [record["val/pearsonr"] for record in selected_records],
            "selected_epoch": synthetic_epoch,
            "selected_val_pearsonr": averaged_score,
            "checkpoint_path": str(raw_checkpoint.resolve()),
            "raw_checkpoint_path": str(raw_copy.resolve()),
            "test_data_accessed": False,
        }
        _replace_json_atomic(self.output_path.parent / "averaged_checkpoint_metadata.json", metadata)
        for path in self._snapshot_paths:
            path.unlink(missing_ok=True)
        try:
            (self.output_path.parent / ".swa_snapshots").rmdir()
        except OSError:
            pass


class H7GradientRouteAudit(LightningCallback):
    """Record route gradients without changing the training computation."""

    def __init__(
        self,
        output_path: Path,
        *,
        seed: int | None,
        mean_gradient_scale: float,
        correction_gradient_scale: float,
        max_records_per_epoch: int = 4,
    ):
        self.output_path = output_path
        self.seed = seed
        self.mean_gradient_scale = float(mean_gradient_scale)
        self.correction_gradient_scale = float(correction_gradient_scale)
        self.max_records_per_epoch = int(max_records_per_epoch)
        self.training_started = False
        self._records: list[dict[str, Any]] = []
        self._batch_count = 0

    @staticmethod
    def _find_head(pl_module: Any) -> Any:
        for module in pl_module.modules():
            if module.__class__.__name__ == "MeanRichStatsGradientRoutesHead":
                return module
        raise RuntimeError("H7 model did not expose MeanRichStatsGradientRoutesHead")

    @staticmethod
    def _norm(value: Any) -> float:
        import torch

        if value is None:
            return 0.0
        tensor = torch.as_tensor(value).detach()
        if not torch.isfinite(tensor).all():
            raise RuntimeError("H7 gradient audit saw a non-finite gradient")
        return float(torch.linalg.vector_norm(tensor).cpu())

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        del trainer, pl_module
        self.training_started = True

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        del trainer, pl_module
        self._records.clear()
        self._batch_count = 0

    def on_after_backward(self, trainer: Any, pl_module: Any) -> None:
        if not self.training_started or len(self._records) >= self.max_records_per_epoch:
            return
        head = self._find_head(pl_module)
        route_tensors = getattr(head, "_last_gradient_route_tensors", None)
        if not isinstance(route_tensors, Mapping):
            raise RuntimeError("H7 head did not expose gradient route tensors")
        mean_grad = self._norm(route_tensors["mean"].grad)
        correction_grad = self._norm(route_tensors["statistics"].grad)
        mean_route_grad = self._norm(route_tensors["mean_route"].grad)
        correction_route_grad = self._norm(route_tensors["correction_route"].grad)
        head_gradients = {
            name: self._norm(parameter.grad)
            for name, parameter in head.named_parameters()
            if parameter.grad is not None
        }
        model = getattr(pl_module, "model", None)
        encoder = getattr(model, "encoder", None)
        encoder_norms = (
            [self._norm(parameter.grad) for parameter in encoder.parameters() if parameter.grad is not None]
            if encoder is not None
            else []
        )
        self._records.append(
            {
                "epoch": int(getattr(trainer, "current_epoch", 0)) + 1,
                "batch_idx": self._batch_count,
                "mean_route_gradient_norm": mean_grad,
                "correction_route_gradient_norm": correction_grad,
                "mean_route_tensor_gradient_norm": mean_route_grad,
                "correction_route_tensor_gradient_norm": correction_route_grad,
                "head_gradient_norms": head_gradients,
                "encoder_gradient_norm": math.sqrt(sum(value * value for value in encoder_norms)),
                "finite": True,
            }
        )
        self._batch_count += 1

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        del pl_module
        if not self.training_started or getattr(trainer, "sanity_checking", False):
            return
        history: list[dict[str, Any]] = []
        if self.output_path.is_file():
            try:
                loaded = json.loads(self.output_path.read_text(encoding="utf-8"))
                if isinstance(loaded, Mapping) and isinstance(loaded.get("records"), list):
                    history = [dict(item) for item in loaded["records"] if isinstance(item, Mapping)]
            except (OSError, json.JSONDecodeError):
                history = []
        history.extend(self._records)
        payload = {
            "schema_version": 1,
            "measurement_point": "on_after_backward_before_optimizer_step",
            "batch_provenance": "official_training_batch_only",
            "seed": self.seed,
            "mean_gradient_scale": self.mean_gradient_scale,
            "correction_gradient_scale": self.correction_gradient_scale,
            "expected_route_gradient_ratios": {
                "mean": self.mean_gradient_scale,
                "correction": self.correction_gradient_scale,
            },
            "records": history,
            "finite": all(bool(item.get("finite", False)) for item in history),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.output_path)


class ReliabilityGateAudit(LightningCallback):
    """Record H2 gate health from training and validation forwards only."""

    def __init__(self, output_path: Path, seed: int | None = None, *, alpha_max: float = 0.5):
        self.output_path = output_path
        self.seed = seed
        self.alpha_max = float(alpha_max)
        self.training_started = False
        self._training_values: list[float] = []
        self._training_subject_ids: list[Any] = []
        self._audit_batches: list[Any] = []
        self._validation_values: list[float] = []

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        self.training_started = True

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self._training_values.clear()
        self._training_subject_ids.clear()
        self._audit_batches.clear()
        self._validation_values.clear()

    @staticmethod
    def _gate_snapshot(pl_module: Any) -> Any:
        for module in pl_module.modules():
            values = getattr(module, "_last_gate_values", None)
            if values is not None:
                return values
        return None

    @staticmethod
    def _batch_subject_ids(batch: Any, count: int) -> list[Any]:
        data = getattr(batch, "data", batch)
        if not isinstance(data, Mapping):
            return [None] * count
        values = next(
            (data.get(key) for key in ("subject_ids", "subject_id", "subject", "participant_id") if key in data),
            None,
        )
        if values is None:
            return [None] * count
        if hasattr(values, "detach"):
            values = values.detach().cpu().reshape(-1).tolist()
        else:
            values = list(values)
        if len(values) != count:
            return [None] * count
        return values

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch_idx
        if not self.training_started or getattr(trainer, "sanity_checking", False):
            return
        values = self._gate_snapshot(pl_module)
        if values is None:
            return
        values = values.detach().cpu().reshape(-1)
        self._training_values.extend(float(value) for value in values.tolist())
        self._training_subject_ids.extend(self._batch_subject_ids(batch, len(values)))
        if len(self._audit_batches) < 8:
            data = getattr(batch, "data", None)
            if isinstance(data, Mapping):
                import copy
                from types import SimpleNamespace

                snapshot = {}
                for key, value in data.items():
                    snapshot[key] = value.detach().cpu().clone() if hasattr(value, "detach") else copy.deepcopy(value)
                self._audit_batches.append(SimpleNamespace(data=snapshot))

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del outputs, batch, batch_idx, dataloader_idx
        if not self.training_started or getattr(trainer, "sanity_checking", False):
            return
        values = self._gate_snapshot(pl_module)
        if values is not None:
            self._validation_values.extend(float(value) for value in values.detach().cpu().reshape(-1).tolist())

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if not self.training_started or getattr(trainer, "sanity_checking", False):
            return
        if not self._validation_values:
            raise RuntimeError("reliability gate audit saw no validation gate values")
        from reve_upstream_heads import audit_reliability_gates

        validation = audit_reliability_gates(self._validation_values, alpha_max=self.alpha_max)
        if validation["nonfinite_count"]:
            raise RuntimeError("reliability gate audit saw non-finite validation gate values")
        # Re-evaluate a bounded, training-only cache after validation while the
        # current epoch's parameters are fixed.  This gives the identity proxy
        # a stable model state without starting another multi-worker loader.
        fixed_training_values: list[float] = []
        fixed_training_subject_ids: list[Any] = []
        if self._audit_batches and hasattr(pl_module, "model_forward"):
            import torch

            was_training = bool(pl_module.training)
            pl_module.eval()
            try:
                with torch.inference_mode():
                    for batch in self._audit_batches:
                        from types import SimpleNamespace

                        batch_data = getattr(batch, "data", batch)
                        batch = SimpleNamespace(
                            data={
                                key: value.to(pl_module.device)
                                if hasattr(value, "to")
                                else value
                                for key, value in batch_data.items()
                            }
                        )
                        pl_module.model_forward(batch)
                        values = self._gate_snapshot(pl_module)
                        if values is None:
                            continue
                        values = values.detach().cpu().reshape(-1)
                        fixed_training_values.extend(float(value) for value in values.tolist())
                        fixed_training_subject_ids.extend(
                            self._batch_subject_ids(batch, len(values))
                        )
            finally:
                if was_training:
                    pl_module.train()
        if fixed_training_values:
            training = audit_reliability_gates(
                fixed_training_values,
                fixed_training_subject_ids,
                alpha_max=self.alpha_max,
            )
        else:
            training = {
                "sample_count": 0,
                "raw_sample_count": 0,
                "nonfinite_count": 0,
                "audit_valid": False,
                "gate_mean": None,
                "gate_std": None,
                "low_saturation_fraction": None,
                "high_saturation_fraction": None,
                "combined_saturation_fraction": None,
                "alpha_max": self.alpha_max,
                "saturation_epsilon": 1e-3,
                "eta_squared": 0.0,
                "eta_group_count": 0,
                "eta_sample_count": 0,
                "eta_valid": False,
                "eta_reason": "fixed_state_replay_unavailable",
            }
        record = {
            "seed": self.seed,
            "epoch": int(trainer.current_epoch + 1),
            "validation": validation,
            "training": training,
            "validation_gate_mean": validation["gate_mean"],
            "validation_gate_std": validation["gate_std"],
            "validation_low_saturation_fraction": validation["low_saturation_fraction"],
            "validation_high_saturation_fraction": validation["high_saturation_fraction"],
            "validation_combined_saturation_fraction": validation["combined_saturation_fraction"],
            "training_eta_squared": training["eta_squared"] if training is not None else None,
            "training_eta_valid": training["eta_valid"] if training is not None else False,
            "training_fixed_state_replay": bool(fixed_training_values),
        }
        early_stop_reasons = []
        if validation["nonfinite_count"]:
            early_stop_reasons.append("nonfinite_validation_gate")
        if validation["gate_std"] < 1e-3:
            early_stop_reasons.append("validation_gate_std_below_floor")
        if validation["combined_saturation_fraction"] > 0.95:
            early_stop_reasons.append("validation_gate_saturation_above_threshold")
        if training["eta_valid"] and training["eta_squared"] > 0.50:
            early_stop_reasons.append("training_identity_proxy_above_threshold")
        record["early_stop_reasons"] = early_stop_reasons
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if early_stop_reasons:
            trainer.should_stop = True


class H6GateConsistencyMetrics(LightningCallback):
    """Persist H6's training-only gate consistency without new forwards."""

    def __init__(
        self,
        output_path: Path,
        seed: int | None = None,
        *,
        lambda_gate: float,
        noise_scale: float,
    ):
        self.output_path = output_path
        self.seed = seed
        self.lambda_gate = float(lambda_gate)
        self.noise_scale = float(noise_scale)
        self.training_started = False
        self._values: list[float] = []
        self._sample_count = 0
        self._batch_count = 0

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        del trainer, pl_module
        self.training_started = True

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        del trainer, pl_module
        self._values.clear()
        self._sample_count = 0
        self._batch_count = 0

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, outputs, batch_idx
        if not self.training_started:
            return
        value = getattr(pl_module, "_h6_last_gate_consistency", None)
        if value is None:
            return
        import torch

        value = float(torch.as_tensor(value).detach().cpu())
        if not math.isfinite(value):
            raise RuntimeError("H6 gate consistency metric is non-finite")
        self._values.append(value)
        data = getattr(batch, "data", batch)
        if isinstance(data, Mapping) and hasattr(data.get("target"), "shape"):
            self._sample_count += int(data["target"].shape[0])
        else:
            self._sample_count += 1
        self._batch_count += 1

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        del pl_module
        if not self.training_started or getattr(trainer, "sanity_checking", False):
            return
        if not self._values:
            raise RuntimeError("H6 produced no training gate consistency values")
        record = {
            "seed": self.seed,
            "epoch": int(trainer.current_epoch + 1),
            "lambda_gate": self.lambda_gate,
            "noise_scale": self.noise_scale,
            "augmentation": "identity_plus_bounded_gaussian_noise",
            "batch_count": self._batch_count,
            "sample_count": self._sample_count,
            "gate_consistency": float(sum(self._values) / len(self._values)),
            "weighted_gate_consistency": float(
                self.lambda_gate * sum(self._values) / len(self._values)
            ),
            "finite": True,
        }
        history: list[dict[str, Any]] = []
        if self.output_path.is_file():
            loaded = json.loads(self.output_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping) and isinstance(loaded.get("history"), list):
                history = [dict(item) for item in loaded["history"] if isinstance(item, Mapping)]
        history.append(record)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps({"history": history}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class EpochTestPearson(LightningCallback):
    """Lightning callback that reports test Pearson after each train epoch."""

    def __init__(
        self,
        test_loader: Any,
        output_path: Path | None = None,
        seed: int | None = None,
    ):
        self.test_loader = test_loader
        self.output_path = output_path
        self.seed = seed
        self.training_started = False

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        self.training_started = True

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if not self.training_started or trainer.sanity_checking:
            return

        import torch
        from torchmetrics.regression import PearsonCorrCoef

        # Lightning exposes the validation metrics collected by the official
        # validation loop on ``callback_metrics``.  Copy the monitored value
        # for observability only; it never affects checkpoint selection or the
        # diagnostic test pass below.
        callback_metrics = getattr(trainer, "callback_metrics", {})
        validation_value = callback_metrics.get("val/pearsonr")
        if validation_value is None:
            validation_pearsonr = None
        else:
            validation_pearsonr = float(torch.as_tensor(validation_value).detach().cpu())

        was_training = pl_module.training
        pl_module.eval()
        # Keep the diagnostic metric on CPU. Lightning's strategy may expose
        # ``pl_module.device`` before the callback's first batch transfer is
        # finalized; updating CPU metric state with CUDA predictions then
        # raises a device-mismatch error. The test set is small enough that
        # this diagnostic-only CPU transfer is negligible.
        metric = PearsonCorrCoef()
        with torch.inference_mode():
            for batch_index, batch in enumerate(self.test_loader):
                batch = trainer.strategy.batch_to_device(batch, pl_module.device, dataloader_idx=0)
                y_pred = pl_module.model_forward(batch)
                y_true = batch.data["target"]
                if pl_module.target_scaler is not None:
                    y_true = pl_module.target_scaler.transform(y_true)
                if y_true.ndim == 3 and y_true.shape[1] == 1:
                    y_true = y_true.squeeze(1)
                metric.update(y_pred.detach().cpu(), y_true.detach().cpu())

        score = float(metric.compute().detach().cpu())
        record = {
            "seed": self.seed,
            "epoch": int(trainer.current_epoch + 1),
            "val/pearsonr": validation_pearsonr,
            "test_pearsonr": score,
        }
        message = (
            "EPOCH_TEST "
            f"epoch={record['epoch']} "
            f"val/pearsonr={validation_pearsonr if validation_pearsonr is not None else 'unavailable'} "
            f"test/pearsonr={record['test_pearsonr']:.12f}"
        )
        LOGGER.info(message)
        print(message, flush=True)
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        if was_training:
            pl_module.train()


def _set_frozen_experiment_field(experiment: Any, name: str, value: Any) -> None:
    """Set a declared NeuralBench field after ``exca`` freezes the model.

    ``exca`` installs a guarded ``__setattr__`` on the Pydantic Experiment
    before calling ``Experiment.run``. The official lifecycle still allows
    these fields to be customized during ``setup_run``; bypass only that
    guard while preserving the normal declared-field storage.
    """

    object.__setattr__(experiment, name, value)


def _load_reve_helpers() -> Any:
    """Import the head module in both script and package execution modes."""

    try:
        import reve_upstream_heads as reve
    except ImportError:  # Package-style invocation: ``python -m ...``.
        from . import reve_upstream_heads as reve
    return reve


# ---------------------------------------------------------------------------
# Temporary NeuralBench patches and tuning metadata
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Artifact collection, failure diagnostics, and CLI execution
# ---------------------------------------------------------------------------


def validate_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    """Validate model seeds while preserving their explicit order."""

    resolved = tuple(int(seed) for seed in seeds)
    if not resolved:
        raise ValueError("at least one seed is required")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"seeds must be unique, got {resolved}")
    return resolved


def _ensure_fresh_strict_run_dir(run_dir: Path) -> None:
    """Fail closed instead of mixing a new strict attempt with old evidence."""

    if not run_dir.exists():
        return
    if any(run_dir.iterdir()):
        raise RuntimeError(
            "strict run directory must be empty before a new attempt: "
            f"{run_dir}"
        )


def _get_attr_or_key(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_run_artifacts(output_dir: Path) -> list[dict[str, Any]]:
    """Describe resolved configs, checkpoints, and raw test predictions."""

    records: list[dict[str, Any]] = []
    gate_audit = output_dir / "gate_validation_audit.jsonl"
    for config_path in sorted(output_dir.rglob("config.yaml")):
        run_dir = config_path.parent
        checkpoints = sorted(run_dir.glob("*.ckpt"))
        checkpoint = next((path for path in checkpoints if path.name == "best.ckpt"), checkpoints[0] if checkpoints else None)
        prediction_dir = run_dir / "test_predictions"
        prediction_files = []
        if prediction_dir.is_dir():
            for path in sorted(prediction_dir.rglob("*")):
                if path.is_file():
                    prediction_files.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)})
        records.append(
            {
                "run_dir": str(run_dir),
                "resolved_config": {
                    "path": str(config_path),
                    "sha256": _sha256_file(config_path),
                },
                "selected_checkpoint": (
                    {
                        "path": str(checkpoint),
                        "sha256": _sha256_file(checkpoint),
                    }
                    if checkpoint is not None
                    else None
                ),
                "raw_test_predictions": prediction_files,
                "reliability_gate_audit": (
                    {
                        "path": str(gate_audit),
                        "sha256": _sha256_file(gate_audit),
                    }
                    if gate_audit.is_file()
                    else None
                ),
                "run_metadata": str(run_dir / "run_metadata.json")
                if (run_dir / "run_metadata.json").is_file()
                else None,
            }
        )
    return records


def write_failure_diagnostics(
    output_dir: Path,
    error: BaseException,
    *,
    launch_command: str,
    metadata: Mapping[str, Any],
) -> Path:
    """Persist a structured failure record without masking the original error."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "failure.json"
    payload = {
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "launch_command": launch_command,
        "metadata": dict(metadata),
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _metadata_values_are_finite(value: Any) -> bool:
    """Return whether every numeric value in JSON-style metadata is finite."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_metadata_values_are_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_metadata_values_are_finite(item) for item in value)
    return False


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON evidence file and require an object payload."""

    if not path.is_file():
        raise RuntimeError(f"strict evidence file is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"strict evidence file is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"strict evidence file must contain an object: {path}")
    return payload


def _strict_report_fields(
    *,
    selection_path: Path,
    results: Sequence[Mapping[str, Any]],
    strict_final_test: bool,
) -> dict[str, Any]:
    """Validate strict evidence and return the stable per-seed report fields."""

    selection = _load_json_object(selection_path)
    if selection.get("evaluation_protocol") != "strict":
        raise RuntimeError("strict selection record has the wrong evaluation protocol")
    if bool(selection.get("strict_final_test")) != bool(strict_final_test):
        raise RuntimeError("strict selection gate does not match the run")
    if selection.get("selection_monitor") != "val/pearsonr":
        raise RuntimeError("strict selection monitor must be val/pearsonr")
    if selection.get("selection_mode") != "max":
        raise RuntimeError("strict selection mode must be max")
    data_mode = selection.get("data_mode", "manifest")
    if data_mode not in {"manifest", "full", "selective_task"}:
        raise RuntimeError("strict selection has an invalid data_mode")

    required_hash_fields = (
        "checkpoint_sha256",
        "official_config_sha256",
        "provenance_sha256",
        "validation_history_sha256",
    )
    for field in required_hash_fields:
        value = selection.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"strict selection has invalid {field}")
    for path_field, hash_field in (
        ("checkpoint_path", "checkpoint_sha256"),
        ("official_config_path", "official_config_sha256"),
        ("provenance_path", "provenance_sha256"),
        ("validation_history_path", "validation_history_sha256"),
    ):
        raw_path = selection.get(path_field)
        if not isinstance(raw_path, str):
            raise RuntimeError(f"strict selection is missing {path_field}")
        path = Path(raw_path)
        if not path.is_file():
            raise RuntimeError(f"strict provenance file is missing: {path}")
        if _sha256_file(path) != selection[hash_field]:
            raise RuntimeError(f"strict provenance hash changed: {path}")

    if data_mode == "full":
        timeline_count = selection.get("timeline_count")
        if isinstance(timeline_count, bool) or not isinstance(timeline_count, int) or timeline_count < 1:
            raise RuntimeError("full-data strict selection has an invalid timeline_count")
        if selection.get("manifest_path") is not None or selection.get("manifest_sha256") is not None:
            raise RuntimeError("full-data strict selection must not bind a manifest")
    elif data_mode == "selective_task":
        timeline_count = selection.get("timeline_count")
        if isinstance(timeline_count, bool) or not isinstance(timeline_count, int) or timeline_count < 1:
            raise RuntimeError("selective-task strict selection has an invalid timeline_count")
        if selection.get("manifest_path") is not None or selection.get("manifest_sha256") is not None:
            raise RuntimeError("selective-task strict selection must not bind a manifest")
        acquisition_path = selection.get("acquisition_provenance_path")
        acquisition_digest = selection.get("acquisition_provenance_sha256")
        if (
            not isinstance(acquisition_path, str)
            or not isinstance(acquisition_digest, str)
            or len(acquisition_digest) != 64
        ):
            raise RuntimeError("selective-task strict selection is missing acquisition provenance")
        acquisition_file = Path(acquisition_path)
        if not acquisition_file.is_file() or _sha256_file(acquisition_file) != acquisition_digest:
            raise RuntimeError(f"selective acquisition provenance changed: {acquisition_file}")
        acquisition_sidecar = acquisition_file.with_suffix(".sha256")
        if (
            not acquisition_sidecar.is_file()
            or acquisition_sidecar.read_text(encoding="ascii").strip() != acquisition_digest
        ):
            raise RuntimeError("selective acquisition provenance sidecar mismatch")
    else:
        manifest_path = selection.get("manifest_path")
        manifest_sha256 = selection.get("manifest_sha256")
        if not isinstance(manifest_path, str) or not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
            raise RuntimeError("manifest strict selection is missing manifest provenance")
        manifest_file = Path(manifest_path)
        if not manifest_file.is_file() or _sha256_file(manifest_file) != manifest_sha256:
            raise RuntimeError(f"strict manifest provenance changed: {manifest_file}")

    seed = selection.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RuntimeError("strict selection seed is invalid")
    selected_epoch = selection.get("selected_epoch")
    selected_value = selection.get("selected_val_pearsonr")
    checkpoint_epoch = selection.get("checkpoint_epoch_zero_based")
    if (
        isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, int)
        or selected_epoch < 1
        or isinstance(checkpoint_epoch, bool)
        or not isinstance(checkpoint_epoch, int)
        or checkpoint_epoch < 0
        or selected_epoch != checkpoint_epoch + 1
        or isinstance(selected_value, bool)
        or not isinstance(selected_value, (int, float))
        or not math.isfinite(float(selected_value))
    ):
        raise RuntimeError("strict selection has invalid selected validation fields")

    expected_status = "sealed" if strict_final_test else "withheld"
    if selection.get("test_status") != expected_status:
        raise RuntimeError("strict selection test status does not match the run")

    report: dict[str, Any] = {
        "evaluation_protocol": "strict",
        "data_mode": data_mode,
        "timeline_count": selection.get("timeline_count"),
        "strict_final_test": bool(strict_final_test),
        "selection_monitor": "val/pearsonr",
        "selection_mode": "max",
        "seed": seed,
        "head_variant": selection.get("head_variant"),
        "selected_epoch": selected_epoch,
        "checkpoint_epoch_zero_based": checkpoint_epoch,
        "selected_val_pearsonr": float(selected_value),
        **{field: selection[field] for field in (
            "checkpoint_path",
            "checkpoint_sha256",
            "official_config_path",
            "official_config_sha256",
            "manifest_path",
            "manifest_sha256",
            "provenance_path",
            "provenance_sha256",
            "acquisition_provenance_path",
            "acquisition_provenance_sha256",
            "validation_history_path",
            "validation_history_sha256",
        )},
        "validation_metrics": selection["validation_history_path"],
        "selection_record": str(selection_path.resolve()),
        "test_status": expected_status,
    }

    start_path = selection_path.parent / "test_started.json"
    completed_path = selection_path.parent / "test_completed.json"
    if not strict_final_test:
        if results:
            raise RuntimeError("strict validation-only run unexpectedly returned test results")
        if start_path.exists() or completed_path.exists():
            raise RuntimeError("strict validation-only run consumed test evidence")
        report.update(
            {
                "test_evaluations": 0,
                "checkpoint_integrity_verified": False,
            }
        )
        return report

    if not start_path.is_file() or not completed_path.is_file():
        raise RuntimeError("strict final test is missing start/completion evidence")
    start = _load_json_object(start_path)
    completed = _load_json_object(completed_path)
    selection_sha = _sha256_file(selection_path)
    checkpoint_sha = selection["checkpoint_sha256"]
    if (
        start.get("selection_sha256") != selection_sha
        or start.get("checkpoint_sha256") != checkpoint_sha
        or start.get("test_evaluations") != 1
    ):
        raise RuntimeError("strict test start marker does not bind the selection")
    if (
        completed.get("selection_sha256") != selection_sha
        or completed.get("checkpoint_sha256_after_test") != checkpoint_sha
        or completed.get("test_evaluations") != 1
    ):
        raise RuntimeError("strict test completion marker does not bind the selection")
    test_pearson = _runtime._extract_official_test_pearson(results)
    completed_value = completed.get("test_pearsonr")
    if (
        isinstance(completed_value, bool)
        or not isinstance(completed_value, (int, float))
        or not math.isfinite(float(completed_value))
        or float(completed_value) != test_pearson
    ):
        raise RuntimeError("strict test completion marker does not match test/pearsonr")
    report.update(
        {
            "test_pearsonr": test_pearson,
            "test_evaluations": 1,
            "checkpoint_integrity_verified": True,
        }
    )
    return report


def _strict_summary_fields(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate strict per-seed reports without inventing withheld test data."""

    if not reports:
        raise RuntimeError("strict summary requires at least one completed seed")
    protocols = {report.get("evaluation_protocol") for report in reports}
    if protocols != {"strict"}:
        raise RuntimeError("strict summary contains a non-strict report")
    gates = {bool(report.get("strict_final_test")) for report in reports}
    if len(gates) != 1:
        raise RuntimeError("strict summary mixes validation-only and final-test reports")
    gate = gates.pop()
    has_source_metadata = any(
        any(field in report for field in ("data_mode", "provenance_path", "manifest_path"))
        for report in reports
    )
    if has_source_metadata:
        data_modes = {report.get("data_mode", "manifest") for report in reports}
        if len(data_modes) != 1 or data_modes not in ({"manifest"}, {"full"}, {"selective_task"}):
            raise RuntimeError("strict summary mixes data sources")
        data_mode = next(iter(data_modes))
    else:
        data_mode = "manifest"
    selected_epochs: dict[str, int] = {}
    selected_values: dict[str, float] = {}
    for report in reports:
        seed = report.get("seed")
        epoch = report.get("selected_epoch")
        value = report.get("selected_val_pearsonr")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RuntimeError("strict summary contains an invalid seed")
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise RuntimeError("strict summary contains an invalid selected epoch")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError("strict summary contains an invalid validation metric")
        selected_epochs[str(seed)] = epoch
        selected_values[str(seed)] = float(value)
    summary: dict[str, Any] = {
        "evaluation_protocol": "strict",
        "data_mode": data_mode,
        "strict_final_test": gate,
        "test_status": "completed" if gate else "withheld",
        "completed_seed_count": len(reports),
        "selected_epoch_by_seed": selected_epochs,
        "selected_val_pearson_by_seed": selected_values,
        "mean_selected_val_pearson": sum(selected_values.values()) / len(selected_values),
    }
    if has_source_metadata:
        if data_mode in {"full", "selective_task"}:
            timeline_counts: dict[str, int] = {}
            provenance_paths: dict[str, str] = {}
            provenance_hashes: dict[str, str] = {}
            acquisition_paths: dict[str, str] = {}
            acquisition_hashes: dict[str, str] = {}
            for report in reports:
                seed = report["seed"]
                count = report.get("timeline_count")
                path = report.get("provenance_path")
                digest = report.get("provenance_sha256")
                if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                    raise RuntimeError("non-manifest strict summary has an invalid timeline_count")
                if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
                    raise RuntimeError("non-manifest strict summary is missing provenance metadata")
                timeline_counts[str(seed)] = count
                provenance_paths[str(seed)] = path
                provenance_hashes[str(seed)] = digest
                if data_mode == "selective_task":
                    acquisition_path = report.get("acquisition_provenance_path")
                    acquisition_digest = report.get("acquisition_provenance_sha256")
                    if (
                        not isinstance(acquisition_path, str)
                        or not isinstance(acquisition_digest, str)
                        or len(acquisition_digest) != 64
                    ):
                        raise RuntimeError("selective-task strict summary is missing acquisition metadata")
                    acquisition_paths[str(seed)] = acquisition_path
                    acquisition_hashes[str(seed)] = acquisition_digest
            if len(set(timeline_counts.values())) != 1:
                raise RuntimeError("full-data strict summary mixes timeline counts")
            summary.update(
                {
                    "timeline_count": next(iter(timeline_counts.values())),
                    "timeline_count_by_seed": timeline_counts,
                    "provenance_path": provenance_paths,
                    "provenance_sha256": provenance_hashes,
                    "provenance_path_by_seed": provenance_paths,
                    "provenance_sha256_by_seed": provenance_hashes,
                }
            )
            if data_mode == "selective_task":
                summary.update(
                    {
                        "acquisition_provenance_path": acquisition_paths,
                        "acquisition_provenance_sha256": acquisition_hashes,
                        "acquisition_provenance_path_by_seed": acquisition_paths,
                        "acquisition_provenance_sha256_by_seed": acquisition_hashes,
                    }
                )
        else:
            manifest_paths = {report.get("manifest_path") for report in reports}
            manifest_hashes = {report.get("manifest_sha256") for report in reports}
            if len(manifest_paths) != 1 or len(manifest_hashes) != 1:
                raise RuntimeError("manifest strict summary mixes manifest provenance")
            timeline_counts = {
                report.get("timeline_count")
                for report in reports
                if "timeline_count" in report
            }
            if timeline_counts:
                if len(timeline_counts) != 1 or None in timeline_counts:
                    raise RuntimeError("manifest strict summary mixes timeline counts")
            summary.update(
                {
                    "manifest_path": next(iter(manifest_paths)),
                    "manifest_sha256": next(iter(manifest_hashes)),
                    "provenance_path": next(iter(manifest_paths)),
                    "provenance_sha256": next(iter(manifest_hashes)),
                }
            )
            if timeline_counts:
                summary["timeline_count"] = next(iter(timeline_counts))
    if gate:
        test_values: dict[str, float] = {}
        for report in reports:
            seed = report["seed"]
            value = report.get("test_pearsonr")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError("strict final summary is missing a test metric")
            test_values[str(seed)] = float(value)
        summary["test_pearson_by_seed"] = test_values
        summary["mean_test_pearson"] = sum(test_values.values()) / len(test_values)
    return summary


def run_official_stack_smoke(
    *,
    head_variant: str,
    device: str = "cpu",
) -> dict[str, Any]:
    """Exercise the real REVE/NeuralTrain interfaces without HBN data.

    This intentionally constructs a small, randomly initialized braindecode
    REVE model. It verifies the output contract used by the production
    wrapper, including the initial positional sequence and every transformer
    layer required by ``all``. Pretrained weights and HBN recordings are not
    touched.
    """

    reve = _load_reve_helpers()

    if head_variant == "last_tuned":
        reve.validate_last_tuned_protocol(head_variant)
    elif head_variant in {"mean_linear_copy", "mean_linear_detached", "mean_linear_warmup", "mean_linear_gradient_scaled", "mean_linear_probe_scaled", "mean_anchor", "mean_residual", "mean_vector_anchor", "mean_mlp_residual", "mean_stats_residual", "mean_stats_residual_detached", "mean_stats_residual_gradient_scaled", "mean_stats_probe_scaled", "mean_stats_attention_residual", "mean_attention_gated", "global_stats_residual", "mean_rich_stats_residual", "mean_rich_stats_gradient_routes", "mean_anchor_ensemble", "mean_reliability_shrinkage", "mean_reliability_stable", "grouped_rich_stats_shrinkage", "grouped_stats_shared_gate", "temporal_pyramid_stats", "mean_covariance_residual", "multi_query_rich_stats", "mean_layer_linear", "mean_layer_mix"}:
        reve.validate_local_head_variant(head_variant)
    else:
        reve.validate_upstream_head_variant(head_variant)
    import torch
    from braindecode.models import REVE
    from neuraltrain.models.reve import _ReveWrapper

    embed_dim = 32
    depth = 2
    n_chans = 3
    n_times = 400
    model = REVE(
        n_outputs=1,
        n_chans=n_chans,
        chs_info=[{"ch_name": name} for name in ("Fp1", "Fp2", "F3")],
        n_times=n_times,
        sfreq=200.0,
        embed_dim=embed_dim,
        depth=depth,
        heads=2,
        head_dim=16,
        mlp_dim_ratio=1.0,
        attention_pooling=True,
    )
    encoder = _ReveWrapper(model, encoder_only=True).to(device)
    if head_variant not in {"last_tuned", "mean_anchor", "mean_residual", "mean_vector_anchor", "mean_stats_attention_residual", "mean_attention_gated"}:
        # Keep the official smoke variants on their exact existing RNG and
        # construction path.  Query-initialization branches need encoder
        # tokens before they can construct their explicit query.
        adapter = reve.UpstreamReveHeadModel(encoder, variant=head_variant, n_outputs=1, dropout=0.0).to(device)
    eeg = torch.randn(2, n_chans, n_times, device=device)
    positions = torch.randn(2, n_chans, 3, device=device)

    with torch.inference_mode():
        raw_layers = model(eeg, pos=positions, return_output=True)
        final = encoder(eeg, pos=positions)
    if head_variant in {"last_tuned", "mean_anchor", "mean_residual", "mean_vector_anchor", "mean_stats_attention_residual", "mean_attention_gated"}:
        if not isinstance(final, torch.Tensor) or final.ndim != 3:
            raise RuntimeError(f"{head_variant} smoke encoder did not return final tokens")
        with torch.inference_mode(False):
            query_token = final[:1].mean(dim=1, keepdim=True).detach().clone()
        if not torch.isfinite(query_token).all():
            raise RuntimeError(f"{head_variant} smoke mean-token query is not finite")
        adapter = reve.UpstreamReveHeadModel(
            encoder,
            variant=head_variant,
            n_outputs=1,
            dropout=0.0,
            query_token=query_token,
            query_initialization_metadata={
                "query_initialization": "smoke_synthetic_mean_token",
                "query_initialization_provenance": "smoke",
            },
        ).to(device)
    with torch.inference_mode():
        prediction = adapter(eeg, channel_positions=positions)

    if not isinstance(raw_layers, (list, tuple)):
        raise RuntimeError("official REVE return_output=True did not return layers")
    if len(raw_layers) != depth + 1:
        raise RuntimeError(f"official REVE layer contract changed: expected positional input plus {depth} layers, got {len(raw_layers)}")
    if tuple(prediction.shape) != (2, 1):
        raise RuntimeError(f"unexpected adapter output shape: {tuple(prediction.shape)}")

    output = {
        "head_variant": head_variant,
        "device": device,
        "token_shapes": [list(layer.shape) for layer in raw_layers],
        "final_shape": list(final.shape),
        "prediction_shape": list(prediction.shape),
        "embed_dim": embed_dim,
        "layer_count_including_initial": len(raw_layers),
        "query_initialization": getattr(adapter.head, "query_initialization", "not_applicable"),
    }
    if head_variant == "last_tuned":
        tuning_metadata = adapter.head.tuning_metadata
        output.update(
            {
                "query_initialization_provenance": tuning_metadata[
                    "query_initialization_provenance"
                ],
                "prediction_finite": bool(torch.isfinite(prediction).all().item()),
                "metadata_finite": _metadata_values_are_finite(tuning_metadata),
            }
        )
    elif head_variant in {"mean_anchor", "mean_residual", "mean_vector_anchor", "mean_stats_attention_residual", "mean_attention_gated"}:
        smoke_metadata = {
            "query_initialization": adapter.head.query_initialization,
            "query_initialization_provenance": "smoke",
        }
        output.update(
            {
                "query_initialization_provenance": smoke_metadata[
                    "query_initialization_provenance"
                ],
                "prediction_finite": bool(torch.isfinite(prediction).all().item()),
                "metadata_finite": _metadata_values_are_finite(smoke_metadata),
            }
        )
    elif head_variant == "mean_mlp_residual":
        output.update(
            {
                "prediction_finite": bool(torch.isfinite(prediction).all().item()),
            }
        )
    elif head_variant == "mean_stats_residual":
        output.update(
            {
                "prediction_finite": bool(torch.isfinite(prediction).all().item()),
            }
        )
    elif head_variant == "global_stats_residual":
        output.update(
            {
                "prediction_finite": bool(torch.isfinite(prediction).all().item()),
            }
        )
    elif head_variant in {"mean_rich_stats_residual", "mean_rich_stats_gradient_routes"}:
        output.update(
            {
                "prediction_finite": bool(torch.isfinite(prediction).all().item()),
            }
        )
    elif head_variant == "multi_query_rich_stats":
        output.update(
            {
                "prediction_finite": bool(torch.isfinite(prediction).all().item()),
                "head_metadata": adapter.head.metadata(),
            }
        )
    elif head_variant in {"grouped_rich_stats_shrinkage", "grouped_stats_shared_gate", "temporal_pyramid_stats", "mean_covariance_residual", "mean_anchor_ensemble", "mean_reliability_shrinkage", "mean_reliability_stable"}:
        output.update(
            {
                "prediction_finite": bool(torch.isfinite(prediction).all().item()),
                "head_metadata": adapter.head.metadata(),
            }
        )
    elif head_variant in {"mean_layer_linear", "mean_layer_mix"}:
        output.update(
            {
                "prediction_finite": bool(torch.isfinite(prediction).all().item()),
                "head_metadata": adapter.head.metadata(),
            }
        )
    elif head_variant == "mean_stats_residual_detached":
        output.update(
            {
                "prediction_finite": bool(torch.isfinite(prediction).all().item()),
            }
        )
    elif head_variant == "mean_linear_detached":
        output.update(
            {
                "prediction_finite": bool(torch.isfinite(prediction).all().item()),
            }
        )
    elif head_variant == "mean_linear_warmup":
        output.update(
            {
                "prediction_finite": bool(torch.isfinite(prediction).all().item()),
            }
        )
    elif head_variant == "mean_linear_gradient_scaled":
        output.update(
            {
                "prediction_finite": bool(torch.isfinite(prediction).all().item()),
            }
        )
    elif head_variant == "mean_linear_probe_scaled":
        output.update(
            {
                "prediction_finite": bool(torch.isfinite(prediction).all().item()),
            }
        )
    elif head_variant == "mean_stats_residual_gradient_scaled":
        output.update(
            {
                "prediction_finite": bool(torch.isfinite(prediction).all().item()),
            }
        )
    elif head_variant == "mean_stats_probe_scaled":
        output.update(
            {
                "prediction_finite": bool(torch.isfinite(prediction).all().item()),
            }
        )
    elif head_variant == "mean_stats_attention_residual":
        output.update(
            {
                "prediction_finite": bool(torch.isfinite(prediction).all().item()),
            }
        )
    elif head_variant == "mean_attention_gated":
        output.update(
            {
                "prediction_finite": bool(torch.isfinite(prediction).all().item()),
            }
        )
    return output




# The runtime module owns temporary monkeypatches; these wrappers retain the
# original public names and keep monkeypatch-based tests local to this facade.
try:
    from . import official_reve_runtime as _runtime
except ImportError:
    import official_reve_runtime as _runtime

_capture_test_result = _runtime._capture_test_result
_head_metadata = _runtime._head_metadata
_last_tuned_report_metadata = _runtime._last_tuned_report_metadata
_merge_last_tuned_result_metadata = _runtime._merge_last_tuned_result_metadata
_selected_validation_checkpoint_epoch = _runtime._selected_validation_checkpoint_epoch
_build_strict_selection_record = _runtime._build_strict_selection_record
_run_strict_test_phase = _runtime._run_strict_test_phase


def _hooks() -> Any:
    return sys.modules[__name__]


def _last_tuned_configure_optimizers(brain_module: Any) -> dict[str, Any]:
    return _runtime._last_tuned_configure_optimizers(brain_module, hooks=_hooks())


def _restore_last_tuned_configure_optimizers(patched_modules: list[dict[str, Any]]) -> None:
    return _runtime._restore_last_tuned_configure_optimizers(patched_modules)


def _patch_official_components(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _runtime._patch_official_components(*args, hooks=_hooks(), **kwargs)


def _restore_official_components(originals: Mapping[str, Any]) -> None:
    return _runtime._restore_official_components(originals, restore_tuned=_restore_last_tuned_configure_optimizers)


def run_official_subset(
    *,
    manifest_path: Path | None = None,
    data_root: Path,
    epoch_metrics_path: Path,
    selection_path: Path,
    config_path: Path,
    head_variant: str = "mean_linear",
    layer_index: int = -1,
    head_dropout: float = 0.0,
    mean_gradient_scale: float = 0.5,
    correction_gradient_scale: float = 1.0,
    swa_window: int = 0,
    correlation_loss_lambda: float = 0.0,
    robust_loss: str = "mse",
    target_scaler_mode: str = "none",
    seeds: Sequence[int] = (33,),
    evaluation_protocol: str = "strict",
    strict_final_test: bool = False,
    two_stage_finetune: bool = False,
    two_stage_warmup_epochs: int = 3,
    two_stage_unfreeze_last_blocks: int = 1,
    two_stage_encoder_gradient_scale: float = 0.1,
    data_mode: str = "manifest",
    provenance_path: Path | None = None,
    acquisition_provenance_path: Path | None = None,
    acquisition_provenance_sha256: str | None = None,
    timeline_count: int | None = None,
    run_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return _runtime.run_official_subset(
        manifest_path=manifest_path,
        data_root=data_root,
        epoch_metrics_path=epoch_metrics_path,
        selection_path=selection_path,
        config_path=config_path,
        head_variant=head_variant,
        layer_index=layer_index,
        head_dropout=head_dropout,
        mean_gradient_scale=mean_gradient_scale,
        correction_gradient_scale=correction_gradient_scale,
        swa_window=swa_window,
        correlation_loss_lambda=correlation_loss_lambda,
        robust_loss=robust_loss,
        target_scaler_mode=target_scaler_mode,
        seeds=seeds,
        evaluation_protocol=evaluation_protocol,
        strict_final_test=strict_final_test,
        two_stage_finetune=two_stage_finetune,
        two_stage_warmup_epochs=two_stage_warmup_epochs,
        two_stage_unfreeze_last_blocks=two_stage_unfreeze_last_blocks,
        two_stage_encoder_gradient_scale=two_stage_encoder_gradient_scale,
        data_mode=data_mode,
        provenance_path=provenance_path,
        acquisition_provenance_path=acquisition_provenance_path,
        acquisition_provenance_sha256=acquisition_provenance_sha256,
        timeline_count=timeline_count,
        run_metadata=run_metadata,
        hooks=_hooks(),
    )


def _run_experiments_synchronously(aggregator: Any) -> None:
    return _runtime._run_experiments_synchronously(aggregator)


def _write_config(
    path: Path,
    *,
    data_root: Path,
    output_dir: Path,
    data_mode: str = "manifest",
    head_variant: str | None = None,
    layer_index: int = -1,
    mean_gradient_scale: float | None = None,
    correction_gradient_scale: float | None = None,
    swa_window: int = 0,
    correlation_loss_lambda: float = 0.0,
    robust_loss: str = "mse",
    target_scaler_mode: str = "none",
    two_stage_finetune: bool = False,
    two_stage_warmup_epochs: int = 3,
    two_stage_unfreeze_last_blocks: int = 1,
    two_stage_encoder_gradient_scale: float = 0.1,
) -> None:
    cache_namespace = {
        "full": "neuralbench_official_cache_full",
        "selective_task": "neuralbench_official_cache_selective_task",
    }.get(data_mode, "neuralbench_official_cache_500")
    config = {
        "USER": "root",
        "ENTITY_NAME": "root",
        "PROJECT_NAME": "neurobench_reve_age_official",
        "CACHE_DIR": str(
            data_root / cache_namespace
        ),
        "SAVE_DIR": str(output_dir),
        "DATA_DIR": str(data_root),
        "WANDB_HOST": "",
        "SLURM_PARTITION": "",
        "SLURM_CONSTRAINT": "",
        "N_CPUS": 2,
        "CLUSTER": None,
    }
    if head_variant is not None:
        config.update(
            {
                "H7_HEAD_VARIANT": head_variant,
                "H7_LAYER_INDEX": int(layer_index),
                "H7_MEAN_GRADIENT_SCALE": mean_gradient_scale,
                "H7_CORRECTION_GRADIENT_SCALE": correction_gradient_scale,
                "SWA_WINDOW": swa_window,
                "CORRELATION_LOSS_LAMBDA": float(correlation_loss_lambda),
                "CORRELATION_LOSS_OBJECTIVE": (
                    "batch_pearson" if correlation_loss_lambda else None
                ),
                "ROBUST_LOSS": robust_loss,
                "TARGET_SCALER_MODE": target_scaler_mode,
                "TWO_STAGE_FINETUNE": bool(two_stage_finetune),
                "TWO_STAGE_WARMUP_EPOCHS": int(two_stage_warmup_epochs),
                "TWO_STAGE_UNFREEZE_LAST_BLOCKS": int(two_stage_unfreeze_last_blocks),
                "TWO_STAGE_ENCODER_GRADIENT_SCALE": float(two_stage_encoder_gradient_scale),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _resolved_head_metadata(run_dir: Path) -> dict[str, Any]:
    """Read late-bound head metadata written after the official model builds."""

    for path in sorted(run_dir.rglob("run_metadata.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metadata = payload.get("head_metadata") if isinstance(payload, Mapping) else None
        if isinstance(metadata, Mapping):
            return dict(metadata)
    return {}


def _resolved_swa_metadata(run_dir: Path) -> dict[str, Any]:
    """Read the checkpoint-averaging evidence written after training."""

    path = run_dir / "averaged_checkpoint_metadata.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="use the official Shirazi2024Hbn timeline discovery without a manifest",
    )
    parser.add_argument(
        "--selective-task",
        action="store_true",
        help="use a complete selective task-RestingState HBN acquisition",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--smoke-head",
        choices=("mean_linear_copy", "mean_linear_detached", "mean_linear_warmup", "mean_linear_gradient_scaled", "mean_linear_probe_scaled", "mean_anchor", "mean_residual", "mean_vector_anchor", "mean_mlp_residual", "mean_stats_residual", "mean_stats_residual_detached", "mean_stats_residual_gradient_scaled", "mean_stats_probe_scaled", "mean_stats_attention_residual", "mean_attention_gated", "global_stats_residual", "mean_rich_stats_residual", "mean_rich_stats_gradient_routes", "mean_anchor_ensemble", "mean_reliability_shrinkage", "mean_reliability_stable", "grouped_rich_stats_shrinkage", "grouped_stats_shared_gate", "temporal_pyramid_stats", "mean_covariance_residual", "multi_query_rich_stats", "mean_layer_linear", "mean_layer_mix", "last_avg", "last", "all", "last_tuned"),
        help="run a data-free smoke test using the installed official stack",
    )
    parser.add_argument(
        "--head-variant",
        choices=("mean_linear", "mean_linear_copy", "mean_linear_detached", "mean_linear_warmup", "mean_linear_gradient_scaled", "mean_linear_probe_scaled", "mean_anchor", "mean_residual", "mean_vector_anchor", "mean_mlp_residual", "mean_stats_residual", "mean_stats_residual_detached", "mean_stats_residual_gradient_scaled", "mean_stats_probe_scaled", "mean_stats_attention_residual", "mean_attention_gated", "global_stats_residual", "mean_rich_stats_residual", "mean_rich_stats_gradient_routes", "mean_anchor_ensemble", "mean_reliability_shrinkage", "mean_reliability_stable", "grouped_rich_stats_shrinkage", "grouped_stats_shared_gate", "temporal_pyramid_stats", "mean_covariance_residual", "multi_query_rich_stats", "mean_layer_linear", "mean_layer_mix", "last_avg", "last", "all", "last_tuned"),
        default="mean_linear",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=-1,
        help="selected transformer layer: positive 1-based or negative final-relative (-1 is final)",
    )
    parser.add_argument(
        "--evaluation-protocol",
        choices=EVALUATION_PROTOCOLS,
        default="strict",
        help="strict holdout (default) or explicit legacy epoch-level diagnostics",
    )
    parser.add_argument(
        "--strict-final-test",
        action="store_true",
        help="consume the single predeclared strict test pass after validation selection",
    )
    parser.add_argument(
        "--two-stage-finetune",
        action="store_true",
        help="freeze the REVE encoder during warm-up, then adapt only its final block",
    )
    parser.add_argument(
        "--two-stage-warmup-epochs",
        type=int,
        default=3,
        help="number of head-only epochs before final-block adaptation",
    )
    parser.add_argument(
        "--two-stage-unfreeze-last-blocks",
        type=int,
        default=1,
        help="number of final REVE transformer blocks to adapt after warm-up",
    )
    parser.add_argument(
        "--two-stage-encoder-gradient-scale",
        type=float,
        default=0.1,
        help="effective update scale for selected encoder blocks after warm-up",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[33])
    parser.add_argument(
        "--mean-gradient-scale",
        type=float,
        default=0.5,
        help="H7 mean-route encoder-gradient scale (finite value in [0, 2])",
    )
    parser.add_argument(
        "--correction-gradient-scale",
        type=float,
        default=1.0,
        help="H7 correction-route encoder-gradient scale (finite value in [0, 2])",
    )
    parser.add_argument(
        "--swa-window",
        type=int,
        choices=(0, 3, 5),
        default=0,
        help="average the final 3 or 5 validation checkpoints (strict rich-stats screen only)",
    )
    parser.add_argument(
        "--correlation-loss-lambda",
        type=float,
        choices=(0.0, 0.02, 0.05),
        default=0.0,
        help="add lambda * (1 - batch Pearson) to MSE (strict rich-stats screen only)",
    )
    parser.add_argument(
        "--robust-loss",
        choices=("mse", "smooth_l1"),
        default="mse",
        help="training loss (strict rich-stats screen only)",
    )
    parser.add_argument(
        "--target-scaler",
        choices=("none", "zscore"),
        default="none",
        help="fit a training-only target z-score scaler (strict rich-stats screen only)",
    )
    args = parser.parse_args(argv)

    try:
        args.evaluation_protocol, args.strict_final_test = validate_evaluation_options(
            args.evaluation_protocol,
            strict_final_test=args.strict_final_test,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.swa_window and (
        args.evaluation_protocol != "strict" or args.head_variant != "mean_rich_stats_residual"
    ):
        parser.error("--swa-window requires strict mean_rich_stats_residual evaluation")
    if args.correlation_loss_lambda and (
        args.evaluation_protocol != "strict" or args.head_variant != "mean_rich_stats_residual"
    ):
        parser.error(
            "--correlation-loss-lambda requires strict mean_rich_stats_residual evaluation"
        )
    if args.robust_loss != "mse" and (
        args.evaluation_protocol != "strict" or args.head_variant != "mean_rich_stats_residual"
    ):
        parser.error("--robust-loss smooth_l1 requires strict mean_rich_stats_residual evaluation")
    if args.target_scaler != "none" and (
        args.evaluation_protocol != "strict" or args.head_variant != "mean_rich_stats_residual"
    ):
        parser.error("--target-scaler zscore requires strict mean_rich_stats_residual evaluation")

    if args.two_stage_finetune:
        try:
            from two_stage_finetuning import (
                TwoStageFineTuneConfig,
                validate_two_stage_options,
            )

            validate_two_stage_options(
                head_variant=args.head_variant,
                data_mode="manifest" if args.manifest is not None else (
                    "full" if args.full_data else "selective_task"
                ),
                evaluation_protocol=args.evaluation_protocol,
            )
            TwoStageFineTuneConfig(
                warmup_epochs=args.two_stage_warmup_epochs,
                unfreeze_last_blocks=args.two_stage_unfreeze_last_blocks,
                encoder_gradient_scale=args.two_stage_encoder_gradient_scale,
            )
        except ValueError as error:
            parser.error(str(error))

    if args.smoke_head is not None:
        print(json.dumps(run_official_stack_smoke(head_variant=args.smoke_head), indent=2))
        return 0
    required = {
        "--data-root": args.data_root,
        "--output-dir": args.output_dir,
        "--config": args.config,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    source_modes = int(args.manifest is not None) + int(args.full_data) + int(args.selective_task)
    if source_modes != 1:
        parser.error("exactly one of --manifest, --full-data, or --selective-task is required")

    resolved_seeds = validate_seeds(args.seeds)
    for name, value in (
        ("--mean-gradient-scale", args.mean_gradient_scale),
        ("--correction-gradient-scale", args.correction_gradient_scale),
    ):
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 2.0:
            parser.error(f"{name} must be finite and in [0, 2]")
    if not math.isfinite(float(args.correlation_loss_lambda)) or not 0.0 <= float(args.correlation_loss_lambda) <= 0.1:
        parser.error("--correlation-loss-lambda must be finite and in [0, 0.1]")
    try:
        source = _resolve_data_source(
            manifest_path=args.manifest,
            full_data=args.full_data,
            selective_task=args.selective_task,
            data_root=args.data_root,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        parser.error(str(error))
    launch_command = shlex.join([sys.executable, *sys.argv])
    rows = (
        len(load_manifest_timelines(source.manifest_path, source.data_root))
        if source.data_mode == "manifest"
        else None
    )

    try:
        reve = _load_reve_helpers()

        metadata = _head_metadata(
            reve,
            head_variant=args.head_variant,
            layer_index=args.layer_index,
            mean_gradient_scale=args.mean_gradient_scale,
            correction_gradient_scale=args.correction_gradient_scale,
            correlation_loss_lambda=args.correlation_loss_lambda,
            robust_loss=args.robust_loss,
            target_scaler_mode=args.target_scaler,
            two_stage_finetune=args.two_stage_finetune,
            two_stage_warmup_epochs=args.two_stage_warmup_epochs,
            two_stage_unfreeze_last_blocks=args.two_stage_unfreeze_last_blocks,
            two_stage_encoder_gradient_scale=args.two_stage_encoder_gradient_scale,
            data_mode=source.data_mode,
            manifest_path=source.manifest_path,
            manifest_digest=source.manifest_sha256,
            provenance_path=source.manifest_path,
            provenance_digest=source.manifest_sha256,
            timeline_count=rows,
            rows=rows,
            seeds=resolved_seeds,
            launch_command=launch_command,
        )

        reports: list[dict[str, Any]] = []
        summary_path = args.output_dir / "summary.json"
        for seed in resolved_seeds:
            run_dir = args.output_dir / args.head_variant / f"seed{seed}"
            if args.evaluation_protocol == "strict":
                _ensure_fresh_strict_run_dir(run_dir)
            config_path = run_dir / "neuralbench_config.json"
            epoch_metrics_path = run_dir / (
                "epoch_validation_metrics.jsonl"
                if args.evaluation_protocol == "strict"
                else "epoch_test_metrics.jsonl"
            )
            selection_path = run_dir / "selection.json"
            provenance_path = (
                run_dir / "full_data_provenance.json"
                if source.data_mode == "full"
                else run_dir / "selective_task_timeline_provenance.json"
                if source.data_mode == "selective_task"
                else source.manifest_path
            )
            seed_metadata = {
                **metadata,
                "seed": seed,
                "data_seed": 33,
                "provenance_path": (
                    str(provenance_path)
                    if source.data_mode == "manifest" and provenance_path is not None
                    else None
                ),
                "provenance_sha256": source.manifest_sha256,
                "acquisition_provenance_path": None,
                "acquisition_provenance_sha256": None,
                "timeline_count": rows,
                "mean_gradient_scale": args.mean_gradient_scale,
                "correction_gradient_scale": args.correction_gradient_scale,
                "swa_window": args.swa_window,
                "correlation_loss_lambda": args.correlation_loss_lambda,
                "correlation_loss_objective": (
                    "batch_pearson" if args.correlation_loss_lambda else None
                ),
                "robust_loss": args.robust_loss,
                "target_scaler_mode": args.target_scaler,
                "two_stage_finetune": bool(args.two_stage_finetune),
                "two_stage_warmup_epochs": (
                    args.two_stage_warmup_epochs if args.two_stage_finetune else None
                ),
                "two_stage_unfreeze_last_blocks": (
                    args.two_stage_unfreeze_last_blocks if args.two_stage_finetune else None
                ),
                "two_stage_encoder_gradient_scale": (
                    args.two_stage_encoder_gradient_scale if args.two_stage_finetune else None
                ),
                "layer_index": args.layer_index,
            }
            try:
                if source.data_mode == "selective_task":
                    acquisition_path, acquisition_digest = _copy_selective_acquisition_snapshot(
                        source,
                        run_dir,
                    )
                    seed_metadata.update(
                        {
                            "acquisition_provenance_path": str(acquisition_path),
                            "acquisition_provenance_sha256": acquisition_digest,
                        }
                    )
                _write_config(
                    config_path,
                    data_root=source.data_root,
                    output_dir=run_dir,
                    data_mode=source.data_mode,
                    head_variant=args.head_variant,
                    layer_index=args.layer_index,
                    mean_gradient_scale=args.mean_gradient_scale,
                    correction_gradient_scale=args.correction_gradient_scale,
                    swa_window=args.swa_window,
                    correlation_loss_lambda=args.correlation_loss_lambda,
                    robust_loss=args.robust_loss,
                    target_scaler_mode=args.target_scaler,
                    two_stage_finetune=args.two_stage_finetune,
                    two_stage_warmup_epochs=args.two_stage_warmup_epochs,
                    two_stage_unfreeze_last_blocks=args.two_stage_unfreeze_last_blocks,
                    two_stage_encoder_gradient_scale=args.two_stage_encoder_gradient_scale,
                )
                results = run_official_subset(
                    manifest_path=source.manifest_path,
                    data_root=source.data_root,
                    epoch_metrics_path=epoch_metrics_path,
                    selection_path=selection_path,
                    config_path=config_path,
                    head_variant=args.head_variant,
                    layer_index=args.layer_index,
                    mean_gradient_scale=args.mean_gradient_scale,
                    correction_gradient_scale=args.correction_gradient_scale,
                    swa_window=args.swa_window,
                    correlation_loss_lambda=args.correlation_loss_lambda,
                    robust_loss=args.robust_loss,
                    target_scaler_mode=args.target_scaler,
                    seeds=(seed,),
                    evaluation_protocol=args.evaluation_protocol,
                    strict_final_test=args.strict_final_test,
                    two_stage_finetune=args.two_stage_finetune,
                    two_stage_warmup_epochs=args.two_stage_warmup_epochs,
                    two_stage_unfreeze_last_blocks=args.two_stage_unfreeze_last_blocks,
                    two_stage_encoder_gradient_scale=args.two_stage_encoder_gradient_scale,
                    data_mode=source.data_mode,
                    provenance_path=provenance_path,
                    acquisition_provenance_path=(
                        Path(seed_metadata["acquisition_provenance_path"])
                        if seed_metadata.get("acquisition_provenance_path") is not None
                        else None
                    ),
                    acquisition_provenance_sha256=seed_metadata.get(
                        "acquisition_provenance_sha256"
                    ),
                    timeline_count=rows,
                    run_metadata=seed_metadata,
                )
                tuning_metadata = (
                    _merge_last_tuned_result_metadata(results)
                    if args.head_variant == "last_tuned"
                    else {}
                )
                report = {
                    "status": "completed",
                    **seed_metadata,
                    **tuning_metadata,
                    "official_results": results,
                    "epoch_metrics": str(epoch_metrics_path),
                    "selection_path": str(selection_path),
                    "artifacts": collect_run_artifacts(run_dir),
                }
                resolved_head_metadata = _resolved_head_metadata(run_dir)
                if resolved_head_metadata:
                    report["head_metadata"] = resolved_head_metadata
                    if "parameter_count" in resolved_head_metadata:
                        report["head_parameter_count"] = resolved_head_metadata["parameter_count"]
                if source.data_mode in {"full", "selective_task"}:
                    if provenance_path is None or not provenance_path.is_file():
                        raise RuntimeError(
                            f"{source.data_mode} run did not produce timeline provenance"
                        )
                    provenance_payload = _load_json_object(provenance_path)
                    if provenance_payload.get("data_mode") != source.data_mode:
                        raise RuntimeError(
                            f"{source.data_mode} timeline provenance has the wrong data_mode"
                        )
                    if (
                        source.data_mode == "selective_task"
                        and provenance_payload.get("task") != SELECTIVE_TASK
                    ):
                        raise RuntimeError("selective-task timeline provenance has the wrong task")
                    actual_timeline_count = provenance_payload.get("timeline_count")
                    if (
                        isinstance(actual_timeline_count, bool)
                        or not isinstance(actual_timeline_count, int)
                        or actual_timeline_count < 1
                    ):
                        raise RuntimeError("full-data provenance has an invalid timeline_count")
                    report.update(
                        {
                            "provenance_path": str(provenance_path.resolve()),
                            "provenance_sha256": _sha256_file(provenance_path),
                            "timeline_count": actual_timeline_count,
                        }
                    )
                    if source.data_mode == "selective_task":
                        acquisition_path = Path(seed_metadata["acquisition_provenance_path"])
                        acquisition_digest = seed_metadata["acquisition_provenance_sha256"]
                        if (
                            not acquisition_path.is_file()
                            or _sha256_file(acquisition_path) != acquisition_digest
                            or not acquisition_path.with_suffix(".sha256").is_file()
                            or acquisition_path.with_suffix(".sha256").read_text(encoding="ascii").strip()
                            != acquisition_digest
                        ):
                            raise RuntimeError("selective acquisition snapshot is not intact")
                        report.update(
                            {
                                "acquisition_provenance_path": str(acquisition_path.resolve()),
                                "acquisition_provenance_sha256": acquisition_digest,
                            }
                        )
                if args.evaluation_protocol == "strict":
                    report.update(
                        _strict_report_fields(
                            selection_path=selection_path,
                            results=results,
                            strict_final_test=args.strict_final_test,
                        )
                    )
                    swa_metadata = _resolved_swa_metadata(run_dir)
                    if swa_metadata:
                        report["checkpoint_averaging"] = swa_metadata
                else:
                    report.update(
                        {
                            "evaluation_protocol": "legacy",
                            "strict_final_test": False,
                            "test_status": "epoch_diagnostic",
                            "test_evaluations": len(results),
                        }
                    )
                if args.head_variant == "last_tuned":
                    selected_checkpoint_epoch = _selected_validation_checkpoint_epoch(results)
                    if selected_checkpoint_epoch is not None:
                        report["selected_checkpoint_epoch"] = selected_checkpoint_epoch
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
            except Exception as error:
                if source.data_mode in {"full", "selective_task"} and provenance_path is not None:
                    try:
                        provenance_path.unlink(missing_ok=True)
                    except OSError as cleanup_error:
                        error.add_note(
                            f"failed to remove {source.data_mode} provenance after failure: {cleanup_error!r}"
                        )
                write_failure_diagnostics(run_dir, error, launch_command=launch_command, metadata=seed_metadata)
                report_path = run_dir / "report.json"
                try:
                    report_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    error.add_note(f"failed to remove stale report.json after failure: {cleanup_error!r}")
                try:
                    summary_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    error.add_note(f"failed to remove stale summary.json after failure: {cleanup_error!r}")
                raise

            reports.append(report)

        summary = {
            "status": "completed",
            "head_variant": args.head_variant,
            "seeds": list(resolved_seeds),
            "runs": reports,
            "data_mode": source.data_mode,
            "timeline_count": rows,
            "manifest_path": str(source.manifest_path) if source.manifest_path is not None else None,
            "manifest_sha256": source.manifest_sha256,
            "two_stage_finetune": bool(args.two_stage_finetune),
            "two_stage_warmup_epochs": (
                args.two_stage_warmup_epochs if args.two_stage_finetune else None
            ),
            "two_stage_unfreeze_last_blocks": (
                args.two_stage_unfreeze_last_blocks if args.two_stage_finetune else None
            ),
            "two_stage_encoder_gradient_scale": (
                args.two_stage_encoder_gradient_scale if args.two_stage_finetune else None
            ),
            "provenance_path": (
                str(source.manifest_path) if source.data_mode == "manifest" and source.manifest_path is not None else None
            ),
            "provenance_sha256": source.manifest_sha256,
        }
        if args.evaluation_protocol == "strict":
            summary.update(_strict_summary_fields(reports))
        else:
            if source.data_mode in {"full", "selective_task"}:
                timeline_counts = {report.get("timeline_count") for report in reports}
                provenance_paths = {
                    report.get("provenance_path") for report in reports
                }
                provenance_hashes = {
                    report.get("provenance_sha256") for report in reports
                }
                if (
                    len(timeline_counts) != 1
                    or None in timeline_counts
                    or len(provenance_paths) != len(reports)
                    or None in provenance_paths
                    or len(provenance_hashes) != len(reports)
                    or None in provenance_hashes
                ):
                    raise RuntimeError("full-data legacy summary is missing provenance metadata")
                summary.update(
                    {
                        "timeline_count": next(iter(timeline_counts)),
                        "provenance_path": {
                            str(report["seed"]): report["provenance_path"]
                            for report in reports
                        },
                        "provenance_sha256": {
                            str(report["seed"]): report["provenance_sha256"]
                            for report in reports
                        },
                    }
                )
                if source.data_mode == "selective_task":
                    acquisition_paths = {
                        report.get("acquisition_provenance_path") for report in reports
                    }
                    acquisition_hashes = {
                        report.get("acquisition_provenance_sha256") for report in reports
                    }
                    if (
                        len(acquisition_paths) != len(reports)
                        or None in acquisition_paths
                        or len(acquisition_hashes) != len(reports)
                        or None in acquisition_hashes
                    ):
                        raise RuntimeError("selective-task legacy summary is missing acquisition metadata")
                    summary.update(
                        {
                            "acquisition_provenance_path": {
                                str(report["seed"]): report["acquisition_provenance_path"]
                                for report in reports
                            },
                            "acquisition_provenance_sha256": {
                                str(report["seed"]): report["acquisition_provenance_sha256"]
                                for report in reports
                            },
                        }
                    )
            summary.update(
                {
                    "evaluation_protocol": "legacy",
                    "strict_final_test": False,
                    "test_status": "epoch_diagnostic",
                }
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, default=str))
        return 0
    except Exception as error:
        try:
            (args.output_dir / "summary.json").unlink(missing_ok=True)
        except OSError as cleanup_error:
            error.add_note(f"failed to remove stale summary.json after failure: {cleanup_error!r}")
        LOGGER.error("official REVE run failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
