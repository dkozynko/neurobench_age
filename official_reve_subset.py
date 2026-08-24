"""Run the official NeuralBench REVE Age baseline on a fixed manifest.

The public NeuralBench CLI discovers every recording under ``DATA_DIR``.  This
module keeps the official experiment unchanged, but replaces only the HBN
study's timeline iterator with rows from the canonical 500-subject manifest.
Strict evaluation records validation Pearson after each training epoch and
withholds the test set until an explicit one-time finalist gate. An explicit
legacy mode retains the historical read-only test pass for parity diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import shlex
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

LOGGER = logging.getLogger(__name__)

try:
    from lightning.pytorch.callbacks import Callback as LightningCallback
except ImportError:  # Keep manifest-only helpers importable without Lightning.
    class LightningCallback:  # type: ignore[no-redef]
        """Fallback base used only when the optional official stack is absent."""



# ---------------------------------------------------------------------------
# Manifest and diagnostic metric helpers
# ---------------------------------------------------------------------------


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

    def __init__(self, output_path: Path, seed: int | None = None):
        self.output_path = output_path
        self.seed = seed
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

    required_hash_fields = (
        "checkpoint_sha256",
        "official_config_sha256",
        "manifest_sha256",
        "validation_history_sha256",
    )
    for field in required_hash_fields:
        value = selection.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"strict selection has invalid {field}")
    for path_field, hash_field in (
        ("checkpoint_path", "checkpoint_sha256"),
        ("official_config_path", "official_config_sha256"),
        ("manifest_path", "manifest_sha256"),
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
        "strict_final_test": gate,
        "test_status": "completed" if gate else "withheld",
        "completed_seed_count": len(reports),
        "selected_epoch_by_seed": selected_epochs,
        "selected_val_pearson_by_seed": selected_values,
        "mean_selected_val_pearson": sum(selected_values.values()) / len(selected_values),
    }
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
    elif head_variant in {"mean_linear_copy", "mean_linear_detached", "mean_linear_warmup", "mean_linear_gradient_scaled", "mean_linear_probe_scaled", "mean_anchor", "mean_residual", "mean_vector_anchor", "mean_mlp_residual", "mean_stats_residual", "mean_stats_residual_detached", "mean_stats_residual_gradient_scaled", "mean_stats_probe_scaled", "mean_stats_attention_residual", "mean_attention_gated", "global_stats_residual", "mean_rich_stats_residual", "grouped_rich_stats_shrinkage", "grouped_stats_shared_gate"}:
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
    elif head_variant == "mean_rich_stats_residual":
        output.update(
            {
                "prediction_finite": bool(torch.isfinite(prediction).all().item()),
            }
        )
    elif head_variant in {"grouped_rich_stats_shrinkage", "grouped_stats_shared_gate"}:
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
    manifest_path: Path,
    data_root: Path,
    epoch_metrics_path: Path,
    selection_path: Path,
    config_path: Path,
    head_variant: str = "mean_linear",
    head_dropout: float = 0.0,
    seeds: Sequence[int] = (33,),
    evaluation_protocol: str = "strict",
    strict_final_test: bool = False,
    run_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return _runtime.run_official_subset(
        manifest_path=manifest_path,
        data_root=data_root,
        epoch_metrics_path=epoch_metrics_path,
        selection_path=selection_path,
        config_path=config_path,
        head_variant=head_variant,
        head_dropout=head_dropout,
        seeds=seeds,
        evaluation_protocol=evaluation_protocol,
        strict_final_test=strict_final_test,
        run_metadata=run_metadata,
        hooks=_hooks(),
    )


def _run_experiments_synchronously(aggregator: Any) -> None:
    return _runtime._run_experiments_synchronously(aggregator)


def _write_config(path: Path, *, data_root: Path, output_dir: Path) -> None:
    config = {
        "USER": "root",
        "ENTITY_NAME": "root",
        "PROJECT_NAME": "neurobench_reve_age_official",
        "CACHE_DIR": str(data_root / "neuralbench_official_cache_500"),
        "SAVE_DIR": str(output_dir),
        "DATA_DIR": str(data_root),
        "WANDB_HOST": "",
        "SLURM_PARTITION": "",
        "SLURM_CONSTRAINT": "",
        "N_CPUS": 2,
        "CLUSTER": None,
    }
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--smoke-head",
        choices=("mean_linear_copy", "mean_linear_detached", "mean_linear_warmup", "mean_linear_gradient_scaled", "mean_linear_probe_scaled", "mean_anchor", "mean_residual", "mean_vector_anchor", "mean_mlp_residual", "mean_stats_residual", "mean_stats_residual_detached", "mean_stats_residual_gradient_scaled", "mean_stats_probe_scaled", "mean_stats_attention_residual", "mean_attention_gated", "global_stats_residual", "mean_rich_stats_residual", "grouped_rich_stats_shrinkage", "grouped_stats_shared_gate", "last_avg", "last", "all", "last_tuned"),
        help="run a data-free smoke test using the installed official stack",
    )
    parser.add_argument(
        "--head-variant",
        choices=("mean_linear", "mean_linear_copy", "mean_linear_detached", "mean_linear_warmup", "mean_linear_gradient_scaled", "mean_linear_probe_scaled", "mean_anchor", "mean_residual", "mean_vector_anchor", "mean_mlp_residual", "mean_stats_residual", "mean_stats_residual_detached", "mean_stats_residual_gradient_scaled", "mean_stats_probe_scaled", "mean_stats_attention_residual", "mean_attention_gated", "global_stats_residual", "mean_rich_stats_residual", "grouped_rich_stats_shrinkage", "grouped_stats_shared_gate", "last_avg", "last", "all", "last_tuned"),
        default="mean_linear",
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[33])
    args = parser.parse_args(argv)

    try:
        args.evaluation_protocol, args.strict_final_test = validate_evaluation_options(
            args.evaluation_protocol,
            strict_final_test=args.strict_final_test,
        )
    except ValueError as error:
        parser.error(str(error))

    if args.smoke_head is not None:
        print(json.dumps(run_official_stack_smoke(head_variant=args.smoke_head), indent=2))
        return 0
    required = {
        "--manifest": args.manifest,
        "--data-root": args.data_root,
        "--output-dir": args.output_dir,
        "--config": args.config,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))

    resolved_seeds = validate_seeds(args.seeds)
    digest = manifest_sha256(args.manifest)
    launch_command = shlex.join([sys.executable, *sys.argv])
    rows = len(load_manifest_timelines(args.manifest, args.data_root))

    try:
        reve = _load_reve_helpers()

        metadata = _head_metadata(
            reve,
            head_variant=args.head_variant,
            manifest_path=args.manifest,
            manifest_digest=digest,
            rows=rows,
            seeds=resolved_seeds,
            launch_command=launch_command,
        )

        reports: list[dict[str, Any]] = []
        summary_path = args.output_dir / "summary.json"
        for seed in resolved_seeds:
            run_dir = args.output_dir / args.head_variant / f"seed{seed}"
            config_path = run_dir / "neuralbench_config.json"
            epoch_metrics_path = run_dir / (
                "epoch_validation_metrics.jsonl"
                if args.evaluation_protocol == "strict"
                else "epoch_test_metrics.jsonl"
            )
            selection_path = run_dir / "selection.json"
            seed_metadata = {**metadata, "seed": seed, "data_seed": 33}
            try:
                _write_config(config_path, data_root=args.data_root, output_dir=run_dir)
                results = run_official_subset(
                    manifest_path=args.manifest,
                    data_root=args.data_root,
                    epoch_metrics_path=epoch_metrics_path,
                    selection_path=selection_path,
                    config_path=config_path,
                    head_variant=args.head_variant,
                    seeds=(seed,),
                    evaluation_protocol=args.evaluation_protocol,
                    strict_final_test=args.strict_final_test,
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
                if args.evaluation_protocol == "strict":
                    report.update(
                        _strict_report_fields(
                            selection_path=selection_path,
                            results=results,
                            strict_final_test=args.strict_final_test,
                        )
                    )
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
        }
        if args.evaluation_protocol == "strict":
            summary.update(_strict_summary_fields(reports))
        else:
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
