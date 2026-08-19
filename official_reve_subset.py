"""Run the official NeuralBench REVE Age baseline on a fixed manifest.

The public NeuralBench CLI discovers every recording under ``DATA_DIR``.  This
module keeps the official experiment unchanged, but replaces only the HBN
study's timeline iterator with rows from the canonical 500-subject manifest.
It also adds a read-only test pass after every training epoch so the test
Pearson trajectory is visible without affecting validation, checkpointing, or
early stopping.
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
import sys
import traceback
import types
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LOGGER = logging.getLogger(__name__)

try:
    from lightning.pytorch.callbacks import Callback as LightningCallback
except ImportError:  # Keep manifest-only helpers importable without Lightning.
    class LightningCallback:  # type: ignore[no-redef]
        """Fallback base used only when the optional official stack is absent."""



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
            raise ValueError(
                "manifest must contain exactly the canonical Age fields: "
                f"{sorted(required)}"
            )
        for row in reader:
            relative = Path(row["recording_relpath"]).as_posix()
            recording = (data_root / relative).resolve()
            try:
                recording.relative_to(data_root)
            except ValueError as exc:
                raise ValueError(
                    f"manifest recording escapes data root: {relative}"
                ) from exc
            if not recording.is_file():
                raise FileNotFoundError(f"missing recording: {recording}")
            if relative in seen_paths:
                raise ValueError(f"duplicate manifest recording: {relative}")
            seen_paths.add(relative)

            task, run, filename_subject = _parse_timeline_name(recording)
            if filename_subject != row["subject"]:
                raise ValueError(
                    f"manifest subject does not match filename: {relative}"
                )
            if task != "task-RestingState":
                raise ValueError(f"manifest contains a non-resting recording: {relative}")
            rows.append(
                {
                    "release": row["release"],
                    "subject": row["subject"],
                    "task": task,
                    "run": run,
                }
            )

    if not rows:
        raise ValueError(f"manifest is empty: {manifest_path}")
    return tuple(rows)


def manifest_sha256(path: Path) -> str:
    """Return the manifest digest recorded with the run."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


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
                batch = trainer.strategy.batch_to_device(
                    batch,
                    pl_module.device,
                    dataloader_idx=0,
                )
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
            "test_pearsonr": score,
        }
        message = (
            "EPOCH_TEST "
            f"epoch={record['epoch']} "
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


_CONFIGURE_OPTIMIZERS_ABSENT = object()


def _last_tuned_configure_optimizers(brain_module: Any) -> dict[str, Any]:
    """Build the tuned optimizer from one prepared BrainModule instance."""

    try:
        from reve_upstream_heads import build_last_tuned_optimizer_config
    except ImportError:  # Package-style invocation: ``python -m ...``.
        from .reve_upstream_heads import build_last_tuned_optimizer_config

    model = getattr(brain_module, "model", None)
    trainer = getattr(brain_module, "trainer", None)
    return build_last_tuned_optimizer_config(model, trainer=trainer)


def _patch_last_tuned_configure_optimizers(
    brain_module: Any,
    patched_modules: list[dict[str, Any]],
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
    previous = instance_attributes.get(
        "configure_optimizers", _CONFIGURE_OPTIMIZERS_ABSENT
    )
    record = {
        "module": brain_module,
        "previous": previous,
    }
    patched_modules.append(record)
    try:
        brain_module.configure_optimizers = types.MethodType(
            _last_tuned_configure_optimizers,
            brain_module,
        )
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
        error = RuntimeError(
            "failed to restore one or more last_tuned configure_optimizers patches"
        )
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


def _patch_official_components(
    manifest_path: Path,
    data_root: Path,
    epoch_metrics_path: Path,
    *,
    head_variant: str = "mean_linear",
    head_dropout: float = 0.0,
    seeds: Sequence[int] = (33,),
    run_metadata: Mapping[str, Any] | None = None,
    final_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Install the fixed-manifest and optional upstream-head patches.

    The patch is deliberately scoped to one ``run_benchmark`` call.  The
    official package remains untouched after :func:`_restore_official_components`
    runs, which is important when several variants are evaluated in one Python
    process.
    """

    try:
        from reve_upstream_heads import (
            PROTOCOL_CONTRACT,
            last_tuned_optimizer_metadata,
            make_upstream_reve_wrapper,
            _resolve_last_tuned_model,
            validate_head_variant,
            validate_last_tuned_protocol,
            validate_official_protocol,
        )
    except ImportError:  # Package-style invocation: ``python -m ...``.
        from .reve_upstream_heads import (
            PROTOCOL_CONTRACT,
            last_tuned_optimizer_metadata,
            make_upstream_reve_wrapper,
            _resolve_last_tuned_model,
            validate_head_variant,
            validate_last_tuned_protocol,
            validate_official_protocol,
        )

    validate_head_variant(head_variant)
    if head_dropout != 0.0:
        raise ValueError(
            "the upstream REVE comparison fixes head dropout at 0.0; "
            f"got {head_dropout}"
        )
    resolved_seeds = validate_seeds(seeds)

    from neuralbench.data import Data
    from neuralbench.main import Experiment
    from neuralfetch.studies import shirazi2024hbn

    timelines = load_manifest_timelines(manifest_path, data_root)

    original_iter_timelines = shirazi2024hbn.Shirazi2024Hbn.iter_timelines
    original_info = shirazi2024hbn.Shirazi2024Hbn._info
    original_prepare = Data.prepare
    original_test = Experiment._test
    original_setup_run = Experiment.setup_run
    original_prepare_pl_module = Experiment.prepare_pl_module
    original_setup_trainer = Experiment.setup_trainer

    def iter_manifest_timelines(_study: Any) -> Iterable[dict[str, Any]]:
        return iter(timelines)

    captured_loaders: dict[int, dict[str, Any]] = {}
    patched_brain_modules: list[dict[str, Any]] = []
    tuning_metadata_by_experiment: dict[int, dict[str, Any]] = {}

    def prepare_and_capture(data: Any) -> dict[str, Any]:
        loaders = original_prepare(data)
        captured_loaders[id(data)] = loaders
        return loaders

    def test_and_capture(
        self: Any,
        loaders: dict[str, Any],
        best_model_path: str | None,
    ) -> dict[str, Any]:
        result = original_test(self, loaders, best_model_path)
        if final_results is not None:
            captured = dict(result)
            if head_variant == "last_tuned":
                captured_metadata = captured.get("tuning_metadata")
                merged_metadata = (
                    dict(captured_metadata)
                    if isinstance(captured_metadata, Mapping)
                    else {}
                )
                merged_metadata.update(tuning_metadata_by_experiment.get(id(self), {}))
                captured["tuning_metadata"] = merged_metadata
            final_results.append(captured)
        return result

    def setup_with_epoch_test(self: Any, is_test: bool = False) -> Any:
        trainer = original_setup_trainer(self, is_test=is_test)
        if not is_test:
            loaders = captured_loaders.get(id(self.data))
            if loaders is None or "test" not in loaders:
                raise RuntimeError("official test loader was not captured")
            trainer.callbacks.append(
                EpochTestPearson(
                    loaders["test"],
                    epoch_metrics_path,
                    seed=getattr(self, "seed", None),
                )
            )
        return trainer

    def setup_with_metadata(self: Any) -> Any:
        # The standard REVE YAML wrapper is mean-pooling plus a linear probe.
        # For upstream variants, replace only that downstream config; the
        # already-built official NtReve encoder still comes from NeuralBench.
        if head_variant != "mean_linear":
            _set_frozen_experiment_field(
                self,
                "downstream_model_wrapper",
                make_upstream_reve_wrapper(
                    variant=head_variant,
                    dropout=head_dropout,
                ),
            )
        if head_variant == "last_tuned":
            # NeuralBench expresses the actual checkpoint criterion through
            # ``trainer_config``.  Record its resolved tuning counterpart on
            # this run instance so the separate tuning validator can reject a
            # relabeling of the diagnostic test callback as a selector.
            _set_frozen_experiment_field(
                self,
                "checkpoint_selection",
                {
                    "monitor": "val/pearsonr",
                    "mode": "max",
                    "test_pearsonr_role": "diagnostic_only",
                },
            )
        _set_frozen_experiment_field(self, "save_test_predictions", True)
        # Keep the selected checkpoint and raw prediction cache available for
        # post-run hashing/export. This does not affect training or selection.
        _set_frozen_experiment_field(self, "delete_checkpoints_on_exit", False)
        result = original_setup_run(self)
        uid_folder = self.infra.uid_folder()
        if uid_folder is not None:
            payload = dict(run_metadata or {})
            payload.update(
                {
                    "head_variant": head_variant,
                    "head_dropout": float(head_dropout),
                    "seed": int(self.seed),
                    "data_seed": _get_attr_or_key(self.data, "seed"),
                    "protocol": PROTOCOL_CONTRACT,
                }
            )
            (uid_folder / "run_metadata.json").write_text(
                json.dumps(payload, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
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
        path.write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def prepare_with_protocol(
        self: Any,
        train_loader: Any,
        val_loader: Any = None,
    ) -> Any:
        result = original_prepare_pl_module(self, train_loader, val_loader)
        if head_variant == "last_tuned":
            brain_module = getattr(self, "_brain_module", None)
            model = getattr(brain_module, "model", None)
            tuning_model = _resolve_last_tuned_model(model)
            query_metadata = getattr(
                getattr(tuning_model, "head", None), "tuning_metadata", None
            )
            if not isinstance(query_metadata, Mapping):
                raise RuntimeError("last_tuned prepared model did not expose tuning metadata")
            optimizer_metadata = last_tuned_optimizer_metadata(tuning_model)
            _patch_last_tuned_configure_optimizers(
                brain_module,
                patched_brain_modules,
            )
            validate_last_tuned_protocol(
                head_variant,
                experiment=self,
                optimizer_config=optimizer_metadata,
            )
            tuning_metadata = _last_tuned_report_metadata(
                query_metadata=query_metadata,
                optimizer_config=optimizer_metadata,
            )
            tuning_metadata_by_experiment[id(self)] = tuning_metadata
            persist_tuning_metadata(self, tuning_metadata)
        else:
            loaders = captured_loaders.get(id(self.data))
            validate_official_protocol(
                self,
                loaders=loaders,
                n_total_params=self._n_total_params,
                n_trainable_params=self._n_trainable_params,
            )
        return result

    # NeuralBench's CLI and experiment_config modules each keep a local alias
    # to the YAML loader. Patch both so a task-specific grid cannot silently
    # reintroduce the default (33, 34, 35) seed expansion.
    import neuralbench.cli as cli
    import neuralbench.experiment_config as experiment_config

    original_cli_load_yaml_config = cli.load_yaml_config
    original_experiment_load_yaml_config = experiment_config.load_yaml_config

    originals = {
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
        shirazi2024hbn.Shirazi2024Hbn.iter_timelines = iter_manifest_timelines
        if original_info is not None:
            shirazi2024hbn.Shirazi2024Hbn._info = original_info.model_copy(
                update={"num_timelines": len(timelines)}
            )
        Data.prepare = prepare_and_capture
        Experiment._test = test_and_capture
        Experiment.setup_trainer = setup_with_epoch_test
        Experiment.setup_run = setup_with_metadata
        Experiment.prepare_pl_module = prepare_with_protocol
        cli.load_yaml_config = load_seed_grid
        experiment_config.load_yaml_config = load_seed_grid_for_experiment_config
    except BaseException as active_error:
        try:
            _restore_official_components(originals)
        except BaseException as cleanup_error:
            active_error.add_note(f"patch setup cleanup failure: {cleanup_error!r}")
        raise
    return originals


def _restore_official_components(originals: Mapping[str, Any]) -> None:
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

    attempt(
        "Shirazi2024Hbn.iter_timelines",
        lambda: setattr(
            shirazi2024hbn.Shirazi2024Hbn,
            "iter_timelines",
            originals["iter_timelines"],
        ),
    )
    attempt(
        "Shirazi2024Hbn._info",
        lambda: setattr(
            shirazi2024hbn.Shirazi2024Hbn,
            "_info",
            originals["info"],
        ),
    )
    attempt("Data.prepare", lambda: setattr(Data, "prepare", originals["prepare"]))
    attempt(
        "Experiment.setup_run",
        lambda: setattr(Experiment, "setup_run", originals["setup_run"]),
    )
    attempt(
        "Experiment._test",
        lambda: setattr(Experiment, "_test", originals["test"]),
    )
    attempt(
        "Experiment.prepare_pl_module",
        lambda: setattr(
            Experiment,
            "prepare_pl_module",
            originals["prepare_pl_module"],
        ),
    )
    attempt(
        "Experiment.setup_trainer",
        lambda: setattr(
            Experiment,
            "setup_trainer",
            originals["setup_trainer"],
        ),
    )
    attempt(
        "neuralbench.cli.load_yaml_config",
        lambda: setattr(cli, "load_yaml_config", original_cli_loader),
    )
    attempt(
        "neuralbench.experiment_config.load_yaml_config",
        lambda: setattr(
            experiment_config,
            "load_yaml_config",
            original_experiment_loader,
        ),
    )
    attempt(
        "last_tuned.configure_optimizers",
        lambda: _restore_last_tuned_configure_optimizers(
            originals.get("patched_brain_modules", [])
        ),
    )

    if restoration_errors:
        error = RuntimeError("official component restoration failed")
        for label, restoration_error in restoration_errors:
            error.add_note(f"{label}: {restoration_error!r}")
        raise error


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
        checkpoint = next(
            (path for path in checkpoints if path.name == "best.ckpt"),
            checkpoints[0] if checkpoints else None,
        )
        prediction_dir = run_dir / "test_predictions"
        prediction_files = []
        if prediction_dir.is_dir():
            for path in sorted(prediction_dir.rglob("*")):
                if path.is_file():
                    prediction_files.append(
                        {
                            "path": str(path),
                            "size_bytes": path.stat().st_size,
                            "sha256": _sha256_file(path),
                        }
                    )
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

    try:
        from reve_upstream_heads import (
            UpstreamReveHeadModel,
            validate_last_tuned_protocol,
            validate_upstream_head_variant,
        )
    except ImportError:
        from .reve_upstream_heads import (
            UpstreamReveHeadModel,
            validate_last_tuned_protocol,
            validate_upstream_head_variant,
        )

    if head_variant == "last_tuned":
        validate_last_tuned_protocol(head_variant)
    else:
        validate_upstream_head_variant(head_variant)
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
    if head_variant != "last_tuned":
        # Keep the official smoke variants on their exact existing RNG and
        # construction path.  Only the tuning branch needs encoder tokens
        # before it can construct its explicit query.
        adapter = UpstreamReveHeadModel(
            encoder,
            variant=head_variant,
            n_outputs=1,
            dropout=0.0,
        ).to(device)
    eeg = torch.randn(2, n_chans, n_times, device=device)
    positions = torch.randn(2, n_chans, 3, device=device)

    with torch.inference_mode():
        raw_layers = model(eeg, pos=positions, return_output=True)
        final = encoder(eeg, pos=positions)
    if head_variant == "last_tuned":
        if not isinstance(final, torch.Tensor) or final.ndim != 3:
            raise RuntimeError("last_tuned smoke encoder did not return final tokens")
        with torch.inference_mode(False):
            query_token = final[:1].mean(dim=1, keepdim=True).detach().clone()
        if not torch.isfinite(query_token).all():
            raise RuntimeError("last_tuned smoke mean-token query is not finite")
        adapter = UpstreamReveHeadModel(
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
        raise RuntimeError(
            "official REVE layer contract changed: expected positional input "
            f"plus {depth} layers, got {len(raw_layers)}"
        )
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
        "query_initialization": adapter.head.query_initialization,
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
    return output


def run_official_subset(
    *,
    manifest_path: Path,
    data_root: Path,
    epoch_metrics_path: Path,
    config_path: Path,
    head_variant: str = "mean_linear",
    head_dropout: float = 0.0,
    seeds: Sequence[int] = (33,),
    run_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run official REVE on the manifest and collect cached results."""

    os.environ["NEURALBENCH_CONFIG"] = str(config_path)
    final_results: list[dict[str, Any]] = []
    originals: Mapping[str, Any] | None = None
    benchmark_aggregator: Any = None
    original_aggregator_prepare: Any = None
    try:
        originals = _patch_official_components(
            manifest_path,
            data_root,
            epoch_metrics_path,
            head_variant=head_variant,
            head_dropout=head_dropout,
            seeds=seeds,
            run_metadata=run_metadata,
            final_results=final_results,
        )
        from neuralbench.main import BenchmarkAggregator

        benchmark_aggregator = BenchmarkAggregator
        original_aggregator_prepare = BenchmarkAggregator.prepare
        BenchmarkAggregator.prepare = _run_experiments_synchronously
        from neuralbench import run_benchmark

        run_benchmark(
            device="eeg",
            task="age",
            model="reve",
            force=True,
        )
        # The public runner returns no result for a non-debug local run. The
        # official Experiment._test result is captured above instead. Avoid a
        # second ``plot_cached`` call: it reconstructs the canonical mean head
        # and collides with this run's custom upstream head UID.
        return final_results
    finally:
        active_error = sys.exc_info()[1]
        cleanup_errors: list[BaseException] = []
        if benchmark_aggregator is not None and original_aggregator_prepare is not None:
            try:
                benchmark_aggregator.prepare = original_aggregator_prepare
            except BaseException as error:
                cleanup_errors.append(error)
        if originals is not None:
            try:
                _restore_official_components(originals)
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--smoke-head",
        choices=("last_avg", "last", "all", "last_tuned"),
        help="run a data-free smoke test using the installed official stack",
    )
    parser.add_argument(
        "--head-variant",
        choices=("mean_linear", "last_avg", "last", "all", "last_tuned"),
        default="mean_linear",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[33])
    args = parser.parse_args(argv)

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
        try:
            from reve_upstream_heads import (
                LAST_TUNED_BASE_LR,
                LAST_TUNED_HEAD_ARCHITECTURE,
                LAST_TUNED_HEAD_SOURCE,
                LAST_TUNED_INITIAL_ALPHA,
                LAST_TUNED_PROTOCOL_CLASS,
                LAST_TUNED_QUERY_LR,
                LAST_TUNED_SCHEDULER_DIV_FACTOR,
                LAST_TUNED_SCHEDULER_FINAL_DIV_FACTOR,
                LAST_TUNED_SCHEDULER_MAX_LR,
                LAST_TUNED_SCHEDULER_PCT_START,
                LAST_TUNED_WEIGHT_DECAY,
                PROTOCOL_CONTRACT,
                runtime_metadata,
                source_lock_metadata,
            )
        except ImportError:
            from .reve_upstream_heads import (
                LAST_TUNED_BASE_LR,
                LAST_TUNED_HEAD_ARCHITECTURE,
                LAST_TUNED_HEAD_SOURCE,
                LAST_TUNED_INITIAL_ALPHA,
                LAST_TUNED_PROTOCOL_CLASS,
                LAST_TUNED_QUERY_LR,
                LAST_TUNED_SCHEDULER_DIV_FACTOR,
                LAST_TUNED_SCHEDULER_FINAL_DIV_FACTOR,
                LAST_TUNED_SCHEDULER_MAX_LR,
                LAST_TUNED_SCHEDULER_PCT_START,
                LAST_TUNED_WEIGHT_DECAY,
                PROTOCOL_CONTRACT,
                runtime_metadata,
                source_lock_metadata,
            )

        if args.head_variant == "mean_linear":
            query_initialization = "neuralbench_default"
        elif args.head_variant == "last_avg":
            query_initialization = "upstream_random_unused"
        elif args.head_variant == "last_tuned":
            query_initialization = "train_dummy_final_token_mean"
        else:
            query_initialization = "upstream_random"

        metadata = {
            "head_variant": args.head_variant,
            "head_source": (
                "neuralbench_default"
                if args.head_variant == "mean_linear"
                else "upstream_reve"
            ),
            "head_dropout": 0.0,
            "head_query_initialization": query_initialization,
            "head_linear_initialization": (
                "neuralbench_default"
                if args.head_variant == "mean_linear"
                else {
                    "distribution": "truncated_normal",
                    "std": 512**-0.5,
                    "cutoff": 3.0,
                    "bias": 0.0,
                }
            ),
            "manifest": str(args.manifest),
            "manifest_sha256": digest,
            "rows": rows,
            "device": "eeg",
            "seeds": list(resolved_seeds),
            "launch_command": launch_command,
            "protocol": PROTOCOL_CONTRACT,
            "runtime": runtime_metadata(),
        }
        if args.head_variant == "last_tuned":
            metadata.update(
                {
                    "head_source": LAST_TUNED_HEAD_SOURCE,
                    "head_architecture": LAST_TUNED_HEAD_ARCHITECTURE,
                    "protocol_class": LAST_TUNED_PROTOCOL_CLASS,
                    "residual_initial_alpha": LAST_TUNED_INITIAL_ALPHA,
                    "query_initialization": query_initialization,
                    "base_learning_rate": LAST_TUNED_BASE_LR,
                    "query_learning_rate": LAST_TUNED_QUERY_LR,
                    "optimizer": "AdamW",
                    "weight_decay": LAST_TUNED_WEIGHT_DECAY,
                    "scheduler": "OneCycleLR",
                    "scheduler_max_lr": list(LAST_TUNED_SCHEDULER_MAX_LR),
                    "scheduler_pct_start": LAST_TUNED_SCHEDULER_PCT_START,
                    "scheduler_anneal_strategy": "cos",
                    "scheduler_div_factor": LAST_TUNED_SCHEDULER_DIV_FACTOR,
                    "scheduler_final_div_factor": LAST_TUNED_SCHEDULER_FINAL_DIV_FACTOR,
                    "scheduler_interval": "step",
                    "scheduler_frequency": 1,
                    "monitor": "val/pearsonr",
                    "checkpoint_selection_monitor": "val/pearsonr",
                    "test_pearsonr_role": "diagnostic_only",
                }
            )
        if args.head_variant != "mean_linear":
            metadata["head_source_lock"] = source_lock_metadata()

        reports: list[dict[str, Any]] = []
        for seed in resolved_seeds:
            run_dir = args.output_dir / args.head_variant / f"seed{seed}"
            config_path = run_dir / "neuralbench_config.json"
            epoch_metrics_path = run_dir / "epoch_test_metrics.jsonl"
            seed_metadata = {**metadata, "seed": seed, "data_seed": 33}
            try:
                _write_config(
                    config_path,
                    data_root=args.data_root,
                    output_dir=run_dir,
                )
                results = run_official_subset(
                    manifest_path=args.manifest,
                    data_root=args.data_root,
                    epoch_metrics_path=epoch_metrics_path,
                    config_path=config_path,
                    head_variant=args.head_variant,
                    seeds=(seed,),
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
                    "artifacts": collect_run_artifacts(run_dir),
                }
                if args.head_variant == "last_tuned":
                    selected_checkpoint_epoch = _selected_validation_checkpoint_epoch(
                        results
                    )
                    if selected_checkpoint_epoch is not None:
                        report["selected_checkpoint_epoch"] = selected_checkpoint_epoch
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "report.json").write_text(
                    json.dumps(report, indent=2, default=str) + "\n",
                    encoding="utf-8",
                )
            except Exception as error:
                failure_path = write_failure_diagnostics(
                    run_dir,
                    error,
                    launch_command=launch_command,
                    metadata=seed_metadata,
                )
                report_path = run_dir / "report.json"
                try:
                    report_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    error.add_note(
                        f"failed to remove stale report.json after failure: {cleanup_error!r}"
                    )
                raise

            reports.append(report)

        summary = {
            "status": "completed",
            "head_variant": args.head_variant,
            "seeds": list(resolved_seeds),
            "runs": reports,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, default=str))
        return 0
    except Exception as error:
        LOGGER.error("official REVE run failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
