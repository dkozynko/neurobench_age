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
    manifest_path: Path,
    manifest_digest: str,
    rows: int,
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
        "last_avg": "upstream_random_unused",
        "last_tuned": "train_dummy_final_token_mean",
        "last": "upstream_random",
        "all": "upstream_random",
    }[head_variant]
    is_default_head = head_variant == "mean_linear"
    is_local_head = head_variant in {"mean_linear_copy", "mean_linear_detached", "mean_linear_warmup", "mean_linear_gradient_scaled", "mean_linear_probe_scaled", "mean_anchor", "mean_residual", "mean_vector_anchor", "mean_mlp_residual", "mean_stats_residual", "mean_stats_residual_detached", "mean_stats_residual_gradient_scaled", "mean_stats_probe_scaled", "mean_stats_attention_residual", "mean_attention_gated"}
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
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_digest,
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


# ---------------------------------------------------------------------------
# Official run patch lifecycle
# ---------------------------------------------------------------------------


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
    hooks: Any,
) -> dict[str, Any]:
    """Install the fixed-manifest and optional upstream-head patches.

    The patch is deliberately scoped to one ``run_benchmark`` call.  The
    official package remains untouched after :func:`_restore_official_components`
    runs, which is important when several variants are evaluated in one Python
    process.
    """

    reve = hooks._load_reve_helpers()

    reve.validate_head_variant(head_variant)
    if head_dropout != 0.0:
        raise ValueError(f"the upstream REVE comparison fixes head dropout at 0.0; got {head_dropout}")
    resolved_seeds = hooks.validate_seeds(seeds)

    from neuralbench.data import Data
    from neuralbench.main import Experiment
    from neuralfetch.studies import shirazi2024hbn

    timelines = hooks.load_manifest_timelines(manifest_path, data_root)

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
            final_results.append(
                hooks._capture_test_result(
                    result,
                    head_variant=head_variant,
                    experiment_id=id(self),
                    tuning_metadata_by_experiment=tuning_metadata_by_experiment,
                )
            )
        return result

    def setup_with_epoch_test(self: Any, is_test: bool = False) -> Any:
        trainer = original_setup_trainer(self, is_test=is_test)
        if not is_test:
            loaders = captured_loaders.get(id(self.data))
            if loaders is None or "test" not in loaders:
                raise RuntimeError("official test loader was not captured")
            trainer.callbacks.append(hooks.EpochTestPearson(loaders["test"], epoch_metrics_path, seed=getattr(self, "seed", None)))
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
                }
            )
            (uid_folder / "run_metadata.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
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
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    def prepare_with_protocol(
        self: Any,
        train_loader: Any,
        val_loader: Any = None,
    ) -> Any:
        result = original_prepare_pl_module(self, train_loader, val_loader)
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
            shirazi2024hbn.Shirazi2024Hbn._info = original_info.model_copy(update={"num_timelines": len(timelines)})
        Data.prepare = prepare_and_capture
        Experiment._test = test_and_capture
        Experiment.setup_trainer = setup_with_epoch_test
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
    manifest_path: Path,
    data_root: Path,
    epoch_metrics_path: Path,
    config_path: Path,
    head_variant: str = "mean_linear",
    head_dropout: float = 0.0,
    seeds: Sequence[int] = (33,),
    run_metadata: Mapping[str, Any] | None = None,
    hooks: Any,
) -> list[dict[str, Any]]:
    """Run official REVE on the manifest and collect cached results."""

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

        run_benchmark(device="eeg", task="age", model="reve", force=True)
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
