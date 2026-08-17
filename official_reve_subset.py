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
import os
import shlex
import sys
import traceback
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
            make_upstream_reve_wrapper,
            validate_head_variant,
            validate_official_protocol,
        )
    except ImportError:  # Package-style invocation: ``python -m ...``.
        from .reve_upstream_heads import (
            PROTOCOL_CONTRACT,
            make_upstream_reve_wrapper,
            validate_head_variant,
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

    shirazi2024hbn.Shirazi2024Hbn.iter_timelines = iter_manifest_timelines
    if original_info is not None:
        shirazi2024hbn.Shirazi2024Hbn._info = original_info.model_copy(
            update={"num_timelines": len(timelines)}
        )

    captured_loaders: dict[int, dict[str, Any]] = {}

    def prepare_and_capture(data: Any) -> dict[str, Any]:
        loaders = original_prepare(data)
        captured_loaders[id(data)] = loaders
        return loaders

    Data.prepare = prepare_and_capture

    def test_and_capture(
        self: Any,
        loaders: dict[str, Any],
        best_model_path: str | None,
    ) -> dict[str, Any]:
        result = original_test(self, loaders, best_model_path)
        if final_results is not None:
            final_results.append(dict(result))
        return result

    Experiment._test = test_and_capture

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

    Experiment.setup_trainer = setup_with_epoch_test

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

    Experiment.setup_run = setup_with_metadata

    def prepare_with_protocol(
        self: Any,
        train_loader: Any,
        val_loader: Any = None,
    ) -> Any:
        result = original_prepare_pl_module(self, train_loader, val_loader)
        loaders = captured_loaders.get(id(self.data))
        validate_official_protocol(
            self,
            loaders=loaders,
            n_total_params=self._n_total_params,
            n_trainable_params=self._n_trainable_params,
        )
        return result

    Experiment.prepare_pl_module = prepare_with_protocol

    # NeuralBench's CLI and experiment_config modules each keep a local alias
    # to the YAML loader. Patch both so a task-specific grid cannot silently
    # reintroduce the default (33, 34, 35) seed expansion.
    import neuralbench.cli as cli
    import neuralbench.experiment_config as experiment_config

    original_cli_load_yaml_config = cli.load_yaml_config
    original_experiment_load_yaml_config = experiment_config.load_yaml_config

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

    cli.load_yaml_config = load_seed_grid
    experiment_config.load_yaml_config = load_seed_grid_for_experiment_config
    return {
        "iter_timelines": original_iter_timelines,
        "info": original_info,
        "prepare": original_prepare,
        "test": original_test,
        "setup_run": original_setup_run,
        "prepare_pl_module": original_prepare_pl_module,
        "setup_trainer": original_setup_trainer,
        "cli_loader": (cli, original_cli_load_yaml_config),
        "experiment_loader": (
            experiment_config,
            original_experiment_load_yaml_config,
        ),
    }


def _restore_official_components(originals: Mapping[str, Any]) -> None:
    from neuralbench.data import Data
    from neuralbench.main import Experiment
    from neuralfetch.studies import shirazi2024hbn

    cli, original_cli_loader = originals["cli_loader"]
    experiment_config, original_experiment_loader = originals["experiment_loader"]
    shirazi2024hbn.Shirazi2024Hbn.iter_timelines = originals["iter_timelines"]
    shirazi2024hbn.Shirazi2024Hbn._info = originals["info"]
    Data.prepare = originals["prepare"]
    Experiment.setup_run = originals["setup_run"]
    Experiment._test = originals["test"]
    Experiment.prepare_pl_module = originals["prepare_pl_module"]
    Experiment.setup_trainer = originals["setup_trainer"]
    cli.load_yaml_config = original_cli_loader
    experiment_config.load_yaml_config = original_experiment_loader


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
        from reve_upstream_heads import UpstreamReveHeadModel, validate_upstream_head_variant
    except ImportError:
        from .reve_upstream_heads import (
            UpstreamReveHeadModel,
            validate_upstream_head_variant,
        )

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

    return {
        "head_variant": head_variant,
        "device": device,
        "token_shapes": [list(layer.shape) for layer in raw_layers],
        "final_shape": list(final.shape),
        "prediction_shape": list(prediction.shape),
        "embed_dim": embed_dim,
        "layer_count_including_initial": len(raw_layers),
        "query_initialization": adapter.head.query_initialization,
    }


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

    original_aggregator_prepare = BenchmarkAggregator.prepare
    BenchmarkAggregator.prepare = _run_experiments_synchronously
    try:
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
        BenchmarkAggregator.prepare = original_aggregator_prepare
        _restore_official_components(originals)


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
        choices=("last_avg", "last", "all"),
        help="run a data-free smoke test using the installed official stack",
    )
    parser.add_argument(
        "--head-variant",
        choices=("mean_linear", "last_avg", "last", "all"),
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
                PROTOCOL_CONTRACT,
                runtime_metadata,
                source_lock_metadata,
            )
        except ImportError:
            from .reve_upstream_heads import (
                PROTOCOL_CONTRACT,
                runtime_metadata,
                source_lock_metadata,
            )

        if args.head_variant == "mean_linear":
            query_initialization = "neuralbench_default"
        elif args.head_variant == "last_avg":
            query_initialization = "upstream_random_unused"
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
        if args.head_variant != "mean_linear":
            metadata["head_source_lock"] = source_lock_metadata()

        reports: list[dict[str, Any]] = []
        for seed in resolved_seeds:
            run_dir = args.output_dir / args.head_variant / f"seed{seed}"
            config_path = run_dir / "neuralbench_config.json"
            epoch_metrics_path = run_dir / "epoch_test_metrics.jsonl"
            _write_config(
                config_path,
                data_root=args.data_root,
                output_dir=run_dir,
            )
            seed_metadata = {**metadata, "seed": seed, "data_seed": 33}
            try:
                results = run_official_subset(
                    manifest_path=args.manifest,
                    data_root=args.data_root,
                    epoch_metrics_path=epoch_metrics_path,
                    config_path=config_path,
                    head_variant=args.head_variant,
                    seeds=(seed,),
                    run_metadata=seed_metadata,
                )
                report = {
                    "status": "completed",
                    **seed_metadata,
                    "official_results": results,
                    "epoch_metrics": str(epoch_metrics_path),
                    "artifacts": collect_run_artifacts(run_dir),
                }
            except Exception as error:
                failure_path = write_failure_diagnostics(
                    run_dir,
                    error,
                    launch_command=launch_command,
                    metadata=seed_metadata,
                )
                report = {
                    "status": "failed",
                    **seed_metadata,
                    "failure": str(failure_path),
                    "artifacts": collect_run_artifacts(run_dir),
                }
                reports.append(report)
                (run_dir / "report.json").write_text(
                    json.dumps(report, indent=2, default=str) + "\n",
                    encoding="utf-8",
                )
                raise

            reports.append(report)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "report.json").write_text(
                json.dumps(report, indent=2, default=str) + "\n",
                encoding="utf-8",
            )

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
