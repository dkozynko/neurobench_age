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
from pathlib import Path
from typing import Any, Iterable, Sequence

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

    def __init__(self, test_loader: Any, output_path: Path | None = None):
        self.test_loader = test_loader
        self.output_path = output_path
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
        metric = PearsonCorrCoef().to(pl_module.device)
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
                metric.update(y_pred, y_true)

        score = float(metric.compute().detach().cpu())
        record = {
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


def _patch_official_components(
    manifest_path: Path,
    data_root: Path,
    epoch_metrics_path: Path,
) -> tuple[Any, Any, Any, Any]:
    """Patch only discovery and callback attachment; return originals."""

    from neuralbench.data import Data
    from neuralbench.main import Experiment
    from neuralfetch.studies import shirazi2024hbn

    timelines = load_manifest_timelines(manifest_path, data_root)

    original_iter_timelines = shirazi2024hbn.Shirazi2024Hbn.iter_timelines
    original_info = shirazi2024hbn.Shirazi2024Hbn._info
    original_prepare = Data.prepare
    original_setup_trainer = Experiment.setup_trainer
    original_load_yaml_config = None

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

    def setup_with_epoch_test(self: Any, is_test: bool = False) -> Any:
        trainer = original_setup_trainer(self, is_test=is_test)
        if not is_test:
            loaders = captured_loaders.get(id(self.data))
            if loaders is None or "test" not in loaders:
                raise RuntimeError("official test loader was not captured")
            trainer.callbacks.append(
                EpochTestPearson(loaders["test"], epoch_metrics_path)
            )
        return trainer

    Experiment.setup_trainer = setup_with_epoch_test

    # A full NeuralBench run expands the default seed grid (33, 34, 35).
    # The parity run is intentionally one seed, matching the independent run.
    import neuralbench.cli as cli

    original_load_yaml_config = cli.load_yaml_config

    def load_single_seed_grid(path: Path, safe: bool = False) -> Any:
        if path.name == "grid.yaml":
            return {"seed": [33]}
        return original_load_yaml_config(path, safe=safe)

    cli.load_yaml_config = load_single_seed_grid
    return (
        original_iter_timelines,
        original_info,
        original_prepare,
        original_setup_trainer,
        (cli, original_load_yaml_config),
    )


def _restore_official_components(originals: tuple[Any, Any, Any, Any]) -> None:
    from neuralbench.data import Data
    from neuralbench.main import Experiment
    from neuralfetch.studies import shirazi2024hbn

    original_iter, original_info, original_prepare, original_setup, cli_state = originals
    cli, original_load_yaml_config = cli_state
    shirazi2024hbn.Shirazi2024Hbn.iter_timelines = original_iter
    shirazi2024hbn.Shirazi2024Hbn._info = original_info
    Data.prepare = original_prepare
    Experiment.setup_trainer = original_setup
    cli.load_yaml_config = original_load_yaml_config


def run_official_subset(
    *,
    manifest_path: Path,
    data_root: Path,
    epoch_metrics_path: Path,
    config_path: Path,
) -> list[dict[str, Any]]:
    """Run official REVE once on the manifest and collect cached results."""

    os.environ["NEURALBENCH_CONFIG"] = str(config_path)
    originals = _patch_official_components(
        manifest_path,
        data_root,
        epoch_metrics_path,
    )
    try:
        from neuralbench import run_benchmark

        run_benchmark(
            device="eeg",
            task="age",
            model="reve",
            force=True,
        )
        # The public runner launches/finishes the experiment but intentionally
        # returns no result for local non-debug execution.  A cached-only
        # collection uses the same official config and performs no training.
        return run_benchmark(
            device="eeg",
            task="age",
            model="reve",
            plot_cached=True,
        )
    finally:
        _restore_official_components(originals)


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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)

    digest = manifest_sha256(args.manifest)
    _write_config(args.config, data_root=args.data_root, output_dir=args.output_dir)
    epoch_metrics_path = args.output_dir / "epoch_test_metrics.jsonl"
    results = run_official_subset(
        manifest_path=args.manifest,
        data_root=args.data_root,
        epoch_metrics_path=epoch_metrics_path,
        config_path=args.config,
    )
    report = {
        "manifest": str(args.manifest),
        "manifest_sha256": digest,
        "rows": len(load_manifest_timelines(args.manifest, args.data_root)),
        "official_results": results,
        "epoch_metrics": str(epoch_metrics_path),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
