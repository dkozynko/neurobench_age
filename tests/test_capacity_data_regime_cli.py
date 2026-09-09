from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_capacity_data_regime.py"


def _load_script():
    if not SCRIPT.is_file():
        pytest.fail(f"missing extension runner CLI: {SCRIPT}")
    spec = importlib.util.spec_from_file_location("run_capacity_data_regime", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extension_cli_exposes_preflight_and_training_arguments() -> None:
    module = _load_script()
    parser = module.build_parser()
    actions = {action.dest for action in parser._actions}

    assert {
        "protocol",
        "training_manifest",
        "cache_root",
        "output_root",
        "primary_study_lock",
        "primary_prediction_inventory",
        "device",
        "preflight_only",
    }.issubset(actions)


def test_pilot_only_does_not_apply_full_run_disk_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_script()
    cohort = tmp_path / "cohort.csv"
    with cohort.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("release", "subject", "split"))
        writer.writeheader()
        writer.writerow({"release": "R1", "subject": "s1", "split": "train"})

    extension = SimpleNamespace(
        extension_id="reve_age_capacity_data_regime_v1",
        sha256="a" * 64,
        representation_protocol_path="representation.json",
        training_protocol_path="training.json",
        head_names=("mean_linear", "mean_rich_stats_residual", "mean_mlp_residual_matched(hidden_dim=4)"),
        training_sizes=(200, 400, 800),
        seeds=tuple(range(33, 43)),
        expected_run_count=90,
        expected_prediction_count=6750,
    )
    monkeypatch.setattr(module, "load_capacity_data_regime_protocol", lambda *args, **kwargs: extension)
    monkeypatch.setattr(module, "load_study_protocol", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "load_frozen_probe_training_protocol", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "load_frozen_probe_training_manifest", lambda *args, **kwargs: ())
    monkeypatch.setattr(module, "build_nested_training_cohorts", lambda *args, **kwargs: {200: ("s1",), 400: ("s1",), 800: ("s1",)})
    monkeypatch.setattr(module, "build_cache_manifest_from_records", lambda *args, **kwargs: ((), "b" * 64, 0, 1, 1))
    monkeypatch.setattr(module, "_available_memory_bytes", lambda: 1024**3)
    monkeypatch.setattr(module, "source_tree_sha256", lambda *args, **kwargs: "c" * 64)
    monkeypatch.setattr(module, "compute_capacity_preflight", lambda *args, **kwargs: pytest.fail("full-run preflight must not run for pilot-only"))
    monkeypatch.setattr(module, "run_capacity_pilot", lambda **kwargs: {"status": "pilot_complete"})

    result = module.main(
        [
            "--protocol", str(tmp_path / "protocol.json"),
            "--training-manifest", str(tmp_path / "training.json"),
            "--cohort-manifest", str(cohort),
            "--cache-root", str(tmp_path / "cache"),
            "--output-root", str(tmp_path / "output"),
            "--primary-study-lock", str(tmp_path / "primary-lock.json"),
            "--primary-prediction-inventory", str(tmp_path / "primary-predictions.json"),
            "--estimated-extension-output-bytes", "1073741824",
            "--pilot-only",
        ]
    )

    assert result == 0
