from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

from neurobench_age.analysis.capacity_data_regime import analyze_capacity_data_regime


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_capacity_data_regime_assets.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("capacity_assets", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evidence():
    source = ROOT / "tests/test_capacity_data_regime_analysis.py"
    spec = importlib.util.spec_from_file_location("capacity_analysis_fixture", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._evidence()


def test_assets_are_isolated_and_bound_to_analysis_and_final_lock(tmp_path: Path) -> None:
    final_lock, prediction_inventory, checkpoint_inventory = _evidence()
    analysis = analyze_capacity_data_regime(
        final_lock=final_lock,
        prediction_inventory=prediction_inventory,
        checkpoint_inventory=checkpoint_inventory,
    )
    final_lock_path = tmp_path / "final_lock.json"
    prediction_inventory_path = tmp_path / "prediction_inventory.json"
    checkpoint_inventory_path = tmp_path / "checkpoint_inventory.json"
    analysis_path = tmp_path / "analysis.json"
    final_lock_path.write_text(json.dumps(final_lock, sort_keys=True), encoding="utf-8")
    prediction_inventory_path.write_text(
        json.dumps(prediction_inventory, sort_keys=True), encoding="utf-8"
    )
    checkpoint_inventory_path.write_text(
        json.dumps(checkpoint_inventory, sort_keys=True), encoding="utf-8"
    )
    analysis_path.write_text(json.dumps(analysis, sort_keys=True), encoding="utf-8")
    module = _load_script()
    manifest = module.build_capacity_data_regime_assets(
        analysis_path=analysis_path,
        final_lock_path=final_lock_path,
        checkpoint_inventory_path=checkpoint_inventory_path,
        prediction_inventory_path=prediction_inventory_path,
        output_root=tmp_path / "extension-assets",
        repository_root=ROOT,
    )

    assert manifest["status"] == "complete"
    expected = {
        "capacity_data_regime_cells.csv",
        "capacity_data_regime_cells.tex",
        "capacity_data_regime_contrasts.csv",
        "capacity_data_regime_contrasts.tex",
        "capacity_data_regime_absolute.tex",
        "capacity_data_regime_absolute.pdf",
        "capacity_data_regime_delta.pdf",
        "capacity_data_regime_seed_deltas.csv",
        "capacity_data_regime_seed_deltas.pdf",
        "capacity_data_regime_validation_external_transfer.tex",
    }
    assert set(manifest["files"]) == expected
    output = tmp_path / "extension-assets"
    assert all((output / name).is_file() for name in expected)
    assert all(set(metadata) == {"bytes", "sha256"} for metadata in manifest["files"].values())
    for name, metadata in manifest["files"].items():
        payload = (output / name).read_bytes()
        assert metadata["bytes"] == len(payload)
        assert metadata["sha256"] == hashlib.sha256(payload).hexdigest()
    cells = (output / "capacity_data_regime_cells.csv").read_text(encoding="utf-8")
    assert "parameter_count" in cells
    assert "513" in cells
    for path in output.rglob("*"):
        if path.name == "capacity_data_regime_asset_manifest.json":
            continue
        if path.suffix.casefold() in {".csv", ".json", ".md", ".tex"}:
            text = path.read_text(encoding="utf-8")
            assert "subject_id" not in text
            assert "true_age" not in text
            assert "prediction" not in text
            assert "sub-" not in text
    assert not (ROOT / "manuscript/generated/capacity_data_regime_cells.tex").exists()


def test_assets_refuse_primary_generated_output_root(tmp_path: Path) -> None:
    final_lock, prediction_inventory, checkpoint_inventory = _evidence()
    analysis = analyze_capacity_data_regime(
        final_lock=final_lock,
        prediction_inventory=prediction_inventory,
        checkpoint_inventory=checkpoint_inventory,
    )
    final_lock_path = tmp_path / "final_lock.json"
    prediction_inventory_path = tmp_path / "prediction_inventory.json"
    checkpoint_inventory_path = tmp_path / "checkpoint_inventory.json"
    analysis_path = tmp_path / "analysis.json"
    final_lock_path.write_text(json.dumps(final_lock, sort_keys=True), encoding="utf-8")
    prediction_inventory_path.write_text(
        json.dumps(prediction_inventory, sort_keys=True), encoding="utf-8"
    )
    checkpoint_inventory_path.write_text(
        json.dumps(checkpoint_inventory, sort_keys=True), encoding="utf-8"
    )
    analysis_path.write_text(json.dumps(analysis, sort_keys=True), encoding="utf-8")
    module = _load_script()
    with pytest.raises(module.CapacityDataRegimeAssetError, match="primary"):
        module.build_capacity_data_regime_assets(
            analysis_path=analysis_path,
            final_lock_path=final_lock_path,
            checkpoint_inventory_path=checkpoint_inventory_path,
            prediction_inventory_path=prediction_inventory_path,
            output_root=ROOT / "manuscript/generated/capacity-data-regime",
            repository_root=ROOT,
        )


def test_assets_reject_a_prediction_inventory_hash_mismatch(tmp_path: Path) -> None:
    final_lock, prediction_inventory, checkpoint_inventory = _evidence()
    analysis = analyze_capacity_data_regime(
        final_lock=final_lock,
        prediction_inventory=prediction_inventory,
        checkpoint_inventory=checkpoint_inventory,
    )
    final_lock_path = tmp_path / "final_lock.json"
    prediction_inventory_path = tmp_path / "prediction_inventory.json"
    checkpoint_inventory_path = tmp_path / "checkpoint_inventory.json"
    analysis_path = tmp_path / "analysis.json"
    final_lock_path.write_text(json.dumps(final_lock), encoding="utf-8")
    prediction_inventory["predictions"][0]["prediction"] += 1.0
    prediction_inventory_path.write_text(json.dumps(prediction_inventory), encoding="utf-8")
    checkpoint_inventory_path.write_text(json.dumps(checkpoint_inventory), encoding="utf-8")
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    module = _load_script()

    with pytest.raises(module.CapacityDataRegimeAssetError, match="prediction"):
        module.build_capacity_data_regime_assets(
            analysis_path=analysis_path,
            final_lock_path=final_lock_path,
            checkpoint_inventory_path=checkpoint_inventory_path,
            prediction_inventory_path=prediction_inventory_path,
            output_root=tmp_path / "extension-assets",
            repository_root=ROOT,
        )
