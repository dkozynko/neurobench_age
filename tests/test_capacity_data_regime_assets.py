from __future__ import annotations

import importlib.util
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
    analysis_path = tmp_path / "analysis.json"
    final_lock_path.write_text(json.dumps(final_lock, sort_keys=True), encoding="utf-8")
    analysis_path.write_text(json.dumps(analysis, sort_keys=True), encoding="utf-8")
    module = _load_script()
    manifest = module.build_capacity_data_regime_assets(
        analysis_path=analysis_path,
        final_lock_path=final_lock_path,
        output_root=tmp_path / "extension-assets",
        repository_root=ROOT,
    )

    assert manifest["status"] == "complete"
    assert len(manifest["output_files"]) == 6
    assert all((tmp_path / "extension-assets" / name).is_file() for name in manifest["output_files"])
    assert not (ROOT / "manuscript/generated/capacity_data_regime_cells.tex").exists()


def test_assets_refuse_primary_generated_output_root(tmp_path: Path) -> None:
    final_lock, prediction_inventory, checkpoint_inventory = _evidence()
    analysis = analyze_capacity_data_regime(
        final_lock=final_lock,
        prediction_inventory=prediction_inventory,
        checkpoint_inventory=checkpoint_inventory,
    )
    final_lock_path = tmp_path / "final_lock.json"
    analysis_path = tmp_path / "analysis.json"
    final_lock_path.write_text(json.dumps(final_lock, sort_keys=True), encoding="utf-8")
    analysis_path.write_text(json.dumps(analysis, sort_keys=True), encoding="utf-8")
    module = _load_script()
    with pytest.raises(module.CapacityDataRegimeAssetError, match="primary"):
        module.build_capacity_data_regime_assets(
            analysis_path=analysis_path,
            final_lock_path=final_lock_path,
            output_root=ROOT / "manuscript/generated/capacity-data-regime",
            repository_root=ROOT,
        )
