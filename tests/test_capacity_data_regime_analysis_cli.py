from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_capacity_data_regime.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("capacity_analysis_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analysis_cli_accepts_exploratory_inference_config() -> None:
    module = _load_script()
    actions = {action.dest for action in module.build_parser()._actions}

    assert "exploratory_inference_config" in actions
