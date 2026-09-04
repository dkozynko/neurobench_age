from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from neurobench_age.pipelines import official


ROOT = Path(__file__).resolve().parents[1]


def test_article_phase_config_is_loaded_and_hashed() -> None:
    path = ROOT / "configs" / "article" / "validation_cycle.json"

    config = official._load_article_phase_config(path)

    assert config.payload["name"] == "article_validation_cycle"
    assert len(config.sha256) == 64


def test_article_phase_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "bad",
                "evaluation_mode": "validation_only",
                "test_access": "sealed",
                "seeds": [33],
                "baseline_head": "mean_linear",
                "protocol": "strict",
                "manifest": "fixture",
                "purpose": "test",
                "ignored_option": True,
            }
        )
    )

    with pytest.raises(ValueError, match="unknown fields"):
        official._load_article_phase_config(path)


def test_article_phase_config_rejects_cli_seed_mismatch() -> None:
    path = ROOT / "configs" / "article" / "validation_cycle.json"
    config = official._load_article_phase_config(path)
    args = SimpleNamespace(
        evaluation_mode="validation_only",
        evaluation_protocol="strict",
        seeds=[33, 34],
    )

    with pytest.raises(ValueError, match="seeds"):
        official._enforce_article_phase_config(args, config)


def test_article_phase_config_accepts_exact_runtime_options() -> None:
    path = ROOT / "configs" / "article" / "validation_cycle.json"
    config = official._load_article_phase_config(path)
    args = SimpleNamespace(
        evaluation_mode="validation_only",
        evaluation_protocol="strict",
        seeds=[33, 34, 35],
    )

    official._enforce_article_phase_config(args, config)
