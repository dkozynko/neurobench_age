from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

from neurobench_age.analysis.confirmatory import (
    confirmatory_conclusion,
    stable_improvement_decision,
)
from neurobench_age.research.study_lock import canonical_sha256
from neurobench_age.analysis.manuscript_assets import latex_breakable_hash


HEADS = (
    "mean_linear",
    "mean_layer_linear",
    "mean_rich_stats_residual",
    "multi_query_rich_stats",
)
SEEDS = tuple(range(33, 43))
CANDIDATES = HEADS[1:]
ROOT = Path(__file__).resolve().parents[1]


def test_latex_breakable_hash_preserves_identity_and_adds_safe_breaks() -> None:
    value = "0123456789abcdef" * 4
    rendered = latex_breakable_hash(value)
    assert rendered.replace(r"\allowbreak{}", "") == value
    assert rendered.count(r"\allowbreak{}") == 7


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _with_digest(payload: dict, field: str) -> dict:
    return {**payload, field: canonical_sha256(payload)}


def write_complete_source_bundle(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    subjects = [f"sub-{index:03d}" for index in range(1, 76)]
    statistics = {
        "alpha": 0.05,
        "bootstrap_iterations": 10_000,
        "bootstrap_seed": 20260903,
        "confidence": 0.95,
        "holm_order": list(CANDIDATES),
        "minimum_seed_wins": 8,
        "minimum_worst_seed_delta": -0.01,
        "randomization_tail": "greater",
        "require_ci_above_zero": True,
    }
    checkpoint_body = {
        "schema_version": 3,
        "status": "complete",
        "representation_protocol_sha256": "a" * 64,
        "training_protocol_sha256": "b" * 64,
        "training_source_sha256": "c" * 64,
        "heads": list(HEADS),
        "seeds": list(SEEDS),
        "run_count": 40,
        "runs": [
            {
                "head_name": head,
                "seed": seed,
                "representation_protocol_sha256": "a" * 64,
                "training_protocol_sha256": "b" * 64,
                "training_source_sha256": "c" * 64,
                "run_identity_sha256": canonical_sha256({"head": head, "seed": seed}),
                "run_manifest_sha256": canonical_sha256({"manifest": head, "seed": seed}),
                "checkpoint_sha256": canonical_sha256({"checkpoint": head, "seed": seed}),
                "selected_epoch": 3,
                "head_parameter_count": 100 + HEADS.index(head),
            }
            for head in HEADS
            for seed in SEEDS
        ],
    }
    checkpoint = _with_digest(checkpoint_body, "checkpoint_inventory_sha256")

    lock_body = {
        "schema_version": 2,
        "sealed_at_utc": "2026-09-09T00:00:00+00:00",
        "study_id": "reve_age_external_frozen_probe_v1",
        "protocol_sha256": "a" * 64,
        "training_protocol_sha256": "b" * 64,
        "representation_source_sha256": "d" * 64,
        "training_source_sha256": "c" * 64,
        "git_revision": "synthetic-revision",
        "git_dirty": False,
        "encoder_checkpoint": "brain-bzh/reve-base",
        "encoder_checkpoint_sha256": "e" * 64,
        "checkpoint_inventory_sha256": checkpoint["checkpoint_inventory_sha256"],
        "environment_sha256": "f" * 64,
        "hbn_manifest_sha256": "1" * 64,
        "hbn_training_manifest_sha256": "2" * 64,
        "mipdb_manifest_sha256": "3" * 64,
        "mipdb_pilot_qc_sha256": "4" * 64,
        "mipdb_cohort_qc_sha256": "5" * 64,
        "subject_list_sha256": {
            "hbn_train": "6" * 64,
            "hbn_validation": "7" * 64,
            "mipdb_pilot": "8" * 64,
            "mipdb_primary": canonical_sha256(subjects),
            "mipdb_extrapolation": "9" * 64,
        },
        "heads": list(HEADS),
        "seeds": list(SEEDS),
        "preprocessing_sha256": "0" * 64,
        "statistics_sha256": canonical_sha256(statistics),
        "output_root": str((tmp_path / "external-output").resolve()),
    }
    lock = _with_digest(lock_body, "lock_sha256")

    entries = [
        {
            "head_name": head,
            "seed": seed,
            "subject_id": subject,
            "path": f"predictions/{head}/seed-{seed}/{subject}.json",
            "prediction_sha256": canonical_sha256(
                {"head": head, "seed": seed, "subject": subject}
            ),
        }
        for head in HEADS
        for seed in SEEDS
        for subject in subjects
    ]
    prediction_body = {
        "schema_version": 3,
        "status": "complete",
        "study_id": lock["study_id"],
        "lock_sha256": lock["lock_sha256"],
        "protocol_sha256": lock["protocol_sha256"],
        "training_protocol_sha256": lock["training_protocol_sha256"],
        "heads": list(HEADS),
        "seeds": list(SEEDS),
        "subjects": subjects,
        "prediction_count": len(entries),
        "predictions": entries,
    }
    prediction = _with_digest(prediction_body, "prediction_inventory_sha256")

    comparisons = {}
    raw_p_values = (0.1, 0.2, 0.4)
    adjusted_p_values = (0.3, 0.4, 0.4)
    for index, head in enumerate(CANDIDATES):
        wins = 7 - index
        ci_low = -0.02 - index * 0.01
        worst = -0.03 - index * 0.01
        raw_p_value = raw_p_values[index]
        adjusted = adjusted_p_values[index]
        decision = stable_improvement_decision(
            adjusted_p_value=adjusted,
            bootstrap_ci_low=ci_low,
            wins=wins,
            worst_seed_delta=worst,
            alpha=statistics["alpha"],
            minimum_seed_wins=statistics["minimum_seed_wins"],
            minimum_worst_seed_delta=statistics["minimum_worst_seed_delta"],
            require_ci_above_zero=statistics["require_ci_above_zero"],
            adequately_powered=True,
        )
        per_seed = [
            {
                "seed": seed,
                "candidate": {
                    "pearson": 0.65 + index * 0.01,
                    "mae": 3.1,
                    "rmse": 3.8,
                    "r2": -0.5,
                    "calibration": {"intercept": -2.0, "slope": 1.5},
                },
                "baseline": {
                    "pearson": 0.64,
                    "mae": 3.2,
                    "rmse": 3.9,
                    "r2": -0.6,
                    "calibration": {"intercept": -3.0, "slope": 1.7},
                },
                "pearson_delta": 0.01 + index * 0.01,
            }
            for seed in SEEDS
        ]
        comparisons[head] = {
            "paired": {
                "seeds": list(SEEDS),
                "subject_count": 75,
                "per_seed": per_seed,
                "mean_pearson_delta": 0.01 + index * 0.01,
                "seed_delta_sample_sd": 0.01,
                "wins": wins,
                "ties": 0,
                "losses": 10 - wins,
                "worst_seed_delta": worst,
            },
            "bootstrap": {
                "ci_low": ci_low,
                "ci_high": 0.04 + index * 0.01,
                "confidence": 0.95,
                "iterations": 10_000,
                "valid_iterations": 10_000,
                "failed_iterations": 0,
                "seed": 20260903,
                "seed_count": 10,
                "subject_count": 75,
                "resampling": "paired_seeds_and_subjects_with_replacement",
            },
            "randomization": {
                "observed_mean_delta": 0.01 + index * 0.01,
                "p_value": raw_p_value,
                "permutations": 1024,
                "tail": "greater",
                "zero_deltas": 0,
                "comparison": "permuted_mean_greater_than_or_equal_to_observed",
            },
            "holm_adjusted_p_value": adjusted,
            "decision": decision,
            "candidate_metrics": {
                "per_seed": [row["candidate"] | {"seed": row["seed"]} for row in per_seed],
                "mean_pearson": 0.65 + index * 0.01,
                "pearson_sample_sd": 0.01,
                "mean_mae": 3.1,
                "mean_rmse": 3.8,
                "mean_r2": -0.5,
                "mean_calibration_intercept": -2.0,
                "mean_calibration_slope": 1.5,
            },
            "resources": {
                "head_parameter_count": 101 + index,
                "training_runtime_seconds_total": 100.0,
                "training_runtime_seconds_mean": 10.0,
                "peak_process_rss_bytes_max": 1000,
                "peak_accelerator_memory_bytes_max": 2000,
                "per_seed": [],
            },
        }
    baseline_per_seed = [
        {
            "seed": seed,
            "pearson": 0.64,
            "mae": 3.2,
            "rmse": 3.9,
            "r2": -0.6,
            "calibration": {"intercept": -3.0, "slope": 1.7},
        }
        for seed in SEEDS
    ]
    analysis_body = {
        "schema_version": 3,
        "status": "complete",
        "study_id": lock["study_id"],
        "lock_sha256": lock["lock_sha256"],
        "protocol_sha256": lock["protocol_sha256"],
        "training_protocol_sha256": lock["training_protocol_sha256"],
        "statistics_sha256": lock["statistics_sha256"],
        "prediction_inventory_sha256": prediction["prediction_inventory_sha256"],
        "estimand": "mean_across_seeds_candidate_minus_mean_linear_external_pearson",
        "baseline": {
            "head_name": "mean_linear",
            "metrics": {
                "per_seed": baseline_per_seed,
                "mean_pearson": 0.64,
                "pearson_sample_sd": 0.01,
                "mean_mae": 3.2,
                "mean_rmse": 3.9,
                "mean_r2": -0.6,
                "mean_calibration_intercept": -3.0,
                "mean_calibration_slope": 1.7,
            },
        },
        "comparisons": comparisons,
        "resources": {head: {"head_parameter_count": 100 + HEADS.index(head)} for head in HEADS},
        "cohort": {
            "pilot_subjects": 10,
            "primary_subjects": 75,
            "extrapolation_subjects": 20,
            "minimum_primary_subjects": 50,
            "underpowered": False,
            "power_warning": None,
        },
        "exclusions": [{"subject_id": "private-subject", "reason": "missing_age"}],
        "established_heads": [],
        "conclusion": confirmatory_conclusion([]),
        "protocol_statistics": statistics,
    }
    analysis = _with_digest(analysis_body, "analysis_sha256")

    return {
        "lock_path": _write_json(tmp_path / "lock.json", lock),
        "checkpoint_inventory_path": _write_json(
            tmp_path / "checkpoint_inventory.json", checkpoint
        ),
        "prediction_inventory_path": _write_json(
            tmp_path / "prediction_inventory.json", prediction
        ),
        "analysis_path": _write_json(tmp_path / "analysis.json", analysis),
    }


def test_validate_source_bundle_requires_exact_cartesian_inventory(tmp_path: Path) -> None:
    from neurobench_age.analysis.manuscript_assets import validate_source_bundle

    paths = write_complete_source_bundle(tmp_path)
    bundle = validate_source_bundle(**paths)

    assert bundle.prediction_count == 3000
    assert len(bundle.run_pairs) == 40
    assert len(bundle.subject_ids) == 75


def _rewrite_hashed(path: Path, field: str, mutate) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    body = {key: value for key, value in payload.items() if key != field}
    payload[field] = canonical_sha256(body)
    _write_json(path, payload)
    return payload


@pytest.mark.parametrize(
    ("case", "mutate"),
    [
        (
            "missing",
            lambda payload: (
                payload["predictions"].pop(),
                payload.__setitem__("prediction_count", 2999),
            ),
        ),
        (
            "duplicate",
            lambda payload: payload["predictions"].__setitem__(
                -1, dict(payload["predictions"][0])
            ),
        ),
        (
            "unknown-head",
            lambda payload: payload["predictions"][0].__setitem__(
                "head_name", "unknown"
            ),
        ),
        (
            "swapped-subject-order",
            lambda payload: payload["predictions"].__setitem__(
                slice(0, 2),
                [payload["predictions"][1], payload["predictions"][0]],
            ),
        ),
    ],
)
def test_validate_source_bundle_rejects_inventory_boundary_drift(
    tmp_path: Path, case: str, mutate
) -> None:
    from neurobench_age.analysis.manuscript_assets import (
        ManuscriptEvidenceError,
        validate_source_bundle,
    )

    paths = write_complete_source_bundle(tmp_path)
    _rewrite_hashed(paths["prediction_inventory_path"], "prediction_inventory_sha256", mutate)

    with pytest.raises(ManuscriptEvidenceError, match="prediction|subject"):
        validate_source_bundle(**paths)


def test_validate_source_bundle_rejects_malformed_prediction_digest(tmp_path: Path) -> None:
    from neurobench_age.analysis.manuscript_assets import (
        ManuscriptEvidenceError,
        validate_source_bundle,
    )

    paths = write_complete_source_bundle(tmp_path)
    payload = json.loads(paths["prediction_inventory_path"].read_text(encoding="utf-8"))
    payload["prediction_inventory_sha256"] = "not-a-digest"
    _write_json(paths["prediction_inventory_path"], payload)

    with pytest.raises(ManuscriptEvidenceError, match="does not match"):
        validate_source_bundle(**paths)


def test_validate_source_bundle_recomputes_complete_hash_chain(tmp_path: Path) -> None:
    from neurobench_age.analysis.manuscript_assets import (
        ManuscriptEvidenceError,
        validate_source_bundle,
    )

    paths = write_complete_source_bundle(tmp_path)
    _rewrite_hashed(
        paths["prediction_inventory_path"],
        "prediction_inventory_sha256",
        lambda payload: payload.__setitem__("lock_sha256", "f" * 64),
    )

    with pytest.raises(ManuscriptEvidenceError, match="lock|hash chain|provenance"):
        validate_source_bundle(**paths)


def test_validate_source_bundle_recomputes_decisions_from_primitives(tmp_path: Path) -> None:
    from neurobench_age.analysis.manuscript_assets import (
        ManuscriptEvidenceError,
        validate_source_bundle,
    )

    paths = write_complete_source_bundle(tmp_path)
    _rewrite_hashed(
        paths["analysis_path"],
        "analysis_sha256",
        lambda payload: payload["comparisons"]["mean_layer_linear"]["paired"].__setitem__(
            "wins", 10
        ),
    )

    with pytest.raises(ManuscriptEvidenceError, match="decision"):
        validate_source_bundle(**paths)


def test_export_compact_evidence_is_private_path_free_and_idempotent(tmp_path: Path) -> None:
    from neurobench_age.analysis.manuscript_assets import (
        export_compact_evidence,
        validate_source_bundle,
    )

    paths = write_complete_source_bundle(tmp_path / "source")
    destination = tmp_path / "prospective"
    export_compact_evidence(validate_source_bundle(**paths), destination)
    first = {path.name: path.read_bytes() for path in destination.iterdir()}
    export_compact_evidence(validate_source_bundle(**paths), destination)

    assert set(first) == {
        "confirmatory_analysis.json",
        "study_summary.json",
        "artifact_manifest.json",
        "artifact_manifest.sha256",
    }
    assert first == {path.name: path.read_bytes() for path in destination.iterdir()}
    combined = b"".join(first.values())
    assert b"predictions/" not in combined
    assert b"private-subject" not in combined
    assert str(tmp_path).encode() not in combined
    compact = json.loads(first["confirmatory_analysis.json"])
    assert compact["exclusions_by_reason"] == {"missing_age": 1}
    manifest = json.loads(first["artifact_manifest.json"])
    assert set(manifest["files"]) == {
        "confirmatory_analysis.json",
        "study_summary.json",
    }


def test_export_compact_evidence_refuses_conflicting_destination(tmp_path: Path) -> None:
    from neurobench_age.analysis.manuscript_assets import (
        ManuscriptEvidenceError,
        export_compact_evidence,
        validate_source_bundle,
    )

    paths = write_complete_source_bundle(tmp_path / "source")
    destination = tmp_path / "prospective"
    destination.mkdir()
    (destination / "unexpected.txt").write_text("conflict\n", encoding="utf-8")

    with pytest.raises(ManuscriptEvidenceError, match="destination|conflict"):
        export_compact_evidence(validate_source_bundle(**paths), destination)


def test_cli_import_emits_only_compact_status(tmp_path: Path, capsys) -> None:
    script = ROOT / "scripts" / "build_manuscript_assets.py"
    spec = importlib.util.spec_from_file_location("build_manuscript_assets", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = write_complete_source_bundle(tmp_path / "source")
    output = tmp_path / "prospective"

    result = module.main(
        [
            "import",
            "--analysis",
            str(paths["analysis_path"]),
            "--lock",
            str(paths["lock_path"]),
            "--checkpoint-inventory",
            str(paths["checkpoint_inventory_path"]),
            "--prediction-inventory",
            str(paths["prediction_inventory_path"]),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr().out
    assert result == 0
    assert "status=complete" in captured
    assert "predictions/" not in captured
    assert str(tmp_path) not in captured


def test_latex_formatters_are_stable_and_safe() -> None:
    from neurobench_age.analysis.manuscript_assets import (
        format_decimal,
        format_interval,
        format_p_value,
        latex_escape,
    )

    assert latex_escape(r"A_B & 5% #1") == r"A\_B \& 5\% \#1"
    assert format_decimal(-0.00001, digits=3) == "0.000"
    assert format_decimal(0.644371839, digits=3) == "0.644"
    assert format_interval(-0.01886, 0.03595, digits=3) == r"[-0.019,\ 0.036]"
    assert format_p_value(0.0004) == r"<0.001"
    assert format_p_value(0.03125) == "0.031"
    with pytest.raises(ValueError, match="finite"):
        format_decimal(float("nan"), digits=3)


def test_rendered_text_assets_use_fixed_head_order(tmp_path: Path) -> None:
    from neurobench_age.analysis.manuscript_assets import (
        export_compact_evidence,
        render_text_assets,
        validate_source_bundle,
    )

    paths = write_complete_source_bundle(tmp_path / "source")
    evidence = tmp_path / "evidence"
    export_compact_evidence(validate_source_bundle(**paths), evidence)
    analysis = json.loads(
        (evidence / "confirmatory_analysis.json").read_text(encoding="utf-8")
    )

    assets = render_text_assets(analysis)

    assert set(assets) == {
        "results_macros.tex",
        "main_metrics.tex",
        "confirmatory_comparisons.tex",
        "cohort_summary.tex",
    }
    rows = assets["main_metrics.tex"]
    assert rows.index("Mean-pooled linear") < rows.index("Penultimate-layer linear")
    assert rows.index("Penultimate-layer linear") < rows.index("Rich-statistics residual")
    assert rows.index("Rich-statistics residual") < rows.index("Multi-query rich-statistics")
    assert r"\newcommand{\BaselinePearson}{0.640}" in assets["results_macros.tex"]


def test_render_assets_is_transactional_and_byte_deterministic(tmp_path: Path) -> None:
    from neurobench_age.analysis.manuscript_assets import (
        export_compact_evidence,
        render_assets,
        validate_source_bundle,
    )

    paths = write_complete_source_bundle(tmp_path / "source")
    evidence = tmp_path / "evidence"
    export_compact_evidence(validate_source_bundle(**paths), evidence)
    first = tmp_path / "first"
    second = tmp_path / "second"

    render_assets(evidence, first)
    render_assets(evidence, second)

    expected = {
        "results_macros.tex",
        "main_metrics.tex",
        "confirmatory_comparisons.tex",
        "cohort_summary.tex",
        "seed_deltas.pdf",
        "bootstrap_intervals.pdf",
        "calibration_summary.pdf",
        "assets_manifest.json",
        "assets_manifest.sha256",
    }
    assert {path.name for path in first.iterdir()} == expected
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    assert b"No difference" in (first / "seed_deltas.pdf").read_bytes()
    manifest = json.loads((first / "assets_manifest.json").read_text(encoding="utf-8"))
    assert "assets_manifest.json" not in manifest["files"]
    assert "assets_manifest.sha256" not in manifest["files"]


def test_render_assets_rolls_back_on_renderer_failure(tmp_path: Path, monkeypatch) -> None:
    import neurobench_age.analysis.manuscript_assets as assets

    paths = write_complete_source_bundle(tmp_path / "source")
    evidence = tmp_path / "evidence"
    assets.export_compact_evidence(assets.validate_source_bundle(**paths), evidence)
    destination = tmp_path / "generated"

    def fail(_analysis):
        raise RuntimeError("synthetic renderer failure")

    monkeypatch.setattr(assets, "_render_figure_payloads", fail)
    with pytest.raises(RuntimeError, match="synthetic renderer failure"):
        assets.render_assets(evidence, destination)

    assert not destination.exists()


def test_cli_render_emits_manifest_identity(tmp_path: Path, capsys) -> None:
    script = ROOT / "scripts" / "build_manuscript_assets.py"
    spec = importlib.util.spec_from_file_location("build_manuscript_assets_render", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = write_complete_source_bundle(tmp_path / "source")
    evidence = tmp_path / "evidence"
    from neurobench_age.analysis.manuscript_assets import (
        export_compact_evidence,
        validate_source_bundle,
    )

    export_compact_evidence(validate_source_bundle(**paths), evidence)
    output = tmp_path / "generated"

    result = module.main(
        ["render", "--evidence", str(evidence), "--output", str(output)]
    )

    captured_streams = capsys.readouterr()
    captured = captured_streams.out
    assert result == 0
    assert "status=complete" in captured
    assert "assets_manifest_sha256=" in captured
    assert str(tmp_path) not in captured
    assert captured_streams.err == ""
