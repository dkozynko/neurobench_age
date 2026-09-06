from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from neurobench_age.analysis.confirmatory import (
    ConfirmatoryAnalysisError,
    PredictionSeries,
    calibration_parameters,
    confirmatory_conclusion,
    exact_seed_randomization,
    hierarchical_paired_bootstrap,
    holm_step_down,
    paired_seed_statistics,
    regression_metrics,
    stable_improvement_decision,
    analyze_confirmatory_study,
)
from neurobench_age.pipelines.frozen_probe_training import APPROVED_HEADS
from neurobench_age.research.protocol import load_study_protocol
from neurobench_age.research.study_lock import (
    canonical_sha256,
    seal_study,
    transition_study,
)


SEEDS = tuple(range(33, 43))
ROOT = Path(__file__).resolve().parents[1]


def _series(seed: int, predictions: tuple[float, ...]) -> PredictionSeries:
    targets = tuple(float(index) for index in range(1, len(predictions) + 1))
    return PredictionSeries(
        seed=seed,
        subject_ids=tuple(f"sub-{index:03d}" for index in range(len(predictions))),
        targets=targets,
        predictions=predictions,
        provenance={"lock_sha256": "a" * 64},
    )


def test_regression_and_calibration_match_reference_values() -> None:
    targets = (1.0, 2.0, 3.0, 4.0)
    predictions = (1.0, 2.0, 4.0, 3.0)

    metrics = regression_metrics(targets, predictions)
    calibration = calibration_parameters(targets, predictions)

    assert metrics == pytest.approx(
        {"pearson": 0.8, "mae": 0.5, "rmse": math.sqrt(0.5), "r2": 0.6}
    )
    assert calibration == pytest.approx({"intercept": 0.5, "slope": 0.8})


@pytest.mark.parametrize(
    ("targets", "predictions"),
    [
        ((1.0,), (1.0,)),
        ((1.0, 2.0), (1.0,)),
        ((1.0, 1.0), (1.0, 2.0)),
        ((1.0, 2.0), (1.0, 1.0)),
        ((1.0, 2.0), (1.0, float("nan"))),
    ],
)
def test_metrics_reject_undefined_or_nonfinite_inputs(targets, predictions) -> None:
    with pytest.raises(ConfirmatoryAnalysisError):
        regression_metrics(targets, predictions)


def test_pairing_requires_exact_ten_seeds_subject_order_targets_and_provenance() -> None:
    baseline = {
        seed: _series(seed, (1.2, 2.1, 2.8, 4.2)) for seed in SEEDS
    }
    candidate = {
        seed: _series(seed, (1.1, 2.0, 2.9, 4.1)) for seed in SEEDS
    }

    summary = paired_seed_statistics(candidate, baseline)

    assert summary["seeds"] == list(SEEDS)
    assert len(summary["per_seed"]) == 10

    with pytest.raises(ConfirmatoryAnalysisError, match="seed inventory"):
        paired_seed_statistics({seed: candidate[seed] for seed in SEEDS[:-1]}, baseline)

    changed_order = dict(candidate)
    item = candidate[33]
    changed_order[33] = PredictionSeries(
        seed=33,
        subject_ids=tuple(reversed(item.subject_ids)),
        targets=tuple(reversed(item.targets)),
        predictions=tuple(reversed(item.predictions)),
        provenance=item.provenance,
    )
    with pytest.raises(ConfirmatoryAnalysisError, match="subject order"):
        paired_seed_statistics(changed_order, baseline)

    changed_provenance = dict(candidate)
    changed_provenance[33] = PredictionSeries(
        seed=33,
        subject_ids=item.subject_ids,
        targets=item.targets,
        predictions=item.predictions,
        provenance={"lock_sha256": "b" * 64},
    )
    with pytest.raises(ConfirmatoryAnalysisError, match="provenance"):
        paired_seed_statistics(changed_provenance, baseline)


def test_hierarchical_bootstrap_is_deterministic_and_preserves_pairing() -> None:
    targets = tuple(float(value) for value in range(8, 20))
    baseline = {}
    candidate = {}
    for offset, seed in enumerate(SEEDS):
        noise = np.sin(np.arange(len(targets)) + offset) * 2.0
        baseline_predictions = tuple(
            float(target + error) for target, error in zip(targets, noise)
        )
        candidate_predictions = tuple(
            float(target + error * 0.35) for target, error in zip(targets, noise)
        )
        subject_ids = tuple(f"sub-{index:03d}" for index in range(len(targets)))
        provenance = {"lock_sha256": "a" * 64}
        baseline[seed] = PredictionSeries(
            seed, subject_ids, targets, baseline_predictions, provenance
        )
        candidate[seed] = PredictionSeries(
            seed, subject_ids, targets, candidate_predictions, provenance
        )

    first = hierarchical_paired_bootstrap(
        candidate, baseline, iterations=250, seed=20260903, confidence=0.95
    )
    second = hierarchical_paired_bootstrap(
        candidate, baseline, iterations=250, seed=20260903, confidence=0.95
    )

    assert first == second
    assert first["iterations"] == 250
    assert first["valid_iterations"] + first["failed_iterations"] == 250
    assert first["resampling"] == "paired_seeds_and_subjects_with_replacement"
    assert first["ci_low"] > 0.0


def test_exact_randomization_enumerates_all_sign_flips() -> None:
    result = exact_seed_randomization([0.1] * 10)

    assert result["permutations"] == 1024
    assert result["tail"] == "greater"
    assert result["zero_deltas"] == 0
    assert result["p_value"] == pytest.approx(1.0 / 1024.0)

    null = exact_seed_randomization([0.0] * 10)
    assert null["p_value"] == 1.0
    assert null["zero_deltas"] == 10


def test_holm_step_down_matches_reference_example_and_fixed_tie_order() -> None:
    adjusted = holm_step_down(
        {"a": 0.01, "b": 0.04, "c": 0.03}, order=("a", "b", "c")
    )

    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
    tied = holm_step_down(
        {"a": 0.02, "b": 0.02, "c": 0.5}, order=("b", "a", "c")
    )
    assert list(tied) == ["b", "a", "c"]


def test_stable_improvement_requires_every_predeclared_condition() -> None:
    passing = stable_improvement_decision(
        adjusted_p_value=0.01,
        bootstrap_ci_low=0.02,
        wins=8,
        worst_seed_delta=-0.01,
        alpha=0.05,
        minimum_seed_wins=8,
        minimum_worst_seed_delta=-0.01,
        require_ci_above_zero=True,
        adequately_powered=True,
    )

    assert passing["established_stable_improvement"] is True
    assert all(passing["conditions"].values())

    for field, value in (
        ("adjusted_p_value", 0.05),
        ("bootstrap_ci_low", 0.0),
        ("wins", 7),
        ("worst_seed_delta", -0.011),
    ):
        arguments = {
            "adjusted_p_value": 0.01,
            "bootstrap_ci_low": 0.02,
            "wins": 8,
            "worst_seed_delta": -0.01,
            "alpha": 0.05,
            "minimum_seed_wins": 8,
            "minimum_worst_seed_delta": -0.01,
            "require_ci_above_zero": True,
            "adequately_powered": True,
        }
        arguments[field] = value
        assert stable_improvement_decision(**arguments)[
            "established_stable_improvement"
        ] is False

    underpowered = stable_improvement_decision(
        adjusted_p_value=0.01,
        bootstrap_ci_low=0.02,
        wins=10,
        worst_seed_delta=0.01,
        alpha=0.05,
        minimum_seed_wins=8,
        minimum_worst_seed_delta=-0.01,
        require_ci_above_zero=True,
        adequately_powered=False,
    )
    assert underpowered["established_stable_improvement"] is False
    assert underpowered["conditions"]["adequate_primary_cohort"] is False


def test_negative_conclusion_does_not_claim_equivalence() -> None:
    conclusion = confirmatory_conclusion([])

    assert conclusion == (
        "No tested complex head established a stable external gain under the "
        "predeclared protocol; this does not establish equivalence."
    )
    assert "equivalent" not in conclusion.casefold()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_completed_study(tmp_path: Path) -> dict[str, Path]:
    protocol_path = ROOT / "configs/research/external_frozen_probe.json"
    protocol = load_study_protocol(protocol_path)
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_runs = []
    for head_index, head_name in enumerate(APPROVED_HEADS):
        for seed in SEEDS:
            run_dir = checkpoint_root / head_name / f"seed-{seed}"
            run_dir.mkdir(parents=True)
            checkpoint_sha256 = hashlib.sha256(
                f"{head_name}:{seed}".encode()
            ).hexdigest()
            run_body = {
                "schema_version": 1,
                "status": "complete",
                "head_name": head_name,
                "seed": seed,
                "run_identity_sha256": "1" * 64,
                "checkpoint_sha256": checkpoint_sha256,
                "selected_epoch": 1,
                "head_parameters": {
                    "total": 10 + head_index,
                    "trainable": 10 + head_index,
                },
                "runtime_seconds": float(seed - 30 + head_index),
                "peak_process_rss_bytes": 1000 + seed,
                "peak_accelerator_memory_bytes": 2000 + seed,
            }
            run_manifest = {
                **run_body,
                "run_manifest_sha256": canonical_sha256(run_body),
            }
            run_path = run_dir / "run_manifest.json"
            run_path.write_text(json.dumps(run_manifest, sort_keys=True) + "\n")
            checkpoint_runs.append(
                {
                    "head_name": head_name,
                    "seed": seed,
                    "run_identity_sha256": "1" * 64,
                    "run_manifest_sha256": run_manifest["run_manifest_sha256"],
                    "checkpoint_sha256": checkpoint_sha256,
                    "selected_epoch": 1,
                    "head_parameter_count": 10 + head_index,
                }
            )
    checkpoint_body = {
        "schema_version": 1,
        "status": "complete",
        "heads": list(APPROVED_HEADS),
        "seeds": list(SEEDS),
        "run_count": 40,
        "runs": checkpoint_runs,
    }
    checkpoint_inventory = {
        **checkpoint_body,
        "checkpoint_inventory_sha256": canonical_sha256(checkpoint_body),
    }
    checkpoint_inventory_path = checkpoint_root / "checkpoint_inventory.json"
    checkpoint_inventory_path.write_text(
        json.dumps(checkpoint_inventory, sort_keys=True) + "\n"
    )

    pilot = [f"sub-pilot-{index:02d}" for index in range(10)]
    primary = [f"sub-{index:03d}" for index in range(6)]
    ages = {subject_id: float(8 + index) for index, subject_id in enumerate(primary)}
    subjects = [
        {"subject_id": subject_id, "age": float(index + 1), "recordings": ["rest.set"]}
        for index, subject_id in enumerate(pilot)
    ] + [
        {"subject_id": subject_id, "age": ages[subject_id], "recordings": ["rest.set"]}
        for subject_id in primary
    ]
    manifest = {
        "schema_version": 2,
        "status": "finalized",
        "dataset": "MIPDB",
        "protocol_sha256": protocol.sha256,
        "dataset_manifest_sha256": "b" * 64,
        "subjects": subjects,
        "exclusions": [{"subject_id": "sub-excluded", "reason": "missing_age"}],
        "cohorts": {"pilot": pilot, "primary": primary, "extrapolation": []},
        "subject_list_sha256": {
            "pilot": canonical_sha256(pilot),
            "primary": canonical_sha256(primary),
            "extrapolation": canonical_sha256([]),
        },
        "underpowered": True,
        "minimum_primary_subjects": 50,
        "cohort_qc_sha256": "7" * 64,
    }
    manifest_path = tmp_path / "mipdb_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    prediction_root = tmp_path / "predictions"
    environment_path = tmp_path / "environment.lock"
    environment_path.write_text("synthetic environment\n")
    lock_path = tmp_path / "study_lock.json"
    statistics_payload = asdict(protocol.statistics)
    statistics_payload["holm_order"] = list(protocol.statistics.holm_order)
    lock = seal_study(
        lock_path,
        {
            "study_id": protocol.study_id,
            "protocol_sha256": protocol.sha256,
            "source_tree_sha256": "c" * 64,
            "git_revision": "synthetic-revision",
            "git_dirty": False,
            "encoder_checkpoint": "brain-bzh/reve-base",
            "encoder_checkpoint_sha256": "d" * 64,
            "checkpoint_inventory_sha256": checkpoint_inventory[
                "checkpoint_inventory_sha256"
            ],
            "environment_sha256": _sha256_file(environment_path),
            "hbn_manifest_sha256": "e" * 64,
            "hbn_training_manifest_sha256": "5" * 64,
            "mipdb_manifest_sha256": _sha256_file(manifest_path),
            "mipdb_pilot_qc_sha256": "6" * 64,
            "mipdb_cohort_qc_sha256": "7" * 64,
            "subject_list_sha256": {
                "hbn_train": "1" * 64,
                "hbn_validation": "2" * 64,
                "mipdb_pilot": manifest["subject_list_sha256"]["pilot"],
                "mipdb_primary": manifest["subject_list_sha256"]["primary"],
                "mipdb_extrapolation": manifest["subject_list_sha256"][
                    "extrapolation"
                ],
            },
            "heads": list(APPROVED_HEADS),
            "seeds": list(SEEDS),
            "preprocessing_sha256": "f" * 64,
            "statistics_sha256": canonical_sha256(statistics_payload),
            "output_root": str(prediction_root),
        },
    )
    transition_study(lock_path, "started")

    entries = []
    noise = np.asarray((0.0, 2.0, -2.0, 1.5, -1.0, 0.5))
    targets = np.asarray([ages[subject_id] for subject_id in primary])
    for head_name in APPROVED_HEADS:
        for seed in SEEDS:
            shifted = np.roll(noise, seed - SEEDS[0])
            if head_name == "mean_linear":
                predicted = targets + shifted
            elif head_name == "mean_layer_linear":
                predicted = targets + shifted * 0.15
            elif head_name == "mean_rich_stats_residual":
                predicted = targets + shifted
            else:
                predicted = targets + shifted * 1.5
            checkpoint = next(
                item
                for item in checkpoint_runs
                if item["head_name"] == head_name and item["seed"] == seed
            )
            for subject_id, target, prediction in zip(primary, targets, predicted):
                relative = Path("predictions") / head_name / f"seed-{seed}" / f"{subject_id}.json"
                path = prediction_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                body = {
                    "schema_version": 1,
                    "study_id": protocol.study_id,
                    "lock_sha256": lock["lock_sha256"],
                    "protocol_sha256": protocol.sha256,
                    "source_tree_sha256": "c" * 64,
                    "environment_sha256": _sha256_file(environment_path),
                    "mipdb_manifest_sha256": _sha256_file(manifest_path),
                    "preprocessing_sha256": "f" * 64,
                    "encoder_checkpoint_sha256": "d" * 64,
                    "head_checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "head_name": head_name,
                    "seed": seed,
                    "subject_id": subject_id,
                    "target_age": float(target),
                    "representation_cache_key": hashlib.sha256(
                        subject_id.encode()
                    ).hexdigest(),
                    "qc_status": "passed",
                    "qc_sha256": hashlib.sha256(f"qc:{subject_id}".encode()).hexdigest(),
                    "prediction": float(prediction),
                }
                record = {**body, "prediction_sha256": canonical_sha256(body)}
                path.write_text(json.dumps(record, sort_keys=True) + "\n")
                entries.append(
                    {
                        "head_name": head_name,
                        "seed": seed,
                        "subject_id": subject_id,
                        "path": relative.as_posix(),
                        "prediction_sha256": record["prediction_sha256"],
                    }
                )
    inventory_body = {
        "schema_version": 1,
        "status": "complete",
        "study_id": protocol.study_id,
        "lock_sha256": lock["lock_sha256"],
        "heads": list(APPROVED_HEADS),
        "seeds": list(SEEDS),
        "subjects": primary,
        "prediction_count": len(entries),
        "predictions": entries,
    }
    inventory = {
        **inventory_body,
        "prediction_inventory_sha256": canonical_sha256(inventory_body),
    }
    inventory_path = prediction_root / "prediction_inventory.json"
    inventory_path.write_text(json.dumps(inventory, sort_keys=True) + "\n")
    completion_body = {
        "schema_version": 1,
        "study_id": protocol.study_id,
        "lock_sha256": lock["lock_sha256"],
        "prediction_inventory_sha256": inventory["prediction_inventory_sha256"],
        "status": "complete",
    }
    completion = {
        **completion_body,
        "marker_sha256": canonical_sha256(completion_body),
    }
    (prediction_root / "evaluation_completed.json").write_text(
        json.dumps(completion, sort_keys=True) + "\n"
    )
    transition_study(lock_path, "completed")
    return {
        "protocol_path": protocol_path,
        "lock_path": lock_path,
        "checkpoint_root": checkpoint_root,
        "checkpoint_inventory_path": checkpoint_inventory_path,
        "mipdb_manifest_path": manifest_path,
        "prediction_root": prediction_root,
        "analysis_output_root": tmp_path / "analysis",
    }


def test_completed_study_analysis_reports_all_predeclared_evidence(
    tmp_path: Path,
) -> None:
    paths = _write_completed_study(tmp_path)

    report = analyze_confirmatory_study(**paths)

    assert report["status"] == "complete"
    assert report["cohort"]["primary_subjects"] == 6
    assert report["cohort"]["underpowered"] is True
    assert report["cohort"]["power_warning"]
    assert report["exclusions"] == [
        {"subject_id": "sub-excluded", "reason": "missing_age"}
    ]
    assert set(report["comparisons"]) == set(APPROVED_HEADS[1:])
    improved = report["comparisons"]["mean_layer_linear"]
    assert improved["paired"]["wins"] == 10
    assert improved["randomization"]["permutations"] == 1024
    assert improved["bootstrap"]["iterations"] == 10_000
    assert improved["decision"]["established_stable_improvement"] is False
    assert improved["decision"]["conditions"]["adequate_primary_cohort"] is False
    assert report["established_heads"] == []
    assert "underpowered" in report["conclusion"].casefold()
    assert report["resources"]["mean_linear"]["head_parameter_count"] == 10
    assert report["analysis_sha256"]

    first_bytes = (
        paths["analysis_output_root"] / "confirmatory_analysis.json"
    ).read_bytes()
    resumed = analyze_confirmatory_study(**paths)
    assert resumed == report
    assert (
        paths["analysis_output_root"] / "confirmatory_analysis.json"
    ).read_bytes() == first_bytes


def test_analysis_rejects_incomplete_prediction_inventory(tmp_path: Path) -> None:
    paths = _write_completed_study(tmp_path)
    missing = next(paths["prediction_root"].glob("predictions/*/seed-*/*.json"))
    missing.unlink()

    with pytest.raises(ConfirmatoryAnalysisError, match="incomplete"):
        analyze_confirmatory_study(**paths)


def test_analysis_rejects_rehashed_prediction_provenance_drift(tmp_path: Path) -> None:
    paths = _write_completed_study(tmp_path)
    inventory_path = paths["prediction_root"] / "prediction_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    entry = inventory["predictions"][0]
    prediction_path = paths["prediction_root"] / entry["path"]
    prediction = json.loads(prediction_path.read_text())
    prediction["environment_sha256"] = "0" * 64
    body = {key: value for key, value in prediction.items() if key != "prediction_sha256"}
    prediction["prediction_sha256"] = canonical_sha256(body)
    prediction_path.write_text(json.dumps(prediction, sort_keys=True) + "\n")
    entry["prediction_sha256"] = prediction["prediction_sha256"]
    inventory_body = {
        key: value
        for key, value in inventory.items()
        if key != "prediction_inventory_sha256"
    }
    inventory["prediction_inventory_sha256"] = canonical_sha256(inventory_body)
    inventory_path.write_text(json.dumps(inventory, sort_keys=True) + "\n")
    completion_path = paths["prediction_root"] / "evaluation_completed.json"
    completion = json.loads(completion_path.read_text())
    completion["prediction_inventory_sha256"] = inventory[
        "prediction_inventory_sha256"
    ]
    completion_body = {
        key: value for key, value in completion.items() if key != "marker_sha256"
    }
    completion["marker_sha256"] = canonical_sha256(completion_body)
    completion_path.write_text(json.dumps(completion, sort_keys=True) + "\n")

    with pytest.raises(ConfirmatoryAnalysisError, match="provenance"):
        analyze_confirmatory_study(**paths)


def test_confirmatory_cli_has_no_inferential_overrides() -> None:
    script_path = ROOT / "scripts/analyze_confirmatory.py"
    spec = importlib.util.spec_from_file_location("analyze_confirmatory", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = script_path.read_text(encoding="utf-8")
    for forbidden in (
        '--bootstrap-iterations',
        '--bootstrap-seed',
        '--alpha',
        '--confidence',
        '--candidate',
    ):
        assert forbidden not in source


def test_confirmatory_cli_maps_only_artifact_paths(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    script_path = ROOT / "scripts/analyze_confirmatory.py"
    spec = importlib.util.spec_from_file_location("analyze_confirmatory_main", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {}

    def analyze(**kwargs):
        captured.update(kwargs)
        return {
            "status": "complete",
            "analysis_sha256": "a" * 64,
            "established_heads": [],
            "conclusion": confirmatory_conclusion([]),
        }

    monkeypatch.setattr(module, "analyze_confirmatory_study", analyze)
    arguments = [
        "--protocol",
        str(tmp_path / "protocol.json"),
        "--lock",
        str(tmp_path / "lock.json"),
        "--checkpoint-root",
        str(tmp_path / "checkpoints"),
        "--checkpoint-inventory",
        str(tmp_path / "checkpoint_inventory.json"),
        "--mipdb-manifest",
        str(tmp_path / "mipdb_manifest.json"),
        "--prediction-root",
        str(tmp_path / "predictions"),
        "--output-root",
        str(tmp_path / "analysis"),
    ]

    assert module.main(arguments) == 0
    assert set(captured) == {
        "protocol_path",
        "lock_path",
        "checkpoint_root",
        "checkpoint_inventory_path",
        "mipdb_manifest_path",
        "prediction_root",
        "analysis_output_root",
    }
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
