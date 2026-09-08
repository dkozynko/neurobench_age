from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import neurobench_age.pipelines.external_holdout as external_holdout_module
from neurobench_age.pipelines.external_holdout import (
    ExternalHoldoutError,
    ExternalSubjectMaterial,
    RuntimeProvenance,
    load_cached_external_material,
    run_external_holdout,
)
from neurobench_age.pipelines.frozen_probe import (
    RepresentationCacheIdentity,
    write_cached_representations,
)
from neurobench_age.pipelines.frozen_probe_training import (
    APPROVED_HEADS,
    build_frozen_probe_head,
)
from neurobench_age.research.study_lock import (
    canonical_sha256,
    seal_study,
    transition_study,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_external_holdout_script():
    script_path = ROOT / "scripts/run_external_holdout.py"
    spec = importlib.util.spec_from_file_location("run_external_holdout", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_checkpoints(root: Path) -> tuple[Path, dict[str, object]]:
    runs = []
    for head_name in APPROVED_HEADS:
        for seed in range(33, 43):
            torch.manual_seed(seed)
            head = build_frozen_probe_head(head_name, embed_dim=2)
            run_dir = root / head_name / f"seed-{seed}"
            run_dir.mkdir(parents=True)
            checkpoint_path = run_dir / "head_checkpoint.pt"
            torch.save(
                {
                    "schema_version": 3,
                    "state_dict": head.state_dict(),
                    "head_name": head_name,
                    "seed": seed,
                    "selected_epoch": 1,
                    "representation_protocol_sha256": "a" * 64,
                    "training_protocol_sha256": "9" * 64,
                    "run_identity_sha256": "1" * 64,
                    "training_source_sha256": "c" * 64,
                },
                checkpoint_path,
            )
            runs.append(
                {
                    "head_name": head_name,
                    "seed": seed,
                    "representation_protocol_sha256": "a" * 64,
                    "training_protocol_sha256": "9" * 64,
                    "training_source_sha256": "c" * 64,
                    "run_identity_sha256": "1" * 64,
                    "run_manifest_sha256": "2" * 64,
                    "checkpoint_sha256": _sha256_file(checkpoint_path),
                    "selected_epoch": 1,
                    "head_parameter_count": sum(
                        parameter.numel() for parameter in head.parameters()
                    ),
                }
            )
    body = {
        "schema_version": 3,
        "status": "complete",
        "representation_protocol_sha256": "a" * 64,
        "training_protocol_sha256": "9" * 64,
        "training_source_sha256": "c" * 64,
        "heads": list(APPROVED_HEADS),
        "seeds": list(range(33, 43)),
        "run_count": 40,
        "runs": runs,
    }
    inventory = {
        **body,
        "checkpoint_inventory_sha256": canonical_sha256(body),
    }
    path = root / "checkpoint_inventory.json"
    path.write_text(json.dumps(inventory) + "\n")
    return path, inventory


def _write_mipdb_manifest(
    path: Path, *, contaminate_pilot: bool = False, status: str = "finalized"
) -> dict[str, object]:
    pilot = [f"sub-pilot-{index:02d}" for index in range(10)]
    primary = ["sub-101", "sub-102"]
    if contaminate_pilot:
        primary[0] = pilot[0]
    subjects = [
        {"subject_id": subject_id, "age": float(6 + index), "recordings": ["rest.set"]}
        for index, subject_id in enumerate([*pilot, "sub-101", "sub-102"])
    ]
    manifest = {
        "schema_version": 2,
        "status": status,
        "dataset": "MIPDB",
        "protocol_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "subjects": subjects,
        "cohorts": {"pilot": pilot, "primary": primary, "extrapolation": []},
        "subject_list_sha256": {
            "pilot": _sha256_json(pilot),
            "primary": _sha256_json(primary),
            "extrapolation": _sha256_json([]),
        },
        "underpowered": len(primary) < 50,
        "minimum_primary_subjects": 50,
        "cohort_qc_sha256": "7" * 64,
    }
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return manifest


def _fixture(
    tmp_path: Path,
    *,
    contaminate_pilot: bool = False,
    manifest_status: str = "finalized",
) -> dict[str, object]:
    checkpoint_root = tmp_path / "checkpoints"
    inventory_path, inventory = _write_checkpoints(checkpoint_root)
    mipdb_manifest_path = tmp_path / "mipdb_manifest.json"
    manifest = _write_mipdb_manifest(
        mipdb_manifest_path,
        contaminate_pilot=contaminate_pilot,
        status=manifest_status,
    )
    environment_path = tmp_path / "environment.lock"
    environment_path.write_text("synthetic-environment\n")
    output_root = tmp_path / "external-output"
    lock_path = tmp_path / "study_lock.json"
    payload = {
        "study_id": "reve_age_external_frozen_probe_v1",
        "protocol_sha256": "a" * 64,
        "training_protocol_sha256": "9" * 64,
        "representation_source_sha256": "8" * 64,
        "training_source_sha256": "c" * 64,
        "git_revision": "synthetic-revision",
        "git_dirty": False,
        "encoder_checkpoint": "brain-bzh/reve-base",
        "encoder_checkpoint_sha256": "d" * 64,
        "checkpoint_inventory_sha256": inventory[
            "checkpoint_inventory_sha256"
        ],
        "environment_sha256": _sha256_file(environment_path),
        "hbn_manifest_sha256": "e" * 64,
        "hbn_training_manifest_sha256": "5" * 64,
        "mipdb_manifest_sha256": _sha256_file(mipdb_manifest_path),
        "mipdb_pilot_qc_sha256": "6" * 64,
        "mipdb_cohort_qc_sha256": "7" * 64,
        "subject_list_sha256": {
            "hbn_train": "1" * 64,
            "hbn_validation": "2" * 64,
            "mipdb_pilot": manifest["subject_list_sha256"]["pilot"],
            "mipdb_primary": manifest["subject_list_sha256"]["primary"],
            "mipdb_extrapolation": manifest["subject_list_sha256"]["extrapolation"],
        },
        "heads": list(APPROVED_HEADS),
        "seeds": list(range(33, 43)),
        "preprocessing_sha256": "f" * 64,
        "statistics_sha256": "3" * 64,
        "output_root": str(output_root),
    }
    seal_study(lock_path, payload)
    return {
        "lock_path": lock_path,
        "checkpoint_root": checkpoint_root,
        "inventory_path": inventory_path,
        "mipdb_manifest_path": mipdb_manifest_path,
        "environment_path": environment_path,
        "output_root": output_root,
        "runtime": RuntimeProvenance(
            training_source_sha256="c" * 64,
            git_revision="synthetic-revision",
            git_dirty=False,
            environment_sha256=_sha256_file(environment_path),
        ),
    }


def _material(subject_id, identity) -> ExternalSubjectMaterial:
    value = 1.6 if subject_id == "sub-101" else 1.7
    final = torch.full((2, 3, 2), value)
    return ExternalSubjectMaterial(
        representations={-2: final + 0.25, -1: final},
        cache_identity=identity,
        qc={"status": "passed", "subject_id": subject_id, "window_count": 2},
    )


def test_external_module_has_no_training_or_analysis_capability() -> None:
    source = (ROOT / "src/neurobench_age/pipelines/external_holdout.py").read_text()
    for forbidden in (
        "torch.optim",
        ".backward(",
        "scheduler",
        "calibration",
        "pearson",
        "mean_squared_error",
        "model_selection",
    ):
        assert forbidden not in source


def test_holdout_starts_before_provider_and_completes_exact_inventory(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    calls = []

    def provider(subject_id, identity):
        state = json.loads((tmp_path / "study_state.json").read_text())
        assert state["state"] == "started"
        assert (fixture["output_root"] / "evaluation_started.json").is_file()
        assert identity.dataset_manifest_sha256 == "b" * 64
        calls.append(subject_id)
        return _material(subject_id, identity)

    completion = run_external_holdout(
        **fixture,
        representation_provider=provider,
        device="cpu",
    )

    assert calls == ["sub-101", "sub-102"]
    assert completion["status"] == "complete"
    assert completion["prediction_count"] == 80
    assert not any("metric" in key for key in completion)
    assert not any("metric" in path.name for path in fixture["output_root"].rglob("*"))
    state = json.loads((tmp_path / "study_state.json").read_text())
    assert state["state"] == "completed"
    prediction_path = (
        fixture["output_root"]
        / "predictions/mean_linear/seed-33/sub-101.json"
    )
    prediction = json.loads(prediction_path.read_text())
    assert prediction["subject_id"] == "sub-101"
    assert prediction["target_age"] == 16.0
    assert prediction["qc_status"] == "passed"
    assert prediction["head_name"] == "mean_linear"
    assert prediction["seed"] == 33
    assert len(prediction["prediction_sha256"]) == 64


def test_interrupted_holdout_resumes_only_missing_records(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    calls = []

    def interrupted(subject_id, identity):
        calls.append(subject_id)
        if subject_id == "sub-102":
            raise RuntimeError("synthetic interruption")
        return _material(subject_id, identity)

    with pytest.raises(RuntimeError, match="interruption"):
        run_external_holdout(
            **fixture,
            representation_provider=interrupted,
            device="cpu",
        )
    first_path = (
        fixture["output_root"]
        / "predictions/mean_linear/seed-33/sub-101.json"
    )
    first_bytes = first_path.read_bytes()
    assert not (fixture["output_root"] / "prediction_inventory.json").exists()

    resumed_calls = []

    def resumed(subject_id, identity):
        resumed_calls.append(subject_id)
        return _material(subject_id, identity)

    completion = run_external_holdout(
        **fixture,
        representation_provider=resumed,
        device="cpu",
    )

    assert resumed_calls == ["sub-102"]
    assert first_path.read_bytes() == first_bytes
    assert completion["prediction_count"] == 80


def test_holdout_rejects_corrupted_existing_prediction(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def interrupted(subject_id, identity):
        if subject_id == "sub-102":
            raise RuntimeError("stop")
        return _material(subject_id, identity)

    with pytest.raises(RuntimeError):
        run_external_holdout(
            **fixture,
            representation_provider=interrupted,
            device="cpu",
        )
    prediction_path = (
        fixture["output_root"]
        / "predictions/mean_linear/seed-33/sub-101.json"
    )
    prediction = json.loads(prediction_path.read_text())
    prediction["prediction"] += 1.0
    prediction_path.write_text(json.dumps(prediction))

    with pytest.raises(ExternalHoldoutError, match="prediction hash"):
        run_external_holdout(
            **fixture,
            representation_provider=_material,
            device="cpu",
        )


def test_holdout_rejects_unsealed_or_drifted_inputs_before_provider(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    state_path = tmp_path / "study_state.json"
    state = json.loads(state_path.read_text())
    state["state"] = "draft"
    state_path.write_text(json.dumps(state))
    called = False

    def provider(subject_id, identity):
        nonlocal called
        called = True
        return _material(subject_id, identity)

    with pytest.raises(ExternalHoldoutError, match="sealed or started"):
        run_external_holdout(
            **fixture,
            representation_provider=provider,
            device="cpu",
        )
    assert called is False


def test_holdout_rejects_draft_mipdb_manifest_before_provider(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, manifest_status="draft")
    called = False

    def provider(subject_id, identity):
        nonlocal called
        called = True
        return _material(subject_id, identity)

    with pytest.raises(ExternalHoldoutError, match="finalized"):
        run_external_holdout(
            **fixture,
            representation_provider=provider,
            device="cpu",
        )
    assert called is False


@pytest.mark.parametrize("drift", ["checkpoint", "subjects", "seeds"])
def test_holdout_preflight_rejects_inventory_drift(
    tmp_path: Path, drift: str
) -> None:
    fixture = _fixture(tmp_path)
    if drift == "checkpoint":
        checkpoint = (
            fixture["checkpoint_root"]
            / "mean_linear/seed-33/head_checkpoint.pt"
        )
        with checkpoint.open("ab") as handle:
            handle.write(b"tampered")
    elif drift == "subjects":
        manifest_path = fixture["mipdb_manifest_path"]
        manifest = json.loads(manifest_path.read_text())
        manifest["cohorts"]["primary"].reverse()
        manifest_path.write_text(json.dumps(manifest))
    else:
        inventory_path = fixture["inventory_path"]
        inventory = json.loads(inventory_path.read_text())
        inventory["seeds"] = list(range(33, 42))
        body = {
            key: value
            for key, value in inventory.items()
            if key != "checkpoint_inventory_sha256"
        }
        inventory["checkpoint_inventory_sha256"] = canonical_sha256(body)
        inventory_path.write_text(json.dumps(inventory))

    with pytest.raises((ExternalHoldoutError, RuntimeError)):
        run_external_holdout(
            **fixture,
            representation_provider=_material,
            device="cpu",
        )


def test_holdout_rejects_pilot_contamination(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, contaminate_pilot=True)

    with pytest.raises(ExternalHoldoutError, match="pilot"):
        run_external_holdout(
            **fixture,
            representation_provider=_material,
            device="cpu",
        )


def _write_external_cache(
    cache_root: Path,
    *,
    subject_id: str,
    age: float,
) -> RepresentationCacheIdentity:
    identity = RepresentationCacheIdentity(
        protocol_sha256="a" * 64,
        checkpoint="brain-bzh/reve-base",
        checkpoint_sha256="d" * 64,
        dataset_manifest_sha256="b" * 64,
        preprocessing_sha256="f" * 64,
        subject_id=subject_id,
        source_tree_sha256="c" * 64,
    )
    representation = torch.full((2, 3, 2), age / 10.0)
    write_cached_representations(
        cache_root,
        identity,
        {-2: representation + 0.25, -1: representation},
        evidence={
            "encoder_frozen": True,
            "encoder_eval_mode": True,
            "inference_mode": True,
            "layer_indices": [-2, -1],
            "state_sha256_before": "9" * 64,
            "state_sha256_after": "9" * 64,
            "external_qc": {
                "status": "passed",
                "subject_id": subject_id,
                "window_count": 2,
            },
        },
    )
    return identity


def test_cached_external_material_requires_embedded_matching_qc(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    identity = _write_external_cache(
        cache_root, subject_id="sub-101", age=16.0
    )
    material = load_cached_external_material(cache_root, "sub-101", identity)

    assert material.cache_identity == identity
    assert material.qc["status"] == "passed"
    assert material.representations[-1].shape == (2, 3, 2)


def test_external_holdout_cli_runs_only_the_locked_primary_inventory(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    fixture = _fixture(tmp_path)
    cache_root = tmp_path / "cache"
    for subject_id, age in (("sub-101", 16.0), ("sub-102", 17.0)):
        _write_external_cache(cache_root, subject_id=subject_id, age=age)
    script_path = ROOT / "scripts/run_external_holdout.py"
    module = _load_external_holdout_script()
    monkeypatch.setattr(
        module,
        "load_study_protocol",
        lambda path: SimpleNamespace(
            sha256="a" * 64,
            datasets=SimpleNamespace(minimum_primary_subjects=50),
        ),
    )
    monkeypatch.setattr(
        module,
        "load_frozen_probe_training_protocol",
        lambda path: SimpleNamespace(
            sha256="9" * 64,
            representation_protocol_sha256="a" * 64,
        ),
    )
    monkeypatch.setattr(
        module, "_runtime_provenance", lambda *args: fixture["runtime"]
    )
    provider_arguments = {}

    def provider_factory(**kwargs):
        provider_arguments.update(kwargs)
        return lambda subject_id, identity: load_cached_external_material(
            cache_root, subject_id, identity
        )

    monkeypatch.setattr(module, "LazyMipdbRepresentationProvider", provider_factory)

    result = module.main(
        [
            "--protocol",
            str(tmp_path / "protocol.json"),
            "--training-protocol",
            str(tmp_path / "training-protocol.json"),
            "--lock",
            str(fixture["lock_path"]),
            "--checkpoint-root",
            str(fixture["checkpoint_root"]),
            "--checkpoint-inventory",
            str(fixture["inventory_path"]),
            "--mipdb-manifest",
            str(fixture["mipdb_manifest_path"]),
            "--environment",
            str(fixture["environment_path"]),
            "--cache-root",
            str(cache_root),
            "--bids-root",
            str((tmp_path / "mipdb-bids").resolve()),
            "--mapping",
            str((tmp_path / "reve.json").resolve()),
            "--output-root",
            str(fixture["output_root"]),
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["prediction_count"] == 80
    assert summary["status"] == "complete"
    assert summary["underpowered"] is True
    assert provider_arguments["started_marker_path"] == (
        fixture["output_root"] / "evaluation_started.json"
    )
    assert provider_arguments["expected_lock_sha256"]
    source = script_path.read_text(encoding="utf-8")
    assert "LazyMipdbRepresentationProvider" in source
    assert 'add_argument("--seed' not in source
    assert 'add_argument("--head' not in source
    assert 'add_argument("--subject' not in source


def test_external_holdout_cli_rejects_malformed_primary_cohort(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"cohorts": {"primary": "sub-not-an-array"}})
    )
    module = _load_external_holdout_script()

    with pytest.raises(ExternalHoldoutError, match="ordered array"):
        module._validate_protocol_and_power(
            protocol=SimpleNamespace(
                sha256="a" * 64,
                datasets=SimpleNamespace(minimum_primary_subjects=50),
            ),
            training_protocol=SimpleNamespace(
                sha256="9" * 64,
                representation_protocol_sha256="a" * 64,
            ),
            lock={
                "protocol_sha256": "a" * 64,
                "training_protocol_sha256": "9" * 64,
            },
            manifest_path=manifest_path,
        )


def test_external_cli_records_resumable_failure_after_started_transition(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)
    module = _load_external_holdout_script()
    monkeypatch.setattr(
        module,
        "load_study_protocol",
        lambda path: SimpleNamespace(
            sha256="a" * 64,
            datasets=SimpleNamespace(minimum_primary_subjects=50),
        ),
    )
    monkeypatch.setattr(
        module,
        "load_frozen_probe_training_protocol",
        lambda path: SimpleNamespace(
            sha256="9" * 64,
            representation_protocol_sha256="a" * 64,
        ),
    )
    monkeypatch.setattr(
        module, "_runtime_provenance", lambda *args: fixture["runtime"]
    )

    def provider_factory(**kwargs):
        def fail(subject_id, identity):
            raise RuntimeError("synthetic extraction interruption")

        return fail

    monkeypatch.setattr(module, "LazyMipdbRepresentationProvider", provider_factory)

    with pytest.raises(SystemExit):
        module.main(
            [
                "--protocol", str(tmp_path / "protocol.json"),
                "--training-protocol", str(tmp_path / "training-protocol.json"),
                "--lock", str(fixture["lock_path"]),
                "--checkpoint-root", str(fixture["checkpoint_root"]),
                "--checkpoint-inventory", str(fixture["inventory_path"]),
                "--mipdb-manifest", str(fixture["mipdb_manifest_path"]),
                "--environment", str(fixture["environment_path"]),
                "--cache-root", str((tmp_path / "cache").resolve()),
                "--bids-root", str((tmp_path / "bids").resolve()),
                "--mapping", str((tmp_path / "mapping.json").resolve()),
                "--output-root", str(fixture["output_root"]),
                "--device", "cpu",
            ]
        )

    evidence = sorted((tmp_path / "study_failures").glob("*.json"))
    assert len(evidence) == 1
    assert "synthetic extraction interruption" in evidence[0].read_text()
    assert json.loads((tmp_path / "study_state.json").read_text())["state"] == "started"


def test_completion_resumes_after_marker_to_state_transition_interruption(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)
    transition_study(fixture["lock_path"], "started")
    original_transition = external_holdout_module.transition_study

    def interrupt_completion(lock_path, target):
        assert target == "completed"
        raise RuntimeError("synthetic completion interruption")

    monkeypatch.setattr(
        external_holdout_module, "transition_study", interrupt_completion
    )
    with pytest.raises(RuntimeError, match="completion interruption"):
        run_external_holdout(
            **fixture,
            representation_provider=_material,
            device="cpu",
        )
    assert (fixture["output_root"] / "evaluation_completed.json").is_file()
    state = json.loads((tmp_path / "study_state.json").read_text())
    assert state["state"] == "started"

    monkeypatch.setattr(
        external_holdout_module, "transition_study", original_transition
    )

    def forbidden_provider(subject_id, identity):
        raise AssertionError("completed predictions must not be recomputed")

    inventory = run_external_holdout(
        **fixture,
        representation_provider=forbidden_provider,
        device="cpu",
    )

    assert inventory["prediction_count"] == 80
    state = json.loads((tmp_path / "study_state.json").read_text())
    assert state["state"] == "completed"
