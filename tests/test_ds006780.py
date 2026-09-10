from __future__ import annotations

import json
from pathlib import Path
import shutil
import types

import numpy as np
import pytest
from jsonschema import ValidationError

from neurobench_age.data.ds006780 import (
    Ds006780Error,
    apply_reference_policy,
    build_target_free_qc,
    build_target_free_manifest,
    discover_runs,
    load_preprocessing_contract,
    parse_channel_inventory,
    parse_event_inventory,
    verify_ds006780_manifest,
    validate_external_config,
    write_manifest,
    write_qc_report,
)
from neurobench_age.data.ds006780_cohort import (
    Ds006780CohortError,
    finalize_ds006780_cohort,
    write_target_manifest,
)
from neurobench_age.research.strict_json import (
    StrictJsonError,
    canonical_json_bytes,
    canonical_sha256,
    load_json_strict,
    reject_target_fields,
    validate_schema,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "research" / "ds006780_external_transfer.json"
MANIFEST_SCHEMA = ROOT / "schemas" / "research" / "ds006780_target_free_manifest.schema.json"
QC_SCHEMA = ROOT / "schemas" / "research" / "ds006780_qc.schema.json"
TARGET_SCHEMA = ROOT / "schemas" / "research" / "ds006780_target_manifest.schema.json"


class _FakeRaw:
    def __init__(self, *, sfreq: float = 512.0, n_times: int = 1024) -> None:
        self.info = {"sfreq": sfreq}
        self.n_times = n_times
        mne = pytest.importorskip("mne")
        self.ch_names = list(mne.channels.make_standard_montage("biosemi64").ch_names)
        self.ch_names.extend(f"EXG{index}" for index in range(1, 9))
        self.ch_names.append("Status")


class _SignalRaw(_FakeRaw):
    def __init__(self, *, sfreq: float = 512.0, n_times: int = 1024) -> None:
        super().__init__(sfreq=sfreq, n_times=n_times)
        rng = np.random.default_rng(123)
        self._data = rng.normal(size=(len(self.ch_names), n_times))
        self.reference_calls: list[object] = []

    def get_data(self) -> np.ndarray:
        return self._data

    def set_eeg_reference(self, ref_channels: object, *, verbose: str) -> None:
        self.reference_calls.append(ref_channels)
        self._data[:64] -= self._data[:64].mean(axis=0, keepdims=True)


def _fake_mne() -> types.SimpleNamespace:
    mne = pytest.importorskip("mne")
    montage = mne.channels.make_standard_montage("biosemi64")
    return types.SimpleNamespace(
        __version__=mne.__version__,
        channels=types.SimpleNamespace(
            make_standard_montage=lambda name: montage,
        ),
    )


def _write_ds006780_tree(tmp_path: Path, *, sfreq: float = 512.0) -> tuple[Path, Path]:
    root = tmp_path / "ds006780"
    eeg = root / "sub-001" / "eeg"
    eeg.mkdir(parents=True)
    (root / "dataset_description.json").write_text("{}", encoding="utf-8")
    (root / "README").write_text("synthetic ds006780", encoding="utf-8")
    (root / "participants.tsv").write_text("participant_id\tage\nsub-001\t12\n", encoding="utf-8")
    stem = eeg / "sub-001_task-Restingstate_run-01"
    (stem.with_name(stem.name + "_eeg.bdf")).write_bytes(b"BDF payload\x00")
    mne = pytest.importorskip("mne")
    names = mne.channels.make_standard_montage("biosemi64").ch_names
    channel_lines = ["name\ttype\tunits\tstatus\tsampling_frequency"]
    channel_lines.extend(f"{name}\tEEG\tuV\tgood\t{sfreq}" for name in names)
    channel_lines.extend(f"EXG{index}\tEMG\tuV\tn/a\t{sfreq}" for index in range(1, 9))
    channel_lines.append(f"Status\tTRIG\tNA\tn/a\t{sfreq}")
    (stem.with_name(stem.name + "_channels.tsv")).write_text(
        "\n".join(channel_lines) + "\n", encoding="utf-8"
    )
    (stem.with_name(stem.name + "_events.tsv")).write_text(
        "onset\tduration\ttrial_type\tvalue\tsample\n"
        "0\t0\tRecording_start\tstart\t0\n",
        encoding="utf-8",
    )
    (stem.with_name(stem.name + "_eeg.json")).write_text(
        json.dumps(
            {
                "SamplingFrequency": sfreq,
                "RecordingDuration": 2.0,
                "EEGReference": "n/a",
            }
        ),
        encoding="utf-8",
    )
    return root, stem.with_name(stem.name + "_eeg.bdf")


def test_strict_json_rejects_duplicate_keys_at_any_depth() -> None:
    with pytest.raises(StrictJsonError, match="duplicate JSON key"):
        load_json_strict(b'{"outer": {"x": 1, "x": 2}}')


@pytest.mark.parametrize("payload", [b"NaN", b"Infinity", b"-Infinity"])
def test_strict_json_rejects_nonfinite_numbers(payload: bytes) -> None:
    with pytest.raises(StrictJsonError, match="non-finite"):
        load_json_strict(payload)


def test_rfc8785_canonical_bytes_are_key_order_independent() -> None:
    left = {"b": 1.0, "a": -0.0, "nested": [{"z": True, "y": None}]}
    right = {"nested": [{"y": None, "z": True}], "a": 0, "b": 1}

    assert canonical_json_bytes(left) == b'{"a":0,"b":1,"nested":[{"y":null,"z":true}]}'
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_sha256(left) == canonical_sha256(right)


def test_canonical_sha256_can_exclude_self_hash_field() -> None:
    body = {"value": 3, "artifact_sha256": "placeholder"}
    assert canonical_sha256(body, exclude_fields=("artifact_sha256",)) == canonical_sha256(
        {"value": 3}
    )


def test_schema_validation_rejects_unknown_and_missing_fields(tmp_path: Path) -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["name"],
        "properties": {"name": {"type": "string"}},
    }
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    validate_schema({"name": "ok"}, schema_path)
    with pytest.raises(ValidationError):
        validate_schema({"name": "ok", "extra": True}, schema_path)
    with pytest.raises(ValidationError):
        validate_schema({}, schema_path)


def test_target_free_payload_rejects_forbidden_keys_recursively() -> None:
    with pytest.raises(StrictJsonError, match="target-bearing field"):
        reject_target_fields({"runs": [{"qc": {"age_years": 12.0}}]})


def test_discovery_orders_primary_and_records_out_of_scope_runs(tmp_path: Path) -> None:
    root, recording = _write_ds006780_tree(tmp_path)
    other = root / "sub-001" / "eeg" / "sub-001_task-Restingstate_run-02_eeg.bdf"
    other.write_bytes(b"other")
    session = root / "sub-001" / "ses-02" / "eeg"
    session.mkdir(parents=True)
    (session / "sub-001_ses-02_task-Restingstate_run-01_eeg.bdf").write_bytes(b"session")

    candidates, out_of_scope = discover_runs(root)

    assert [candidate.recording_path for candidate in candidates] == [
        recording.relative_to(root).as_posix()
    ]
    assert {item["exclusion_code"] for item in out_of_scope} == {
        "non_primary_run",
        "session_variant",
    }


def test_manifest_contains_provenance_and_verifier_detects_file_drift(tmp_path: Path) -> None:
    root, _ = _write_ds006780_tree(tmp_path)
    fake_mne = _fake_mne()
    project_root = tmp_path / "project"
    project_root.mkdir()
    shutil.copy2(ROOT / "uv.lock", project_root / "uv.lock")
    manifest = build_target_free_manifest(
        root,
        config_path=CONFIG_PATH,
        project_root=project_root,
        source_commit="799d1502296ba5f74033734e149160c3d333e470",
        openneuro_version="1.0.0",
        raw_loader=lambda path: _FakeRaw(),
        mne_module=fake_mne,
    )
    output = tmp_path / "manifest.json"
    written = write_manifest(output, manifest, schema_path=MANIFEST_SCHEMA)
    verify_ds006780_manifest(
        root,
        written,
        config_path=CONFIG_PATH,
        project_root=project_root,
    )

    mutated_config = tmp_path / "mutated_config.json"
    mutated = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mutated["reference"]["primary_policy"] = "average_reference"
    mutated_config.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(Ds006780Error, match="protocol_sha256"):
        verify_ds006780_manifest(
            root,
            written,
            config_path=mutated_config,
            project_root=project_root,
        )

    (project_root / "uv.lock").write_text(
        (project_root / "uv.lock").read_text(encoding="utf-8") + "\n# drift\n",
        encoding="utf-8",
    )
    with pytest.raises(Ds006780Error, match="environment identity"):
        verify_ds006780_manifest(
            root,
            written,
            config_path=CONFIG_PATH,
            project_root=project_root,
        )

    shutil.copy2(ROOT / "uv.lock", project_root / "uv.lock")
    recording = root / written["candidate_runs"][0]["recording_path"]
    recording.write_bytes(recording.read_bytes() + b"drift")
    with pytest.raises(Ds006780Error, match="identity drift"):
        verify_ds006780_manifest(
            root,
            written,
            config_path=CONFIG_PATH,
            project_root=project_root,
        )


def test_manifest_rejects_unresolved_annex_pointer_and_wrong_source_frequency(
    tmp_path: Path,
) -> None:
    root, recording = _write_ds006780_tree(tmp_path)
    recording.write_text(
        "version https://git-annex.branchable.com/\nkey SHA256E-s12--pointer\n",
        encoding="utf-8",
    )
    with pytest.raises(Ds006780Error, match="unresolved_git_annex_pointer"):
        build_target_free_manifest(
            root,
            config_path=CONFIG_PATH,
            project_root=ROOT,
            source_commit="799d1502296ba5f74033734e149160c3d333e470",
            openneuro_version="1.0.0",
            raw_loader=lambda path: _FakeRaw(),
            mne_module=_fake_mne(),
        )

    root, _ = _write_ds006780_tree(tmp_path / "wrong-frequency")
    with pytest.raises(Ds006780Error, match="unsupported_sampling_frequency"):
        build_target_free_manifest(
            root,
            config_path=CONFIG_PATH,
            project_root=ROOT,
            source_commit="799d1502296ba5f74033734e149160c3d333e470",
            openneuro_version="1.0.0",
            raw_loader=lambda path: _FakeRaw(sfreq=256.0),
            mne_module=_fake_mne(),
        )


def test_event_parser_enforces_boundaries_and_sample_consistency(tmp_path: Path) -> None:
    valid = tmp_path / "valid.tsv"
    valid.write_text(
        "onset\tduration\ttrial_type\tvalue\tsample\n"
        "1\t0\tRecording_start\tstart\t512\n"
        "1.5\t0\tRecording_end\tend\t768\n",
        encoding="utf-8",
    )
    events = parse_event_inventory(valid, sampling_frequency_hz=512.0, n_times=1024)
    assert events[0]["sample"] == 512

    invalid = tmp_path / "invalid.tsv"
    invalid.write_text(
        "onset\tduration\ttrial_type\tvalue\tsample\n"
        "1\t0\tRecording_start\tstart\t600\n",
        encoding="utf-8",
    )
    with pytest.raises(Ds006780Error, match="sample/onset mismatch"):
        parse_event_inventory(invalid, sampling_frequency_hz=512.0, n_times=1024)


def test_preserve_reference_is_an_exact_noop() -> None:
    raw = _SignalRaw()
    before = raw.get_data().copy()

    returned, provenance, operation = apply_reference_policy(
        raw,
        policy="preserve_acquisition",
        source_reference_value="n/a",
    )

    assert returned is raw
    assert provenance == "unknown"
    assert operation == "none"
    assert raw.reference_calls == []
    np.testing.assert_array_equal(before, raw.get_data())


def test_qc_report_contains_only_target_free_signal_fields(tmp_path: Path) -> None:
    root, _ = _write_ds006780_tree(tmp_path)
    config = load_json_strict(CONFIG_PATH)
    fake_mne = _fake_mne()
    raw = _SignalRaw()
    windows, qc = build_target_free_qc(
        root,
        {
            "subject_id": "sub-001",
            "run_id": "run-01",
            "recording_path": "sub-001/eeg/sub-001_task-Restingstate_run-01_eeg.bdf",
            "channels_path": "sub-001/eeg/sub-001_task-Restingstate_run-01_channels.tsv",
            "events_path": "sub-001/eeg/sub-001_task-Restingstate_run-01_events.tsv",
            "sidecar_path": "sub-001/eeg/sub-001_task-Restingstate_run-01_eeg.json",
        },
        manifest_sha256="a" * 64,
        config=config,
        raw_loader=lambda path: raw,
        mne_module=fake_mne,
    )

    assert windows.shape == (1, 64, 400)
    assert qc["reference_policy"] == "preserve_acquisition"
    assert qc["reference_provenance"] == "unknown"
    assert qc["reference_operation"] == "none"
    assert "age" not in qc
    written = write_qc_report(tmp_path / "qc.json", qc, schema_path=QC_SCHEMA)
    assert written["qc_sha256"] == canonical_sha256(written, exclude_fields=("qc_sha256",))


def test_average_reference_is_explicitly_applied() -> None:
    raw = _SignalRaw()
    returned, provenance, operation = apply_reference_policy(
        raw,
        policy="average_reference",
        source_reference_value="n/a",
    )

    assert returned is raw
    assert raw.reference_calls == ["average"]
    assert provenance == "unknown"
    assert operation == "average_reference_applied"


def test_precision_gate_requires_a_number_only_when_marked_approved() -> None:
    config = load_json_strict(CONFIG_PATH)
    validate_external_config(config)
    approved = json.loads(json.dumps(config))
    approved["precision_gate"]["status"] = "approved"
    approved["precision_gate"]["max_primary_confidence_interval_width"] = None
    with pytest.raises(Ds006780Error, match="numeric threshold"):
        validate_external_config(approved)
    approved["precision_gate"]["max_primary_confidence_interval_width"] = 0.2
    validate_external_config(approved, require_approved_precision=True)


def test_target_finalizer_joins_exact_subjects_and_records_source_digest(tmp_path: Path) -> None:
    participant_path = tmp_path / "participants.tsv"
    participant_path.write_text(
        "participant_id\tage\nsub-001\t12\nsub-999\t30\n",
        encoding="utf-8",
    )
    qc_body = {"subject_id": "sub-001", "signal": "finite"}
    qc = {**qc_body, "qc_sha256": canonical_sha256(qc_body)}
    target_free = {
        "manifest_sha256": "a" * 64,
        "candidate_runs": [{"subject_id": "sub-001"}],
    }

    manifest = finalize_ds006780_cohort(
        target_free,
        participant_metadata_path=participant_path,
        qc_reports={"sub-001": qc},
    )
    assert manifest["subjects"] == {
        "sub-001": {
            "signal_qc_sha256": qc["qc_sha256"],
            "age_years": 12.0,
            "age_units": "years",
            "age_support_eligible": True,
        }
    }
    written = write_target_manifest(tmp_path / "target.json", manifest, schema_path=TARGET_SCHEMA)
    from neurobench_age.data.ds006780 import sha256_file

    assert written["participant_metadata_sha256"] == sha256_file(participant_path)
    assert written["exclusions"] == []


def test_target_finalizer_records_missing_age_as_explicit_exclusion(
    tmp_path: Path,
) -> None:
    participant_path = tmp_path / "participants_missing_age.tsv"
    participant_path.write_text(
        "participant_id\tage\nsub-001\t12\nsub-002\tn/a\n",
        encoding="utf-8",
    )
    target_free = {
        "manifest_sha256": "a" * 64,
        "candidate_runs": [
            {"subject_id": "sub-001"},
            {"subject_id": "sub-002"},
        ],
    }
    qc_reports = {}
    for subject_id in ("sub-001", "sub-002"):
        body = {"subject_id": subject_id}
        qc_reports[subject_id] = {**body, "qc_sha256": canonical_sha256(body)}

    manifest = finalize_ds006780_cohort(
        target_free,
        participant_metadata_path=participant_path,
        qc_reports=qc_reports,
    )

    assert set(manifest["subjects"]) == {"sub-001"}
    assert manifest["exclusions"] == [
        {
            "subject_id": "sub-002",
            "reason": "missing_age",
            "signal_qc_sha256": qc_reports["sub-002"]["qc_sha256"],
        }
    ]


def test_target_finalizer_rejects_duplicate_normalized_ids_and_missing_matches(
    tmp_path: Path,
) -> None:
    duplicate_path = tmp_path / "duplicate.tsv"
    duplicate_path.write_text(
        "participant_id\tage\n sub-001 \t12\n\ufeffsub-001\t13\n",
        encoding="utf-8",
    )
    with pytest.raises(Ds006780CohortError, match="duplicate normalized"):
        finalize_ds006780_cohort(
            {"manifest_sha256": "a" * 64, "candidate_runs": []},
            participant_metadata_path=duplicate_path,
            qc_reports={},
        )

    participant_path = tmp_path / "missing.tsv"
    participant_path.write_text("participant_id\tage\nsub-001\t12\n", encoding="utf-8")
    body = {"subject_id": "sub-002"}
    qc = {**body, "qc_sha256": canonical_sha256(body)}
    manifest = finalize_ds006780_cohort(
        {
            "manifest_sha256": "a" * 64,
            "candidate_runs": [{"subject_id": "sub-002"}],
        },
        participant_metadata_path=participant_path,
        qc_reports={"sub-002": qc},
    )
    assert manifest["subjects"] == {}
    assert manifest["exclusions"] == [
        {
            "subject_id": "sub-002",
            "reason": "missing_participant_metadata",
            "signal_qc_sha256": qc["qc_sha256"],
        }
    ]


def test_target_finalizer_marks_support_boundaries_correctly(tmp_path: Path) -> None:
    participant_path = tmp_path / "participants.tsv"
    participant_path.write_text(
        "participant_id\tage\nsub-001\t5.06\nsub-002\t21.67\nsub-003\t4.99\nsub-004\t21.68\n",
        encoding="utf-8",
    )
    target_free = {
        "manifest_sha256": "a" * 64,
        "candidate_runs": [{"subject_id": f"sub-{index:03d}"} for index in range(1, 5)],
    }
    qc_reports = {}
    for index in range(1, 5):
        body = {"subject_id": f"sub-{index:03d}"}
        qc_reports[body["subject_id"]] = {**body, "qc_sha256": canonical_sha256(body)}
    manifest = finalize_ds006780_cohort(
        target_free,
        participant_metadata_path=participant_path,
        qc_reports=qc_reports,
    )
    assert manifest["subjects"]["sub-001"]["age_support_eligible"] is True
    assert manifest["subjects"]["sub-002"]["age_support_eligible"] is True
    assert manifest["subjects"]["sub-003"]["age_support_eligible"] is False
    assert manifest["subjects"]["sub-004"]["age_support_eligible"] is False
