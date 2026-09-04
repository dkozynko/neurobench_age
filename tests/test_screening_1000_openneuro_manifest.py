from neurobench_age.data.openneuro_manifest import candidate_from_metadata


def test_candidate_from_metadata_uses_sidecar_duration_and_official_path():
    candidate = candidate_from_metadata(
        release="R1",
        filename="sub-TEST/eeg/sub-TEST_task-RestingState_eeg.json",
        age=12.5,
        sidecar={"RecordingDuration": 321.25},
    )

    assert candidate.subject == "sub-TEST"
    assert candidate.age == 12.5
    assert candidate.duration_s == 321.25
    assert candidate.recording_relpath == (
        "R1/download/sub-TEST/eeg/"
        "sub-TEST_task-RestingState_eeg.set"
    )
