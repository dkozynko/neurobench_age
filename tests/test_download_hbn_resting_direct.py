from pathlib import Path

import neurobench_age.data.download as downloader


def test_download_all_reports_status_for_newly_completed_release(tmp_path, monkeypatch):
    remote = downloader.RemoteFile(
        filename="sub-TEST/eeg/sub-TEST_task-RestingState_eeg.set",
        url="https://example.invalid/file",
        size=1,
    )

    monkeypatch.setitem(downloader.RELEASE_TO_DATASET, "R1", "ds-test")
    monkeypatch.setattr(
        downloader,
        "_metadata",
        lambda dataset, **kwargs: ("snapshot", [remote]),
    )

    def fake_download_one(remote_file, destination: Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"x")
        return "downloaded"

    monkeypatch.setattr(downloader, "_download_one", fake_download_one)

    result = downloader.download_all(tmp_path, ["R1"], workers=1)

    assert result["releases"][0]["status"] == "completed"
    assert (tmp_path / "direct_download_provenance.json").is_file()


def test_manifest_filter_keeps_only_selected_resting_subjects():
    assert downloader._include_for_subjects("participants.tsv", {"sub-KEEP"})
    assert downloader._include_for_subjects(
        "sub-KEEP/eeg/sub-KEEP_task-RestingState_eeg.set", {"sub-KEEP"}
    )
    assert not downloader._include_for_subjects(
        "sub-DROP/eeg/sub-DROP_task-RestingState_eeg.set", {"sub-KEEP"}
    )
    assert not downloader._include_for_subjects(
        "sub-KEEP/eeg/sub-KEEP_task-Movie_eeg.set", {"sub-KEEP"}
    )


def test_official_study_alias_points_at_download_root(tmp_path):
    downloader.ensure_official_study_alias(tmp_path)

    alias = tmp_path / "Shirazi2024Hbn"
    assert alias.is_symlink()
    assert alias.resolve() == tmp_path.resolve()
