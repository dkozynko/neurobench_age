from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_article_launcher_is_portable_and_validation_only_by_default() -> None:
    launcher = ROOT / "scripts" / "run_article_experiment.sh"
    text = launcher.read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    assert "REPO_ROOT" in text
    assert "/workspace" not in text
    assert "/venv/main" not in text
    assert "validation_only" in text
    assert "rejects final-test override" in text


def test_analysis_wrapper_uses_installed_package() -> None:
    wrapper = ROOT / "scripts" / "analyze_paper_evidence.py"
    text = wrapper.read_text(encoding="utf-8")
    assert "neurobench_age.analysis.paper_evidence" in text


def test_confirmation_phase_selects_confirmation_config_by_default(tmp_path: Path) -> None:
    launcher = ROOT / "scripts" / "run_article_experiment.sh"
    fake_python = tmp_path / "fake-python"
    args_file = tmp_path / "args.txt"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['ARGS_FILE']).write_text('\\n'.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "MANIFEST": "external-manifest.csv",
            "DATA_ROOT": "external-data",
            "OUTPUT_ROOT": str(tmp_path / "output"),
            "PHASE": "confirmation",
            "SEEDS": "34 35",
            "PYTHON_BIN": str(fake_python),
            "ARGS_FILE": str(args_file),
        }
    )
    subprocess.run(["bash", str(launcher)], cwd=ROOT, env=env, check=True)

    args = args_file.read_text(encoding="utf-8").splitlines()
    config_index = args.index("--config") + 1
    assert args[config_index].endswith("configs/article/confirmation.json")
    assert args[args.index("--evaluation-mode") + 1] == "validation_only"
    assert args[args.index("--seeds") + 1 :] == ["34", "35"]


def test_validation_launcher_rejects_final_test_overrides(tmp_path: Path) -> None:
    launcher = ROOT / "scripts" / "run_article_experiment.sh"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "MANIFEST": "external-manifest.csv",
            "DATA_ROOT": "external-data",
            "OUTPUT_ROOT": str(tmp_path / "output"),
            "PYTHON_BIN": str(fake_python),
        }
    )
    result = subprocess.run(
        ["bash", str(launcher), "--evaluation-mode", "final_test"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "final-test" in result.stderr
