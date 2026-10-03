import subprocess
import sys
from types import SimpleNamespace

import launch


def fake_setup(monkeypatch, tmp_path):
    root = tmp_path / "Portfolio Lab"
    root.mkdir()
    (root / "requirements.txt").write_text("demo==1\n")
    monkeypatch.setattr(launch, "ROOT", root)
    monkeypatch.setattr(launch, "sys", SimpleNamespace(
        version_info=(3, 14), executable=sys.executable,
        argv=["launch.py", "--server.port=8502"]))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["-m", "venv"]:
            python = root / ".venv" / ("Scripts/python.exe" if launch.os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True)
            python.touch()
        return subprocess.CompletedProcess(command, 0, stdout="(3, 14)\n")

    monkeypatch.setattr(launch.subprocess, "run", run)
    return root, calls


def test_first_setup_is_reused_and_requirement_changes_reinstall(monkeypatch, tmp_path):
    root, calls = fake_setup(monkeypatch, tmp_path)
    assert launch.main() == 0
    assert launch.main() == 0
    assert sum(command[1:3] == ["-m", "venv"] for command in calls) == 1
    assert sum(command[1:3] == ["-m", "pip"] for command in calls) == 1
    (root / "requirements.txt").write_text("demo==2\n")
    assert launch.main() == 0
    assert sum(command[1:3] == ["-m", "pip"] for command in calls) == 2
    starts = [command for command in calls if command[1:3] == ["-m", "streamlit"]]
    assert len(starts) == 3
    assert "--server.headless=false" in starts[0]
    assert "--server.showEmailPrompt=false" in starts[0]
    assert starts[0][-1] == "--server.port=8502"


def test_failed_install_is_retried_before_app_starts(monkeypatch, tmp_path):
    root, calls = fake_setup(monkeypatch, tmp_path)
    successful_run = launch.subprocess.run

    def fail_install(command, **kwargs):
        if command[1:3] == ["-m", "pip"]:
            raise subprocess.CalledProcessError(1, command)
        return successful_run(command, **kwargs)

    monkeypatch.setattr(launch.subprocess, "run", fail_install)
    assert launch.main() == 1
    assert not (root / ".venv/portfolio-lab-requirements.sha256").exists()
    assert not any(command[1:3] == ["-m", "streamlit"] for command in calls)
    monkeypatch.setattr(launch.subprocess, "run", successful_run)
    assert launch.main() == 0
    assert (root / ".venv/portfolio-lab-requirements.sha256").exists()


def test_wrong_python_does_not_create_environment(monkeypatch, tmp_path, capsys):
    root, calls = fake_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(launch.sys, "version_info", (3, 11))
    assert launch.main() == 1
    assert not calls and not (root / ".venv").exists()
    assert "Python 3.14" in capsys.readouterr().out
