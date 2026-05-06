from __future__ import annotations

import subprocess
from pathlib import Path

from babysitter_hermes.training_launch import (
    TrainingLaunchConfig,
    build_tmux_new_session_command,
    launch_training_tmux,
)


def test_build_tmux_command_runs_script_in_configured_working_dir(tmp_path: Path) -> None:
    script = tmp_path / "train.sh"
    script.write_text("#!/usr/bin/env bash\necho train\n")
    working_dir = tmp_path / "training"
    working_dir.mkdir()

    command = build_tmux_new_session_command(
        session_name="babylm",
        script_path=script,
        working_dir=working_dir,
    )

    assert command == [
        "tmux",
        "new-session",
        "-d",
        "-s",
        "babylm",
        "-c",
        str(working_dir),
        "--",
        "bash",
        "-lc",
        f"exec bash {script}",
    ]


def test_launch_training_tmux_skips_existing_session(tmp_path: Path, monkeypatch) -> None:
    calls = []
    script = tmp_path / "train.sh"
    script.write_text("echo train\n")

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["tmux", "has-session"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError("new-session should not be called")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = launch_training_tmux(
        launch=TrainingLaunchConfig(
            enabled=True,
            backend="tmux",
            session_name="existing",
            script_path=script,
            working_dir=tmp_path,
        ),
        artifact_dir=tmp_path / "launch",
    )

    assert result.status == "skipped_existing"
    assert calls == [["tmux", "has-session", "-t", "existing"]]
    assert (tmp_path / "launch" / "launch_record.json").exists()


def test_launch_training_tmux_starts_session_and_writes_record(tmp_path: Path, monkeypatch) -> None:
    calls = []
    script = tmp_path / "train.sh"
    script.write_text("echo train\n")

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["tmux", "has-session"]:
            return subprocess.CompletedProcess(command, 1, "", "missing")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = launch_training_tmux(
        launch=TrainingLaunchConfig(
            enabled=True,
            backend="tmux",
            session_name="new-session",
            script_path=script,
            working_dir=tmp_path,
        ),
        artifact_dir=tmp_path / "launch",
    )

    assert result.status == "created"
    assert calls[1][0:5] == ["tmux", "new-session", "-d", "-s", "new-session"]
    assert result.record_path == tmp_path / "launch" / "launch_record.json"
