from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import TrainingLaunchConfig


@dataclass(frozen=True)
class LaunchResult:
    status: str
    session_name: str
    record_path: Path
    command: list[str]


def build_tmux_new_session_command(
    *,
    session_name: str,
    script_path: Path,
    working_dir: Path,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    env_exports = " ".join(
        f"export {key}={shlex.quote(value)};" for key, value in (extra_env or {}).items()
    )
    shell_command = f"{env_exports} exec bash {shlex.quote(str(script_path))}".strip()
    return [
        "tmux",
        "new-session",
        "-d",
        "-s",
        session_name,
        "-c",
        str(working_dir),
        "--",
        "bash",
        "-lc",
        shell_command,
    ]


def launch_training_tmux(*, launch: TrainingLaunchConfig, artifact_dir: Path) -> LaunchResult:
    if not launch.enabled and launch.backend == "none":
        return _write_launch_record(
            artifact_dir=artifact_dir,
            status="disabled",
            session_name=launch.session_name or "",
            command=[],
            stderr="",
        )
    if launch.backend != "tmux":
        raise ValueError(f"Unsupported launch backend: {launch.backend}")
    if launch.script_path is None:
        raise ValueError("script_path is required for tmux launch.")

    session_name = launch.session_name or f"babysitter-{launch.script_path.stem}"
    working_dir = launch.working_dir or launch.script_path.parent
    has_session_command = ["tmux", "has-session", "-t", session_name]
    existing = subprocess.run(
        has_session_command,
        text=True,
        capture_output=True,
        check=False,
    )
    if existing.returncode == 0 and launch.skip_if_session_exists:
        return _write_launch_record(
            artifact_dir=artifact_dir,
            status="skipped_existing",
            session_name=session_name,
            command=has_session_command,
            stderr=existing.stderr,
        )

    command = build_tmux_new_session_command(
        session_name=session_name,
        script_path=launch.script_path,
        working_dir=working_dir,
        extra_env=launch.extra_env,
    )
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    status = "created" if completed.returncode == 0 else "failed"
    result = _write_launch_record(
        artifact_dir=artifact_dir,
        status=status,
        session_name=session_name,
        command=command,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"tmux launch failed: {completed.stderr}")

    log_path = artifact_dir / "tmux-pane.log"
    subprocess.run(
        ["tmux", "pipe-pane", "-t", session_name, "-o", f"cat >> {shlex.quote(str(log_path))}"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result


def _write_launch_record(
    *,
    artifact_dir: Path,
    status: str,
    session_name: str,
    command: list[str],
    stderr: str,
    returncode: int | None = None,
) -> LaunchResult:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    record_path = artifact_dir / "launch_record.json"
    record = {
        "status": status,
        "session_name": session_name,
        "command": command,
        "stderr": stderr,
        "returncode": returncode,
        "created_at": datetime.now().isoformat(),
    }
    record_path.write_text(json.dumps(record, indent=2, default=str))
    return LaunchResult(
        status=status,
        session_name=session_name,
        record_path=record_path,
        command=command,
    )
