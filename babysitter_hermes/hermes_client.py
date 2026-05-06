from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


@dataclass(frozen=True)
class HermesInvocation:
    command: list[str]
    prompt: str
    cwd: Path | None = None


def build_hermes_invocation(
    *,
    hermes_command: str,
    prompt: str,
    profile: str | None = None,
    model: str | None = None,
    toolsets: list[str] | None = None,
    max_iterations: int | None = None,
    cwd: Path | None = None,
) -> HermesInvocation:
    del profile, max_iterations

    command = [hermes_command, "chat"]
    if model:
        command.extend(["--model", model])
    if toolsets:
        command.extend(["--toolsets", ",".join(toolsets)])
    # TODO: Add a first-class Hermes plugin installation/discovery step so fresh
    # clones expose the babysitter toolset without relying on global Hermes
    # plugin state.
    command.extend(["-q", prompt])
    return HermesInvocation(command=command, prompt=prompt, cwd=cwd)


def run_hermes(
    invocation: HermesInvocation,
    *,
    timeout_seconds: int | None = None,
    stream: bool = True,
    stdout_writer: TextIO | None = None,
    stderr_writer: TextIO | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_path = output_dir / "stdout.log" if output_dir else None
    stderr_path = output_dir / "stderr.log" if output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        invocation.command,
        cwd=invocation.cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    stdout_thread = threading.Thread(
        target=_consume_stream,
        kwargs={
            "source": process.stdout,
            "chunks": stdout_chunks,
            "writer": stdout_writer or sys.stdout,
            "stream": stream,
            "output_path": stdout_path,
        },
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_consume_stream,
        kwargs={
            "source": process.stderr,
            "chunks": stderr_chunks,
            "writer": stderr_writer or sys.stderr,
            "stream": stream,
            "output_path": stderr_path,
        },
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        stdout_thread.join()
        stderr_thread.join()
        raise
    stdout_thread.join()
    stderr_thread.join()

    return {
        "returncode": returncode,
        "stdout": "".join(stdout_chunks),
        "stderr": "".join(stderr_chunks),
        "command": invocation.command,
        "stdout_log": str(stdout_path) if stdout_path else None,
        "stderr_log": str(stderr_path) if stderr_path else None,
    }


def _consume_stream(
    *,
    source: TextIO | None,
    chunks: list[str],
    writer: TextIO,
    stream: bool,
    output_path: Path | None,
) -> None:
    if source is None:
        return

    output_file = output_path.open("w") if output_path else None
    try:
        for line in iter(source.readline, ""):
            chunks.append(line)
            if stream:
                writer.write(line)
                writer.flush()
            if output_file:
                output_file.write(line)
                output_file.flush()
    finally:
        if output_file:
            output_file.close()
        source.close()


def save_hermes_result(path: Path, result: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str))
    return path


def copy_hermes_logs(destination: Path, *, hermes_home: Path | None = None) -> list[Path]:
    source_home = hermes_home or Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    source_dir = source_home / "logs"
    if not source_dir.exists():
        return []

    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in sorted(source_dir.glob("*.log*")):
        if not source.is_file():
            continue
        target = destination / source.name
        shutil.copy2(source, target)
        copied.append(target)
    return copied
