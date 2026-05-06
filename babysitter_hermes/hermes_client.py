from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    command = [hermes_command]
    if profile:
        command.extend(["--profile", profile])
    if model:
        command.extend(["--model", model])
    if toolsets:
        command.extend(["--toolsets", ",".join(toolsets)])
    if max_iterations is not None:
        command.extend(["--max-iterations", str(max_iterations)])
    # TODO: Add a first-class Hermes plugin installation/discovery step so fresh
    # clones expose the babysitter toolset without relying on global Hermes
    # plugin state.
    command.append(prompt)
    return HermesInvocation(command=command, prompt=prompt, cwd=cwd)


def run_hermes(invocation: HermesInvocation, *, timeout_seconds: int | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        invocation.command,
        cwd=invocation.cwd,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "command": invocation.command,
    }


def save_hermes_result(path: Path, result: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str))
    return path
