from __future__ import annotations

import io
import sys
from pathlib import Path

from babysitter_hermes.hermes_client import (
    HermesInvocation,
    build_hermes_invocation,
    run_hermes,
)


def test_build_hermes_invocation_uses_chat_query_cli() -> None:
    invocation = build_hermes_invocation(
        hermes_command="hermes",
        prompt="diagnose this run",
        profile="ignored-by-current-cli",
        model="anthropic/claude-opus-4-7",
        toolsets=["babysitter", "terminal", "web"],
        max_iterations=80,
    )

    assert invocation.command == [
        "hermes",
        "chat",
        "--model",
        "anthropic/claude-opus-4-7",
        "--toolsets",
        "babysitter,terminal,web",
        "-q",
        "diagnose this run",
    ]


def test_run_hermes_streams_stdout_and_stderr_while_collecting(tmp_path: Path) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    invocation = HermesInvocation(
        command=[
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('agent decision: inspect wandb', flush=True); "
                "print('agent warning: needs logs', file=sys.stderr, flush=True)"
            ),
        ],
        prompt="",
    )

    result = run_hermes(
        invocation,
        stdout_writer=stdout,
        stderr_writer=stderr,
        output_dir=tmp_path / "stream",
    )

    assert result["returncode"] == 0
    assert "agent decision: inspect wandb" in stdout.getvalue()
    assert "agent warning: needs logs" in stderr.getvalue()
    assert result["stdout"] == stdout.getvalue()
    assert result["stderr"] == stderr.getvalue()
    assert (tmp_path / "stream" / "stdout.log").read_text() == stdout.getvalue()
    assert (tmp_path / "stream" / "stderr.log").read_text() == stderr.getvalue()
