from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

from babysitter_hermes.config import load_config
from babysitter_hermes.models import WandbStatus
from babysitter_hermes.scheduler import (
    BabysitterScheduler,
    SchedulerState,
    build_configured_hermes_dispatch,
    configure_scheduler_file_logging,
    should_dispatch_analysis,
)


def test_should_dispatch_first_interval_and_terminal_state() -> None:
    state = SchedulerState()

    assert should_dispatch_analysis(
        status=WandbStatus(run_path="entity/project/run", latest_step=10, state="running"),
        scheduler_state=state,
        monitor_every_steps=25,
        terminal_states={"failed", "finished"},
    )

    state.last_analyzed_step = 10
    assert not should_dispatch_analysis(
        status=WandbStatus(run_path="entity/project/run", latest_step=34, state="running"),
        scheduler_state=state,
        monitor_every_steps=25,
        terminal_states={"failed", "finished"},
    )
    assert should_dispatch_analysis(
        status=WandbStatus(run_path="entity/project/run", latest_step=35, state="running"),
        scheduler_state=state,
        monitor_every_steps=25,
        terminal_states={"failed", "finished"},
    )
    assert should_dispatch_analysis(
        status=WandbStatus(run_path="entity/project/run", latest_step=12, state="failed"),
        scheduler_state=state,
        monitor_every_steps=25,
        terminal_states={"failed", "finished"},
    )


def test_scheduler_run_once_dispatches_hermes_and_persists_state(tmp_path: Path) -> None:
    calls: list[str] = []

    def status_provider(run_path: str) -> WandbStatus:
        return WandbStatus(run_path=run_path, latest_step=25, state="running")

    def hermes_dispatch(*, run_path: str, status: WandbStatus, interval_dir: Path) -> dict:
        calls.append(f"{run_path}:{status.latest_step}:{interval_dir.name}")
        return {"severity": "info", "summary": "healthy"}

    scheduler = BabysitterScheduler(
        project_id="demo",
        workdir=tmp_path,
        run_path="entity/project/run",
        monitor_every_steps=25,
        terminal_states={"failed", "finished"},
        status_provider=status_provider,
        hermes_dispatch=hermes_dispatch,
    )

    result = scheduler.run_once()

    assert result.dispatched is True
    assert calls and calls[0].startswith("entity/project/run:25:interval_0001_step_25_")
    state_path = tmp_path / "demo" / "runs" / "entity__project__run" / "0_run_metadata" / "scheduler_state.json"
    assert state_path.exists()
    assert '"last_analyzed_step": 25' in state_path.read_text()


def test_scheduler_logs_status_and_dispatch_progress(tmp_path: Path, caplog) -> None:
    def status_provider(run_path: str) -> WandbStatus:
        return WandbStatus(run_path=run_path, latest_step=25, state="running")

    scheduler = BabysitterScheduler(
        project_id="demo",
        workdir=tmp_path,
        run_path="entity/project/run",
        monitor_every_steps=25,
        terminal_states={"failed", "finished"},
        status_provider=status_provider,
        hermes_dispatch=lambda **kwargs: {"returncode": 0},
    )

    caplog.set_level(logging.INFO, logger="babysitter_hermes.scheduler")
    scheduler.run_once()

    messages = [record.getMessage() for record in caplog.records]
    assert any("Fetching W&B status for entity/project/run" in message for message in messages)
    assert any("W&B status for entity/project/run: step=25 state=running" in message for message in messages)
    assert any("Dispatching Hermes analysis interval 1" in message for message in messages)
    assert any("Hermes analysis interval 1 completed" in message for message in messages)


def test_scheduler_copies_hermes_logs_into_interval_artifacts(tmp_path: Path, monkeypatch) -> None:
    hermes_logs = tmp_path / "fake_home" / ".hermes" / "logs"
    hermes_logs.mkdir(parents=True)
    (hermes_logs / "agent.log").write_text("agent line\n")
    (hermes_logs / "errors.log").write_text("error line\n")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "fake_home" / ".hermes"))

    scheduler = BabysitterScheduler(
        project_id="demo",
        workdir=tmp_path / "artifacts",
        run_path="entity/project/run",
        monitor_every_steps=25,
        terminal_states={"failed", "finished"},
        status_provider=lambda run_path: WandbStatus(
            run_path=run_path,
            latest_step=25,
            state="running",
        ),
        hermes_dispatch=lambda **kwargs: {"returncode": 0},
    )

    result = scheduler.run_once()

    assert result.interval_dir is not None
    copied_logs = result.interval_dir / "7_final_synthesis" / "hermes_logs"
    assert (copied_logs / "agent.log").read_text() == "agent line\n"
    assert (copied_logs / "errors.log").read_text() == "error line\n"


def test_scheduler_file_logging_writes_under_workdir(tmp_path: Path) -> None:
    log_path = configure_scheduler_file_logging(tmp_path / "scheduler.log")

    logging.getLogger("babysitter_hermes.scheduler").info("file logging works")

    assert log_path == tmp_path / "scheduler.log"
    assert "file logging works" in log_path.read_text()


def test_scheduler_does_not_advance_state_when_hermes_fails(tmp_path: Path) -> None:
    def status_provider(run_path: str) -> WandbStatus:
        return WandbStatus(run_path=run_path, latest_step=25, state="running")

    def hermes_dispatch(*, run_path: str, status: WandbStatus, interval_dir: Path) -> dict:
        return {"returncode": 1, "stdout": "", "stderr": "failed"}

    scheduler = BabysitterScheduler(
        project_id="demo",
        workdir=tmp_path,
        run_path="entity/project/run",
        monitor_every_steps=25,
        terminal_states={"failed", "finished"},
        status_provider=status_provider,
        hermes_dispatch=hermes_dispatch,
    )

    result = scheduler.run_once()

    assert result.dispatched is True
    assert result.error == "Hermes dispatch failed with returncode 1"
    assert scheduler.state.last_analyzed_step is None
    assert scheduler.state.last_interval_index == 0


def test_scheduler_records_status_fetch_errors_without_raising(tmp_path: Path) -> None:
    def status_provider(run_path: str) -> WandbStatus:
        raise RuntimeError("network down")

    scheduler = BabysitterScheduler(
        project_id="demo",
        workdir=tmp_path,
        run_path="entity/project/run",
        monitor_every_steps=25,
        terminal_states={"failed", "finished"},
        status_provider=status_provider,
        hermes_dispatch=lambda **kwargs: {"returncode": 0},
    )

    result = scheduler.run_once()

    assert result.dispatched is False
    assert result.error == "W&B status fetch raised RuntimeError: network down"
    assert (
        tmp_path
        / "demo"
        / "runs"
        / "entity__project__run"
        / "0_run_metadata"
        / "latest_status_error.json"
    ).exists()


def test_configured_hermes_dispatch_runs_hermes_with_babysitter_prompt(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project_id: demo
workdir: ./runs
hermes:
  command: hermes
  profile: training
  model: test-model
  toolsets:
    - babysitter
  max_iterations: 12
run:
  wandb_run: entity/project/run
"""
    )
    config = load_config(config_path)
    dispatch = build_configured_hermes_dispatch(config)

    with patch("babysitter_hermes.scheduler.run_hermes") as run_hermes:
        run_hermes.return_value = {"returncode": 0, "stdout": "ok", "stderr": ""}
        result = dispatch(
            run_path="entity/project/run",
            status=WandbStatus(run_path="entity/project/run", latest_step=50, state="running"),
            interval_dir=tmp_path / "interval",
        )

    invocation = run_hermes.call_args.args[0]
    assert result["returncode"] == 0
    assert invocation.command[:7] == [
        "hermes",
        "--profile",
        "training",
        "--model",
        "test-model",
        "--toolsets",
        "babysitter",
    ]
    assert "--max-iterations" in invocation.command
    assert "babysitter_wandb_snapshot" in invocation.prompt
    assert run_hermes.call_args.kwargs["stream"] is True
    assert run_hermes.call_args.kwargs["output_dir"] == tmp_path / "interval" / "7_final_synthesis" / "hermes_stream"


def test_configured_hermes_dispatch_prompt_includes_evidence_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project_id: demo
workdir: ./runs
run:
  wandb_run: entity/project/run
  training_code_dir: ./code
  kb_roots:
    - ./kb
  log_paths:
    - ./logs
  data_paths:
    - ./data
notifications:
  slack_mode: dry-run
  slack_user_id: U123
analysis:
  subagents:
    data: false
"""
    )
    config = load_config(config_path)
    dispatch = build_configured_hermes_dispatch(config)

    with patch("babysitter_hermes.scheduler.run_hermes") as run_hermes:
        run_hermes.return_value = {"returncode": 0, "stdout": "ok", "stderr": ""}
        dispatch(
            run_path="entity/project/run",
            status=WandbStatus(run_path="entity/project/run", latest_step=50, state="running"),
            interval_dir=tmp_path / "interval",
        )

    prompt = run_hermes.call_args.args[0].prompt
    assert str(tmp_path / "code") in prompt
    assert str(tmp_path / "kb") in prompt
    assert str(tmp_path / "logs") in prompt
    assert str(tmp_path / "data") in prompt
    assert "data: disabled" in prompt
    assert "slack_user_id: U123" in prompt
