from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from babysitter.wandb_reader import resolve_wandb_run, wait_for_wandb_run_file

from .artifacts import create_interval_layout, create_run_layout, write_json
from .config import BabysitterHermesConfig, load_config
from .hermes_client import build_hermes_invocation, run_hermes
from .models import SchedulerRunResult, SchedulerState, WandbStatus
from .prompts import build_scheduler_prompt
from .tools.wandb_tools import fetch_status


StatusProvider = Callable[[str], WandbStatus]
HermesDispatch = Callable[..., dict[str, Any]]


def should_dispatch_analysis(
    *,
    status: WandbStatus,
    scheduler_state: SchedulerState,
    monitor_every_steps: int,
    terminal_states: set[str],
) -> bool:
    state = (status.state or "").lower()
    if state in terminal_states and not scheduler_state.terminal_analysis_sent:
        return True
    if status.latest_step is None:
        return False
    if scheduler_state.last_analyzed_step is None:
        return True
    return status.latest_step >= scheduler_state.last_analyzed_step + monitor_every_steps


class BabysitterScheduler:
    def __init__(
        self,
        *,
        project_id: str,
        workdir: Path,
        run_path: str,
        monitor_every_steps: int,
        terminal_states: set[str],
        status_provider: StatusProvider = fetch_status,
        hermes_dispatch: HermesDispatch | None = None,
        notify_severity: str = "warning",
    ) -> None:
        self.project_id = project_id
        self.workdir = workdir
        self.run_path = run_path
        self.monitor_every_steps = monitor_every_steps
        self.terminal_states = terminal_states
        self.status_provider = status_provider
        self.hermes_dispatch = hermes_dispatch or self._default_hermes_dispatch
        self.notify_severity = notify_severity
        self.run_layout = create_run_layout(
            workdir=workdir,
            project_id=project_id,
            run_path=run_path,
        )
        self.state_path = self.run_layout.metadata_dir / "scheduler_state.json"
        self.state = self._load_state()

    def run_once(self) -> SchedulerRunResult:
        try:
            status = self.status_provider(self.run_path)
        except Exception as exc:
            error = f"W&B status fetch raised {type(exc).__name__}: {exc}"
            write_json(self.run_layout.metadata_dir / "latest_status_error.json", {"error": error})
            self._save_state()
            return SchedulerRunResult(
                dispatched=False,
                status=WandbStatus(run_path=self.run_path),
                error=error,
            )
        write_json(self.run_layout.metadata_dir / "latest_status.json", status.model_dump(mode="json"))
        if not should_dispatch_analysis(
            status=status,
            scheduler_state=self.state,
            monitor_every_steps=self.monitor_every_steps,
            terminal_states=self.terminal_states,
        ):
            self._save_state()
            return SchedulerRunResult(dispatched=False, status=status)

        interval_index = self.state.last_interval_index + 1
        interval = create_interval_layout(
            run_layout=self.run_layout,
            interval_index=interval_index,
            latest_step=status.latest_step,
        )
        write_json(interval.wandb_status_dir / "status.json", status.model_dump(mode="json"))
        try:
            hermes_result = self.hermes_dispatch(
                run_path=self.run_path,
                status=status,
                interval_dir=interval.interval_dir,
            )
        except Exception as exc:
            error = f"Hermes dispatch raised {type(exc).__name__}: {exc}"
            write_json(interval.synthesis_dir / "hermes_error.json", {"error": error})
            self._save_state()
            return SchedulerRunResult(
                dispatched=True,
                status=status,
                interval_dir=interval.interval_dir,
                error=error,
            )
        write_json(interval.synthesis_dir / "hermes_result.json", hermes_result)
        returncode = hermes_result.get("returncode")
        if isinstance(returncode, int) and returncode != 0:
            error = f"Hermes dispatch failed with returncode {returncode}"
            self._save_state()
            return SchedulerRunResult(
                dispatched=True,
                status=status,
                interval_dir=interval.interval_dir,
                hermes_result=hermes_result,
                error=error,
            )

        self.state.last_interval_index = interval_index
        if status.latest_step is not None:
            self.state.last_analyzed_step = status.latest_step
        if (status.state or "").lower() in self.terminal_states:
            self.state.terminal_analysis_sent = True
        self._save_state()
        return SchedulerRunResult(
            dispatched=True,
            status=status,
            interval_dir=interval.interval_dir,
            hermes_result=hermes_result,
        )

    def run_forever(self, *, poll_seconds: int) -> None:
        while True:
            self.run_once()
            time.sleep(poll_seconds)

    def _default_hermes_dispatch(
        self,
        *,
        run_path: str,
        status: WandbStatus,
        interval_dir: Path,
    ) -> dict[str, Any]:
        prompt = build_scheduler_prompt(
            run_path=run_path,
            latest_step=status.latest_step,
            artifact_dir=str(interval_dir),
            notify_severity=self.notify_severity,
        )
        return {
            "prompt": prompt,
            "note": "No Hermes dispatch callable was configured.",
        }

    def _load_state(self) -> SchedulerState:
        if self.state_path.exists():
            return SchedulerState.model_validate_json(self.state_path.read_text())
        return SchedulerState()

    def _save_state(self) -> None:
        write_json(self.state_path, self.state.model_dump(mode="json"))


def resolve_configured_run_path(config_path: Path) -> str:
    config = load_config(config_path)
    return resolve_wandb_run(
        wandb_run=config.run.wandb_run,
        wandb_run_file=config.run.wandb_run_file,
    )


def build_configured_hermes_dispatch(config: BabysitterHermesConfig) -> HermesDispatch:
    def dispatch(
        *,
        run_path: str,
        status: WandbStatus,
        interval_dir: Path,
    ) -> dict[str, Any]:
        prompt = build_scheduler_prompt(
            run_path=run_path,
            latest_step=status.latest_step,
            artifact_dir=str(interval_dir),
            notify_severity=config.notifications.notify_severity,
            context={
                "skill": config.hermes.skill,
                "training_code_dir": config.run.training_code_dir,
                "kb_roots": config.run.kb_roots,
                "log_paths": config.run.log_paths,
                "data_paths": config.run.data_paths,
                "slack_mode": config.notifications.slack_mode,
                "slack_user_id": config.notifications.slack_user_id,
                "subagents": _subagent_statuses(config),
            },
        )
        invocation = build_hermes_invocation(
            hermes_command=config.hermes.command,
            prompt=prompt,
            profile=config.hermes.profile,
            model=config.hermes.model,
            toolsets=config.hermes.toolsets,
            max_iterations=config.hermes.max_iterations,
        )
        return run_hermes(invocation)

    return dispatch


def _subagent_statuses(config: BabysitterHermesConfig) -> dict[str, str]:
    return {
        name: "enabled" if value else "disabled"
        for name, value in config.analysis.subagents.model_dump().items()
    }


def run_from_config(path: Path) -> None:
    config = load_config(path)
    run_path = (
        config.run.wandb_run
        if config.run.wandb_run
        else wait_for_wandb_run_file(
            config.run.wandb_run_file,
            timeout_seconds=config.run.wait_for_run_file_seconds,
        )
    )
    scheduler = BabysitterScheduler(
        project_id=config.project_id,
        workdir=config.workdir,
        run_path=run_path,
        monitor_every_steps=config.monitor.monitor_every_steps,
        terminal_states={state.lower() for state in config.monitor.terminal_states},
        hermes_dispatch=build_configured_hermes_dispatch(config),
        notify_severity=config.notifications.notify_severity,
    )
    write_json(scheduler.run_layout.metadata_dir / "config.redacted.json", config.redacted_dict())
    scheduler.run_forever(poll_seconds=config.monitor.poll_seconds)


def cli() -> None:
    parser = argparse.ArgumentParser(prog="babysitter-hermes")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_from_config(args.config)
