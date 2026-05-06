from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .artifacts import create_interval_layout, create_run_layout, write_json
from .config import BabysitterHermesConfig, load_config
from .hermes_client import build_hermes_invocation, copy_hermes_logs, run_hermes
from .models import SchedulerRunResult, SchedulerState, WandbStatus
from .prompts import build_scheduler_prompt
from .tools.wandb_tools import fetch_status
from .wandb_reader import resolve_wandb_run, wait_for_wandb_run_file


StatusProvider = Callable[[str], WandbStatus]
HermesDispatch = Callable[..., dict[str, Any]]
logger = logging.getLogger(__name__)


def configure_scheduler_file_logging(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)
    resolved_path = path.resolve()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == resolved_path:
            return path

    handler = logging.FileHandler(path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root_logger.addHandler(handler)
    return path


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
        logger.info("Fetching W&B status for %s", self.run_path)
        try:
            status = self.status_provider(self.run_path)
        except Exception as exc:
            error = f"W&B status fetch raised {type(exc).__name__}: {exc}"
            logger.exception("W&B status fetch failed for %s", self.run_path)
            write_json(self.run_layout.metadata_dir / "latest_status_error.json", {"error": error})
            self._save_state()
            return SchedulerRunResult(
                dispatched=False,
                status=WandbStatus(run_path=self.run_path),
                error=error,
            )
        logger.info(
            "W&B status for %s: step=%s state=%s",
            self.run_path,
            status.latest_step,
            status.state,
        )
        write_json(self.run_layout.metadata_dir / "latest_status.json", status.model_dump(mode="json"))
        if not should_dispatch_analysis(
            status=status,
            scheduler_state=self.state,
            monitor_every_steps=self.monitor_every_steps,
            terminal_states=self.terminal_states,
        ):
            logger.info(
                "No Hermes dispatch needed for %s; last_analyzed_step=%s",
                self.run_path,
                self.state.last_analyzed_step,
            )
            self._save_state()
            return SchedulerRunResult(dispatched=False, status=status)

        interval_index = self.state.last_interval_index + 1
        interval = create_interval_layout(
            run_layout=self.run_layout,
            interval_index=interval_index,
            latest_step=status.latest_step,
        )
        write_json(interval.wandb_status_dir / "status.json", status.model_dump(mode="json"))
        logger.info(
            "Dispatching Hermes analysis interval %s for %s into %s",
            interval_index,
            self.run_path,
            interval.interval_dir,
        )
        try:
            hermes_result = self.hermes_dispatch(
                run_path=self.run_path,
                status=status,
                interval_dir=interval.interval_dir,
            )
        except Exception as exc:
            error = f"Hermes dispatch raised {type(exc).__name__}: {exc}"
            logger.exception("Hermes dispatch raised for %s", self.run_path)
            write_json(interval.synthesis_dir / "hermes_error.json", {"error": error})
            copied_logs = copy_hermes_logs(interval.synthesis_dir / "hermes_logs")
            if copied_logs:
                logger.info("Copied Hermes logs into %s", interval.synthesis_dir / "hermes_logs")
            self._save_state()
            return SchedulerRunResult(
                dispatched=True,
                status=status,
                interval_dir=interval.interval_dir,
                error=error,
            )
        write_json(interval.synthesis_dir / "hermes_result.json", hermes_result)
        copied_logs = copy_hermes_logs(interval.synthesis_dir / "hermes_logs")
        if copied_logs:
            logger.info("Copied Hermes logs into %s", interval.synthesis_dir / "hermes_logs")
        returncode = hermes_result.get("returncode")
        if isinstance(returncode, int) and returncode != 0:
            error = f"Hermes dispatch failed with returncode {returncode}"
            logger.error("%s; see %s", error, interval.synthesis_dir / "hermes_result.json")
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
        logger.info(
            "Hermes analysis interval %s completed for %s; artifacts=%s",
            interval_index,
            self.run_path,
            interval.interval_dir,
        )
        return SchedulerRunResult(
            dispatched=True,
            status=status,
            interval_dir=interval.interval_dir,
            hermes_result=hermes_result,
        )

    def run_forever(self, *, poll_seconds: int) -> None:
        while True:
            self.run_once()
            logger.info("Sleeping %s seconds before next W&B poll", poll_seconds)
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
        return run_hermes(
            invocation,
            stream=True,
            output_dir=interval_dir / "7_final_synthesis" / "hermes_stream",
        )

    return dispatch


def _subagent_statuses(config: BabysitterHermesConfig) -> dict[str, str]:
    return {
        name: "enabled" if value else "disabled"
        for name, value in config.analysis.subagents.model_dump().items()
    }


def run_from_config(path: Path) -> None:
    logger.info("Loading Babysitter Hermes config from %s", path)
    config = load_config(path)
    scheduler_log_path = configure_scheduler_file_logging(
        config.workdir / config.project_id / "0_scheduler_logs" / "scheduler.log"
    )
    logger.info("Scheduler log file: %s", scheduler_log_path)
    run_path = (
        config.run.wandb_run
        if config.run.wandb_run
        else wait_for_wandb_run_file(
            config.run.wandb_run_file,
            timeout_seconds=config.run.wait_for_run_file_seconds,
        )
    )
    logger.info("Resolved W&B run path: %s", run_path)
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
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_from_config(args.config)
