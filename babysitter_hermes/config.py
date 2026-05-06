from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class HermesConfig(BaseModel):
    command: str = "hermes"
    profile: str = "default"
    model: str = "anthropic/claude-opus-4-7"
    toolsets: list[str] = Field(default_factory=lambda: ["babysitter", "terminal", "web"])
    skill: str = "babysitter:training-babysitter"
    max_iterations: int = 80


class RunConfig(BaseModel):
    wandb_run: str | None = None
    wandb_run_file: Path | None = None
    wait_for_run_file_seconds: int = 1800
    training_code_dir: Path | None = None
    kb_roots: list[Path] = Field(default_factory=list)
    log_paths: list[Path] = Field(default_factory=list)
    data_paths: list[Path] = Field(default_factory=list)
    launch: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_wandb_source(self) -> "RunConfig":
        if bool(self.wandb_run) == bool(self.wandb_run_file):
            raise ValueError("Provide exactly one of run.wandb_run or run.wandb_run_file.")
        return self


class MonitorConfig(BaseModel):
    poll_seconds: int = 30
    monitor_every_steps: int = 25
    recent_steps: int = 100
    terminal_states: list[str] = Field(
        default_factory=lambda: ["finished", "failed", "crashed", "killed"]
    )


class NotificationConfig(BaseModel):
    notify_severity: str = "warning"
    slack_mode: str = "dry-run"
    slack_user_id: str | None = None


class SubagentConfig(BaseModel):
    graph: bool = True
    kb: bool = True
    runtime_logs: bool = True
    code: bool = True
    data: bool = True
    config: bool = True
    fix_strategy: bool = True


class AnalysisLimits(BaseModel):
    max_log_bytes: int = 200000
    max_code_files: int = 40
    max_data_preview_rows: int = 50


class AnalysisConfig(BaseModel):
    subagents: SubagentConfig = Field(default_factory=SubagentConfig)
    limits: AnalysisLimits = Field(default_factory=AnalysisLimits)


class BabysitterHermesConfig(BaseModel):
    project_id: str
    workdir: Path
    hermes: HermesConfig = Field(default_factory=HermesConfig)
    run: RunConfig
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

    def resolve_paths(self, base_dir: Path) -> "BabysitterHermesConfig":
        self.workdir = _resolve_path(self.workdir, base_dir)
        if self.run.wandb_run_file is not None:
            self.run.wandb_run_file = _resolve_path(self.run.wandb_run_file, base_dir)
        if self.run.training_code_dir is not None:
            self.run.training_code_dir = _resolve_path(self.run.training_code_dir, base_dir)
        self.run.kb_roots = [_resolve_path(path, base_dir) for path in self.run.kb_roots]
        self.run.log_paths = [_resolve_path(path, base_dir) for path in self.run.log_paths]
        self.run.data_paths = [_resolve_path(path, base_dir) for path in self.run.data_paths]
        return self

    def redacted_dict(self) -> dict[str, Any]:
        from .artifacts import redact_secrets

        return redact_secrets(self.model_dump(mode="json"))


def load_config(path: Path) -> BabysitterHermesConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    config = BabysitterHermesConfig.model_validate(raw)
    return config.resolve_paths(path.resolve().parent)


def _resolve_path(path: Path, base_dir: Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    return base_dir / expanded
