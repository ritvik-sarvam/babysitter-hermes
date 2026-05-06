from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class WandbStatus(BaseModel):
    run_path: str
    latest_step: int | None = None
    state: str | None = None
    run_url: str | None = None
    fetched_at: datetime = Field(default_factory=datetime.now)


class SchedulerState(BaseModel):
    last_analyzed_step: int | None = None
    last_interval_index: int = 0
    terminal_analysis_sent: bool = False


class SchedulerRunResult(BaseModel):
    dispatched: bool
    status: WandbStatus
    interval_dir: Path | None = None
    hermes_result: dict[str, Any] | None = None
    error: str | None = None


class EvidenceCitation(BaseModel):
    source: str
    path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    text: str | None = None


class SubagentReport(BaseModel):
    agent_name: str
    severity: str = "info"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str
    findings: list[str] = Field(default_factory=list)
    evidence: list[EvidenceCitation] = Field(default_factory=list)
    questions_for_user: list[str] = Field(default_factory=list)
    proposed_next_actions: list[str] = Field(default_factory=list)
    requires_user_approval: bool = False
    limitations: list[str] = Field(default_factory=list)


class FinalDiagnosis(BaseModel):
    severity: str = "info"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str
    reasoning: str = ""
    subagent_reports: list[SubagentReport] = Field(default_factory=list)
    questions_for_user: list[str] = Field(default_factory=list)
    proposed_next_actions: list[str] = Field(default_factory=list)
    requires_user_approval: bool = False
