from __future__ import annotations

from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class TriageStatus(str, Enum):
    HEALTHY = "healthy"
    SUSPICIOUS = "suspicious"
    NEEDS_ACTION = "needs_action"
    ERROR = "error"
    UNKNOWN = "unknown"


SEVERITY_ORDER: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.WARNING: 1,
    Severity.CRITICAL: 2,
}


class MetricCoverage(BaseModel):
    training_rows: int = 0
    system_rows: int = 0
    aligned_system_rows: int = 0
    first_step: int | None = None
    latest_step: int | None = None
    warnings: list[str] = Field(default_factory=list)


class RunSnapshot(BaseModel):
    run_path: str
    run_url: str | None = None
    state: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    training_rows: list[dict[str, Any]] = Field(default_factory=list)
    system_rows: list[dict[str, Any]] = Field(default_factory=list)
    aligned_system_rows: list[dict[str, Any]] = Field(default_factory=list)
    coverage: MetricCoverage = Field(default_factory=MetricCoverage)


class ClaudeTriageDecision(BaseModel):
    status: TriageStatus
    needs_action: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str
    reasoning: str = ""
    signals: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    questions_for_user: list[str] = Field(default_factory=list)
    model_id: str | None = None
    prompt_version: str = "triage-v1"
    created_at: datetime = Field(default_factory=datetime.now)


class RenderedArtifacts(BaseModel):
    full_board: Path
    recent_board: Path
    zoom_board: Path
    metadata_path: Path
    raw_snapshot_path: Path
    latest_step: int | None = None
    suspect_window_rows: list[dict[str, Any]] = Field(default_factory=list)


class DiagnosisReport(BaseModel):
    severity: Severity = Severity.INFO
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str
    reasoning: str = ""
    evidence: list[str] = Field(default_factory=list)
    likely_causes: list[str] = Field(default_factory=list)
    suspect_step_range: str | None = None
    most_likely_first_bad_step: int | None = None
    proposed_fixes: list[str] = Field(default_factory=list)
    requires_user_approval: bool = False
    cited_sources: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


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
