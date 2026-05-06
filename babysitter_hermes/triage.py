from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from .models import ClaudeTriageDecision, DiagnosisReport, Severity, TriageStatus


TRIAGE_TOOL = {
    "name": "submit_triage",
    "description": "Submit the first-pass training health triage.",
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["healthy", "suspicious", "needs_action", "error", "unknown"],
            },
            "needs_action": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "summary": {"type": "string"},
            "reasoning": {"type": "string"},
            "signals": {"type": "array", "items": {"type": "string"}},
            "evidence_paths": {"type": "array", "items": {"type": "string"}},
            "questions_for_user": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["status", "needs_action", "confidence", "summary"],
    },
}


def should_notify_for_triage(decision: ClaudeTriageDecision) -> bool:
    return decision.status != TriageStatus.HEALTHY


def should_run_hermes_for_triage(decision: ClaudeTriageDecision) -> bool:
    return decision.status == TriageStatus.NEEDS_ACTION


def anthropic_model_name(model: str) -> str:
    if model.startswith("anthropic/"):
        return model.split("/", 1)[1]
    return model


def build_triage_report(decision: ClaudeTriageDecision) -> DiagnosisReport:
    severity = {
        TriageStatus.HEALTHY: Severity.INFO,
        TriageStatus.SUSPICIOUS: Severity.WARNING,
        TriageStatus.NEEDS_ACTION: Severity.CRITICAL,
        TriageStatus.ERROR: Severity.CRITICAL,
        TriageStatus.UNKNOWN: Severity.WARNING,
    }[decision.status]
    next_actions = list(decision.questions_for_user)
    if decision.status == TriageStatus.SUSPICIOUS and not next_actions:
        next_actions.append("Should I keep monitoring this run before escalating?")
    return DiagnosisReport(
        severity=severity,
        confidence=decision.confidence,
        summary=decision.summary,
        reasoning=decision.reasoning,
        evidence=decision.evidence_paths,
        proposed_fixes=next_actions,
        requires_user_approval=decision.status != TriageStatus.HEALTHY,
    )


def save_triage_decision(decision: ClaudeTriageDecision, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(decision.model_dump_json(indent=2))
    return path


def run_claude_triage(
    *,
    evidence: dict[str, Any],
    model: str,
    max_tokens: int,
    output_path: Path,
    client: Any | None = None,
) -> ClaudeTriageDecision:
    client = client or Anthropic()
    response = client.messages.create(
        model=anthropic_model_name(model),
        max_tokens=max_tokens,
        system=_triage_system_prompt(),
        messages=[
            {
                "role": "user",
                "content": (
                    "Review this training evidence and classify it. "
                    "Never skip missing or malformed datapoints; ask the user what to do.\n\n"
                    f"{json.dumps(evidence, indent=2, default=str)}"
                ),
            }
        ],
        tools=[TRIAGE_TOOL],
        tool_choice={"type": "tool", "name": "submit_triage"},
    )
    data = _extract_tool_input(response)
    decision = ClaudeTriageDecision.model_validate(
        {
            **data,
            "model_id": model,
        }
    )
    save_triage_decision(decision, output_path)
    return decision


def _triage_system_prompt() -> str:
    return """You are a cheap first-pass ML training babysitter.
Classify the run as healthy, suspicious, needs_action, error, or unknown.
healthy means no user message is needed.
suspicious means message the user with concise evidence and ask what to do.
needs_action means message the user and escalate to Hermes.
error or unknown means message the user and do not silently continue.
Never skip datapoints. If evidence is missing, malformed, or incomplete, put a question in questions_for_user.
"""


def _extract_tool_input(response: Any) -> dict[str, Any]:
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "submit_triage":
            return dict(getattr(block, "input", {}) or {})
        if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == "submit_triage":
            return dict(block.get("input") or {})
    raise ValueError("Claude triage response did not include submit_triage tool output.")
