from __future__ import annotations

import json
from pathlib import Path

from babysitter_hermes.models import ClaudeTriageDecision, TriageStatus
from babysitter_hermes.triage import (
    anthropic_model_name,
    build_triage_report,
    should_notify_for_triage,
    should_run_hermes_for_triage,
)


def test_triage_decision_gates_notifications_and_hermes() -> None:
    healthy = ClaudeTriageDecision(
        status=TriageStatus.HEALTHY,
        needs_action=False,
        confidence=0.9,
        summary="Run looks healthy.",
    )
    suspicious = ClaudeTriageDecision(
        status=TriageStatus.SUSPICIOUS,
        needs_action=False,
        confidence=0.7,
        summary="Loss is wobbling.",
        questions_for_user=["Should I keep watching this trend?"],
    )
    needs_action = ClaudeTriageDecision(
        status=TriageStatus.NEEDS_ACTION,
        needs_action=True,
        confidence=0.8,
        summary="Loss became NaN.",
    )

    assert should_notify_for_triage(healthy) is False
    assert should_run_hermes_for_triage(healthy) is False
    assert should_notify_for_triage(suspicious) is True
    assert should_run_hermes_for_triage(suspicious) is False
    assert should_notify_for_triage(needs_action) is True
    assert should_run_hermes_for_triage(needs_action) is True


def test_inconsistent_suspicious_triage_does_not_run_hermes() -> None:
    decision = ClaudeTriageDecision(
        status=TriageStatus.SUSPICIOUS,
        needs_action=True,
        confidence=0.5,
        summary="Suspicious but not classified needs_action.",
    )

    assert should_notify_for_triage(decision) is True
    assert should_run_hermes_for_triage(decision) is False


def test_anthropic_model_name_strips_hermes_provider_prefix() -> None:
    assert anthropic_model_name("anthropic/claude-opus-4-7") == "claude-opus-4-7"
    assert anthropic_model_name("claude-opus-4-7") == "claude-opus-4-7"


def test_build_triage_report_turns_suspicious_into_user_question() -> None:
    decision = ClaudeTriageDecision(
        status=TriageStatus.SUSPICIOUS,
        needs_action=False,
        confidence=0.72,
        summary="Validation loss spiked.",
        reasoning="Recent graph shows a sudden spike.",
        questions_for_user=["Should I keep monitoring before escalating?"],
    )

    report = build_triage_report(decision)

    assert report.severity.value == "warning"
    assert report.requires_user_approval is True
    assert "Should I keep monitoring" in report.proposed_fixes[0]


def test_triage_decision_can_be_saved_as_json(tmp_path: Path) -> None:
    decision = ClaudeTriageDecision(
        status=TriageStatus.ERROR,
        needs_action=False,
        confidence=0.2,
        summary="Could not parse all evidence.",
        questions_for_user=["A configured log was missing. What should I do?"],
    )

    path = tmp_path / "triage.json"
    path.write_text(decision.model_dump_json(indent=2))

    loaded = ClaudeTriageDecision.model_validate(json.loads(path.read_text()))
    assert loaded.status == TriageStatus.ERROR
    assert loaded.questions_for_user
