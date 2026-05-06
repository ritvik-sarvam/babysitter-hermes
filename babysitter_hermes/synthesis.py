from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import FinalDiagnosis


def diagnosis_to_markdown(data: dict[str, Any] | FinalDiagnosis, *, artifact_dir: str) -> str:
    diagnosis = data if isinstance(data, FinalDiagnosis) else FinalDiagnosis.model_validate(data)

    subagents = []
    for report in diagnosis.subagent_reports:
        subagents.append(
            f"- {report.agent_name}: {report.summary} "
            f"(severity={report.severity}, confidence={report.confidence:.2f})"
        )
    questions = list(diagnosis.questions_for_user)
    for report in diagnosis.subagent_reports:
        questions.extend(report.questions_for_user)

    return f"""# Babysitter Hermes Diagnosis

**Severity:** {diagnosis.severity}
**Confidence:** {diagnosis.confidence:.2f}
**Artifacts:** `{artifact_dir}`

## Summary
{diagnosis.summary}

## Reasoning
{diagnosis.reasoning or "Not provided."}

## Subagent Findings
{_bullets(subagents)}

## Questions For User
{_bullets(questions)}

## Proposed Next Actions
{_bullets(diagnosis.proposed_next_actions)}

Requires user approval before mutation: {diagnosis.requires_user_approval}
"""


def save_final_diagnosis(
    data: dict[str, Any] | FinalDiagnosis,
    *,
    output_dir: Path,
    artifact_dir: str,
) -> dict[str, Path]:
    diagnosis = data if isinstance(data, FinalDiagnosis) else FinalDiagnosis.model_validate(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "diagnosis.json"
    markdown_path = output_dir / "diagnosis.md"
    json_path.write_text(diagnosis.model_dump_json(indent=2))
    markdown_path.write_text(diagnosis_to_markdown(diagnosis, artifact_dir=artifact_dir))
    return {"json": json_path, "markdown": markdown_path}


def _bullets(items: list[str]) -> str:
    if not items:
        return "- None"
    return "\n".join(f"- {item}" for item in items)
