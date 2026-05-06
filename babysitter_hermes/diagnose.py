from __future__ import annotations

from .models import DiagnosisReport, RenderedArtifacts, RunSnapshot, Severity


async def diagnose_snapshot(
    *,
    snapshot: RunSnapshot,
    artifacts: RenderedArtifacts,
) -> DiagnosisReport:
    warnings = list(snapshot.coverage.warnings)
    if not snapshot.training_rows:
        return DiagnosisReport(
            severity=Severity.WARNING,
            confidence=0.6,
            summary="No W&B training rows were available for graph analysis.",
            reasoning="The graph analyzer could not inspect training metrics because the snapshot has no rows.",
            evidence=warnings,
            requires_user_approval=False,
        )
    return DiagnosisReport(
        severity=Severity.INFO,
        confidence=0.4,
        summary="Graph artifacts were rendered for Hermes specialist analysis.",
        reasoning=(
            "Standalone babysitter-hermes renders graph boards and leaves the deeper "
            "visual interpretation to the Hermes agent/subagents."
        ),
        evidence=[
            f"full_board={artifacts.full_board}",
            f"recent_board={artifacts.recent_board}",
            f"zoom_board={artifacts.zoom_board}",
            *warnings,
        ],
        requires_user_approval=False,
    )
