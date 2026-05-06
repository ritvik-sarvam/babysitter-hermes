from __future__ import annotations

from pathlib import Path
from typing import Any

from babysitter.models import DiagnosisReport, Severity
from babysitter.slack import save_slack_payload, send_slack_dm

from ._common import ensure_within_artifact_dir, error_payload, ok_payload, path_arg


def babysitter_send_slack(args: dict[str, Any], **kwargs: Any) -> str:
    report_data = args.get("report") or {}
    if not report_data.get("summary"):
        return error_payload("Missing required report.summary")
    artifact_dir = path_arg(args, "artifact_dir")
    if artifact_dir is None:
        return error_payload("Missing required argument: artifact_dir")

    severity_value = report_data.get("severity") or "info"
    report = DiagnosisReport(
        severity=Severity(severity_value),
        confidence=float(report_data.get("confidence") or 0.0),
        summary=str(report_data["summary"]),
        reasoning=str(report_data.get("reasoning") or ""),
        evidence=list(report_data.get("evidence") or []),
        likely_causes=list(report_data.get("likely_causes") or []),
        suspect_step_range=report_data.get("suspect_step_range"),
        most_likely_first_bad_step=report_data.get("most_likely_first_bad_step"),
        proposed_fixes=list(report_data.get("proposed_fixes") or report_data.get("proposed_next_actions") or []),
        requires_user_approval=bool(report_data.get("requires_user_approval")),
        cited_sources=list(report_data.get("cited_sources") or []),
    )
    output_path = path_arg(args, "output_path") or artifact_dir / "slack_payload.json"
    allowed, error = ensure_within_artifact_dir(
        artifact_dir=artifact_dir,
        target_path=output_path,
    )
    if not allowed:
        return error_payload(error or "output_path is outside artifact_dir")
    slack_user_id = args.get("slack_user_id")
    wandb_run_url = args.get("wandb_run_url")
    latest_step = args.get("latest_step")
    if isinstance(latest_step, float):
        latest_step = int(latest_step)
    mode = str(args.get("mode") or "dry-run")

    try:
        payload_path = save_slack_payload(
            report=report,
            wandb_run_url=wandb_run_url,
            latest_step=latest_step,
            artifact_dir=artifact_dir,
            output_path=output_path,
            slack_user_id=slack_user_id,
        )
        status = f"Slack dry-run payload saved: {payload_path}"
        if mode == "send":
            status = send_slack_dm(
                report=report,
                wandb_run_url=wandb_run_url,
                latest_step=latest_step,
                artifact_dir=artifact_dir,
                slack_user_id=slack_user_id,
            )
    except Exception as exc:
        return error_payload(f"Slack handling failed: {type(exc).__name__}: {exc}")

    return ok_payload(payload_path=str(payload_path), status=status)
