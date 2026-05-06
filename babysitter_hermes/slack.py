from __future__ import annotations

import json
import os
from pathlib import Path

from .models import DiagnosisReport


def build_slack_message(
    *,
    report: DiagnosisReport,
    wandb_run_url: str | None,
    latest_step: int | None,
    artifact_dir: Path,
    slack_user_id: str | None = None,
) -> dict:
    text = f"Babysitter Hermes {report.severity.value}: {report.summary}"
    fields = [
        f"*Severity:* {report.severity.value}",
        f"*Confidence:* {report.confidence:.2f}",
        f"*Latest step:* {latest_step if latest_step is not None else 'unknown'}",
        f"*W&B:* {wandb_run_url or 'unavailable'}",
        f"*Artifacts:* `{artifact_dir}`",
        f"*Requires approval:* {report.requires_user_approval}",
    ]
    message = {
        "text": text,
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": text}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(fields)}},
        ],
    }
    if report.proposed_fixes:
        fixes = "\n".join(f"- {fix}" for fix in report.proposed_fixes[:5])
        message["blocks"].append(
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Next actions:*\n{fixes}"}}
        )
    if slack_user_id:
        message["channel"] = slack_user_id
    return message


def save_slack_payload(
    *,
    report: DiagnosisReport,
    wandb_run_url: str | None,
    latest_step: int | None,
    artifact_dir: Path,
    output_path: Path,
    slack_user_id: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            build_slack_message(
                report=report,
                wandb_run_url=wandb_run_url,
                latest_step=latest_step,
                artifact_dir=artifact_dir,
                slack_user_id=slack_user_id or os.environ.get("SLACK_USER_ID") or None,
            ),
            indent=2,
            default=str,
        )
    )
    return output_path


def send_slack_dm(
    *,
    report: DiagnosisReport,
    wandb_run_url: str | None,
    latest_step: int | None,
    artifact_dir: Path,
    slack_user_id: str | None,
) -> str:
    from slack_sdk import WebClient

    token = os.environ.get("SLACK_BOT_TOKEN", "")
    user_id = slack_user_id or os.environ.get("SLACK_USER_ID", "")
    if not token or not user_id:
        return "Slack credentials missing; set SLACK_BOT_TOKEN and SLACK_USER_ID."
    message = build_slack_message(
        report=report,
        wandb_run_url=wandb_run_url,
        latest_step=latest_step,
        artifact_dir=artifact_dir,
        slack_user_id=user_id,
    )
    response = WebClient(token=token).chat_postMessage(
        channel=message["channel"],
        text=message["text"],
        blocks=message["blocks"],
    )
    return f"Slack DM sent: ts={response.get('ts', 'unknown')}"
