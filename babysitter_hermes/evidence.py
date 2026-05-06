from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import WandbStatus
from .render import render_diagnosis_boards
from .tools.log_tools import babysitter_read_logs
from .tracking import save_wandb_fetch_artifacts
from .wandb_reader import fetch_run_snapshot


def collect_interval_evidence(
    *,
    run_path: str,
    status: WandbStatus,
    interval_dir: Path,
    recent_steps: int,
    log_paths: list[Path],
    max_log_bytes: int,
) -> dict[str, Any]:
    wandb_status_dir = interval_dir / "1_wandb_status"
    wandb_snapshot_dir = interval_dir / "2_wandb_snapshot"
    graphs_dir = interval_dir / "3_graphs"
    logs_dir = interval_dir / "5_logs"
    synthesis_dir = interval_dir / "7_final_synthesis"
    logs_dir.mkdir(parents=True, exist_ok=True)
    synthesis_dir.mkdir(parents=True, exist_ok=True)

    snapshot = fetch_run_snapshot(run_path)
    snapshot_paths = save_wandb_fetch_artifacts(snapshot, wandb_snapshot_dir)
    rendered = render_diagnosis_boards(
        snapshot=snapshot,
        output_dir=graphs_dir,
        recent_steps=recent_steps,
        raw_snapshot_path=snapshot_paths["snapshot"],
    )
    log_summary = json.loads(
        babysitter_read_logs(
            {
                "log_paths": [str(path) for path in log_paths],
                "max_bytes": max_log_bytes,
                "run_hint": run_path.split("/")[-1],
            }
        )
    )
    log_summary_path = logs_dir / "log_summary.json"
    log_summary_path.write_text(json.dumps(log_summary, indent=2, default=str))

    evidence = {
        "run_path": run_path,
        "status": status.model_dump(mode="json"),
        "status_path": str(wandb_status_dir / "status.json"),
        "snapshot_path": str(snapshot_paths["snapshot"]),
        "coverage_path": str(snapshot_paths["coverage"]),
        "coverage": snapshot.coverage.model_dump(mode="json"),
        "graph_paths": {
            "full_board": str(rendered.full_board),
            "recent_board": str(rendered.recent_board),
            "zoom_board": str(rendered.zoom_board),
            "metadata": str(rendered.metadata_path),
        },
        "log_summary_path": str(log_summary_path),
        "log_summary": log_summary,
        "questions_for_user": _questions_for_incomplete_evidence(log_summary),
    }
    evidence_path = synthesis_dir / "evidence_bundle.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, default=str))
    evidence["evidence_path"] = str(evidence_path)
    return evidence


def _questions_for_incomplete_evidence(log_summary: dict[str, Any]) -> list[str]:
    questions = []
    for limitation in log_summary.get("limitations", []):
        questions.append(f"Log evidence limitation: {limitation}. What should I do?")
    return questions
