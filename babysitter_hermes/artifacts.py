from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SECRET_KEY_PARTS = ("api_key", "apikey", "token", "secret", "password", "credential")


@dataclass(frozen=True)
class RunArtifactLayout:
    run_dir: Path
    metadata_dir: Path
    intervals_dir: Path


@dataclass(frozen=True)
class IntervalArtifactLayout:
    interval_dir: Path
    wandb_status_dir: Path
    wandb_snapshot_dir: Path
    graphs_dir: Path
    kb_dir: Path
    logs_dir: Path
    subagents_dir: Path
    synthesis_dir: Path
    user_message_dir: Path


def safe_run_dir_name(run_path: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "__", run_path.strip())
    return cleaned.strip("_") or "unknown_run"


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(part in key_text for part in SECRET_KEY_PARTS):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


def create_run_layout(*, workdir: Path, project_id: str, run_path: str) -> RunArtifactLayout:
    run_dir = workdir / project_id / "runs" / safe_run_dir_name(run_path)
    metadata_dir = run_dir / "0_run_metadata"
    intervals_dir = run_dir / "intervals"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    intervals_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifactLayout(
        run_dir=run_dir,
        metadata_dir=metadata_dir,
        intervals_dir=intervals_dir,
    )


def create_interval_layout(
    *,
    run_layout: RunArtifactLayout,
    interval_index: int,
    latest_step: int | None,
    timestamp_label: str | None = None,
) -> IntervalArtifactLayout:
    timestamp = timestamp_label or datetime.now().strftime("%Y%m%d_%H%M%S")
    step_label = str(latest_step) if latest_step is not None else "unknown"
    interval_dir = (
        run_layout.intervals_dir
        / f"interval_{interval_index:04d}_step_{step_label}_{timestamp}"
    )
    paths = {
        "wandb_status": interval_dir / "1_wandb_status",
        "wandb_snapshot": interval_dir / "2_wandb_snapshot",
        "graphs": interval_dir / "3_graphs",
        "kb": interval_dir / "4_kb",
        "logs": interval_dir / "5_logs",
        "subagents": interval_dir / "6_subagents",
        "synthesis": interval_dir / "7_final_synthesis",
        "user_message": interval_dir / "8_user_message",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return IntervalArtifactLayout(
        interval_dir=interval_dir,
        wandb_status_dir=paths["wandb_status"],
        wandb_snapshot_dir=paths["wandb_snapshot"],
        graphs_dir=paths["graphs"],
        kb_dir=paths["kb"],
        logs_dir=paths["logs"],
        subagents_dir=paths["subagents"],
        synthesis_dir=paths["synthesis"],
        user_message_dir=paths["user_message"],
    )
