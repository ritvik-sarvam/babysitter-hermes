from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from babysitter_hermes.diagnose import diagnose_snapshot
from babysitter_hermes.models import RenderedArtifacts
from babysitter_hermes.render import render_diagnosis_boards
from babysitter_hermes.wandb_reader import load_snapshot

from ._common import ensure_within_artifact_dir, error_payload, ok_payload, path_arg


def babysitter_render_graphs(args: dict[str, Any], **kwargs: Any) -> str:
    snapshot_path = path_arg(args, "snapshot_path")
    output_dir = path_arg(args, "output_dir")
    if snapshot_path is None:
        return error_payload("Missing required argument: snapshot_path")
    if output_dir is None:
        return error_payload("Missing required argument: output_dir")
    artifact_dir = path_arg(args, "artifact_dir")
    if artifact_dir is None:
        return error_payload("Missing required argument: artifact_dir")
    allowed, error = ensure_within_artifact_dir(
        artifact_dir=artifact_dir,
        target_path=output_dir,
    )
    if not allowed:
        return error_payload(error or "output_dir is outside artifact_dir")
    recent_steps = int(args.get("recent_steps") or 100)

    try:
        snapshot = load_snapshot(snapshot_path)
        artifacts = render_diagnosis_boards(
            snapshot=snapshot,
            output_dir=output_dir,
            recent_steps=recent_steps,
            raw_snapshot_path=snapshot_path,
        )
    except Exception as exc:
        return error_payload(f"Graph rendering failed: {type(exc).__name__}: {exc}")

    return ok_payload(
        full_board=str(artifacts.full_board),
        recent_board=str(artifacts.recent_board),
        zoom_board=str(artifacts.zoom_board),
        metadata_path=str(artifacts.metadata_path),
        raw_snapshot_path=str(artifacts.raw_snapshot_path),
        latest_step=artifacts.latest_step,
        suspect_window_rows=artifacts.suspect_window_rows,
    )


def babysitter_analyze_graphs(args: dict[str, Any], **kwargs: Any) -> str:
    snapshot_path = path_arg(args, "snapshot_path")
    if snapshot_path is None:
        return error_payload("Missing required argument: snapshot_path")

    required_paths = ["full_board", "recent_board", "zoom_board", "metadata_path"]
    missing = [name for name in required_paths if not args.get(name)]
    if missing:
        return error_payload(f"Missing required graph artifact argument(s): {', '.join(missing)}")

    try:
        snapshot = load_snapshot(snapshot_path)
        artifacts = RenderedArtifacts(
            full_board=Path(str(args["full_board"])),
            recent_board=Path(str(args["recent_board"])),
            zoom_board=Path(str(args["zoom_board"])),
            metadata_path=Path(str(args["metadata_path"])),
            raw_snapshot_path=snapshot_path,
            latest_step=args.get("latest_step"),
            suspect_window_rows=args.get("suspect_window_rows") or [],
        )
        report = asyncio.run(diagnose_snapshot(snapshot=snapshot, artifacts=artifacts))
    except Exception as exc:
        return error_payload(f"Graph analysis failed: {type(exc).__name__}: {exc}")

    return ok_payload(report=report.model_dump(mode="json"))
