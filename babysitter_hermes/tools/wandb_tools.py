from __future__ import annotations

from pathlib import Path
from typing import Any

from babysitter_hermes.models import WandbStatus
from babysitter_hermes.tracking import save_wandb_fetch_artifacts
from babysitter_hermes.wandb_reader import fetch_run_snapshot, fetch_run_status, run_url_from_path

from ._common import ensure_within_artifact_dir, error_payload, ok_payload, path_arg


def fetch_status(run_path: str) -> WandbStatus:
    latest_step, state = fetch_run_status(run_path)
    return WandbStatus(
        run_path=run_path,
        latest_step=latest_step,
        state=state,
        run_url=run_url_from_path(run_path),
    )


def babysitter_wandb_status(args: dict[str, Any], **kwargs: Any) -> str:
    run_path = str(args.get("run_path") or "").strip()
    if not run_path:
        return error_payload("Missing required argument: run_path")
    try:
        status = fetch_status(run_path)
    except Exception as exc:
        return error_payload(f"W&B status fetch failed: {type(exc).__name__}: {exc}")
    return ok_payload(status=status.model_dump(mode="json"))


def babysitter_wandb_snapshot(args: dict[str, Any], **kwargs: Any) -> str:
    run_path = str(args.get("run_path") or "").strip()
    if not run_path:
        return error_payload("Missing required argument: run_path")
    output_dir = path_arg(args, "output_dir")
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

    try:
        snapshot = fetch_run_snapshot(run_path)
        paths = save_wandb_fetch_artifacts(snapshot, output_dir)
    except Exception as exc:
        return error_payload(f"W&B snapshot fetch failed: {type(exc).__name__}: {exc}")

    return ok_payload(
        snapshot_path=str(paths["snapshot"]),
        coverage_path=str(paths["coverage"]),
        training_rows_path=str(paths["training_rows"]),
        system_rows_path=str(paths["system_rows"]),
        aligned_system_rows_path=str(paths["aligned_system_rows"]),
        coverage=snapshot.coverage.model_dump(mode="json"),
        row_counts={
            "training_rows": len(snapshot.training_rows),
            "system_rows": len(snapshot.system_rows),
            "aligned_system_rows": len(snapshot.aligned_system_rows),
        },
        latest_step=snapshot.coverage.latest_step,
        state=snapshot.state,
        run_url=snapshot.run_url,
    )
