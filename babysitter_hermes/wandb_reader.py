from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .models import MetricCoverage, RunSnapshot


def resolve_wandb_run(*, wandb_run: str | None, wandb_run_file: Path | None) -> str:
    if bool(wandb_run) == bool(wandb_run_file):
        raise ValueError("Provide exactly one of wandb_run or wandb_run_file.")
    resolved = wandb_run.strip() if wandb_run else wandb_run_file.read_text().strip()
    if len([part for part in resolved.split("/") if part]) != 3:
        raise ValueError("W&B run must look like 'entity/project/run_id'.")
    return resolved


def wait_for_wandb_run_file(path: Path, timeout_seconds: int) -> str:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        if path.exists() and path.read_text().strip():
            try:
                return resolve_wandb_run(wandb_run=None, wandb_run_file=path)
            except ValueError as exc:
                last_error = exc
        time.sleep(1)
    if last_error:
        raise TimeoutError(f"W&B run pointer did not become valid: {last_error}")
    raise TimeoutError(f"W&B run pointer did not appear within {timeout_seconds}s: {path}")


def run_url_from_path(run_path: str) -> str:
    entity, project, run_id = run_path.split("/")
    return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"


def load_snapshot(path: Path) -> RunSnapshot:
    return RunSnapshot.model_validate_json(path.read_text())


def save_snapshot(snapshot: RunSnapshot, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(snapshot.model_dump_json(indent=2, fallback=str))
    return path


def fetch_run_status(run_path: str) -> tuple[int | None, str | None]:
    import wandb

    run = wandb.Api().run(run_path)
    latest_step = getattr(run, "lastHistoryStep", None)
    if latest_step is None:
        rows = list(run.scan_history(keys=["_step"], page_size=1000))
        steps = [row.get("_step") for row in rows if isinstance(row.get("_step"), (int, float))]
        latest_step = int(max(steps)) if steps else None
    return latest_step, getattr(run, "state", None)


def fetch_run_snapshot(run_path: str) -> RunSnapshot:
    import wandb

    run = wandb.Api().run(run_path)
    training_rows = list(run.scan_history(page_size=1000))
    try:
        system_rows = list(run.history(stream="system", pandas=False))
    except Exception:
        system_rows = []
    coverage = build_coverage(training_rows, system_rows)
    return RunSnapshot(
        run_path=run_path,
        run_url=run_url_from_path(run_path),
        state=getattr(run, "state", None),
        config=dict(getattr(run, "config", {}) or {}),
        summary=dict(getattr(run, "summary", {}) or {}),
        training_rows=training_rows,
        system_rows=system_rows,
        aligned_system_rows=[],
        coverage=coverage,
    )


def build_coverage(
    training_rows: list[dict[str, Any]],
    system_rows: list[dict[str, Any]],
) -> MetricCoverage:
    steps = [
        int(step)
        for row in training_rows
        if isinstance((step := row.get("step", row.get("_step"))), (int, float))
    ]
    warnings = []
    if not training_rows:
        warnings.append("No training history rows were fetched from W&B.")
    if system_rows:
        warnings.append("W&B system history may be sampled by the Public API.")
    return MetricCoverage(
        training_rows=len(training_rows),
        system_rows=len(system_rows),
        aligned_system_rows=0,
        first_step=min(steps) if steps else None,
        latest_step=max(steps) if steps else None,
        warnings=warnings,
    )
