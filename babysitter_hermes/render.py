from __future__ import annotations

import json
from pathlib import Path

from .models import RenderedArtifacts, RunSnapshot


def render_diagnosis_boards(
    *,
    snapshot: RunSnapshot,
    output_dir: Path,
    recent_steps: int,
    raw_snapshot_path: Path,
) -> RenderedArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_board = output_dir / "full_board.png"
    recent_board = output_dir / "recent_board.png"
    zoom_board = output_dir / "zoom_board.png"
    metadata_path = output_dir / "metadata.json"

    rows = snapshot.training_rows
    _plot_rows(rows, full_board, title=f"Full Run: {snapshot.run_path}")
    _plot_rows(rows[-recent_steps:], recent_board, title=f"Recent {recent_steps} Steps")
    suspect_window_rows = rows[-min(len(rows), max(10, recent_steps // 2)) :]
    _plot_rows(suspect_window_rows, zoom_board, title="Zoom Window")
    metadata_path.write_text(
        json.dumps(
            {
                "run_path": snapshot.run_path,
                "coverage": snapshot.coverage.model_dump(mode="json"),
                "latest_step": snapshot.coverage.latest_step,
                "suspect_window_rows": suspect_window_rows,
            },
            indent=2,
            default=str,
        )
    )
    return RenderedArtifacts(
        full_board=full_board,
        recent_board=recent_board,
        zoom_board=zoom_board,
        metadata_path=metadata_path,
        raw_snapshot_path=raw_snapshot_path,
        latest_step=snapshot.coverage.latest_step,
        suspect_window_rows=suspect_window_rows,
    )


def _plot_rows(rows: list[dict], path: Path, *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row.get("step", row.get("_step", index)) for index, row in enumerate(rows)]
    numeric_keys = []
    for row in rows:
        for key, value in row.items():
            if key in {"step", "_step"}:
                continue
            if isinstance(value, (int, float)) and key not in numeric_keys:
                numeric_keys.append(key)
        if len(numeric_keys) >= 4:
            break

    plt.figure(figsize=(12, 7))
    if not rows or not numeric_keys:
        plt.text(0.5, 0.5, "No numeric metrics available", ha="center", va="center")
    else:
        for key in numeric_keys[:4]:
            values = [row.get(key) for row in rows]
            plt.plot(steps, values, label=key)
        plt.legend()
    plt.title(title)
    plt.xlabel("step")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
