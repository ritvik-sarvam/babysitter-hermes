from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import RunSnapshot


def save_wandb_fetch_artifacts(snapshot: RunSnapshot, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "snapshot": output_dir / "snapshot.json",
        "coverage": output_dir / "coverage.json",
        "training_rows": output_dir / "training_rows.jsonl",
        "system_rows": output_dir / "system_rows.jsonl",
        "aligned_system_rows": output_dir / "aligned_system_rows.jsonl",
    }
    paths["snapshot"].write_text(snapshot.model_dump_json(indent=2, fallback=str))
    paths["coverage"].write_text(snapshot.coverage.model_dump_json(indent=2))
    _write_jsonl(paths["training_rows"], snapshot.training_rows)
    _write_jsonl(paths["system_rows"], snapshot.system_rows)
    _write_jsonl(paths["aligned_system_rows"], snapshot.aligned_system_rows)
    return paths


def save_kb_retrieval_artifacts(
    *,
    output_dir: Path,
    trace: dict[str, Any],
    retrieved_evidence: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieval_trace.json").write_text(json.dumps(trace, indent=2, default=str))
    (output_dir / "retrieved_evidence.txt").write_text(retrieved_evidence)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(f"{json.dumps(row, default=str)}\n" for row in rows))
