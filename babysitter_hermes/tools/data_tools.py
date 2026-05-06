from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ._common import list_path_arg, ok_payload


TEXT_FIELDS = ("prompt", "completion", "text", "input", "output", "answer", "label")


def babysitter_inspect_data(args: dict[str, Any], **kwargs: Any) -> str:
    # TODO: Wrap per-file inspection in JSON error handling so malformed JSON,
    # bad CSVs, and read failures are reported in limitations instead of
    # escaping the Hermes tool call.
    data_paths = list_path_arg(args, "data_paths")
    max_preview_rows = int(args.get("max_preview_rows") or 50)
    summaries = []
    questions_for_user = []
    limitations = []

    for path in data_paths:
        if not path.exists():
            limitations.append(f"Configured data path does not exist: {path}")
            continue
        if path.is_dir():
            files = [child for child in path.rglob("*") if child.is_file()]
            summaries.append(
                {
                    "path": str(path),
                    "kind": "directory",
                    "file_count": len(files),
                    "preview_files": [str(child) for child in files[:max_preview_rows]],
                }
            )
            if len(files) > max_preview_rows:
                limitations.append(
                    f"Directory {path} has {len(files)} files; previewed {max_preview_rows}. Increase max_preview_rows for broader inspection."
                )
            continue
        summary, questions, file_limitations = _inspect_file(path, max_preview_rows)
        summaries.append(summary)
        questions_for_user.extend(questions)
        limitations.extend(file_limitations)

    return ok_payload(
        summaries=summaries,
        questions_for_user=questions_for_user,
        limitations=limitations,
    )


def _inspect_file(path: Path, max_preview_rows: int) -> tuple[dict[str, Any], list[str], list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _inspect_jsonl(path, max_preview_rows)
    if suffix == ".json":
        return _inspect_json(path, max_preview_rows)
    if suffix == ".csv":
        return _inspect_csv(path, max_preview_rows)
    return _inspect_text(path, max_preview_rows)


def _inspect_jsonl(path: Path, max_preview_rows: int) -> tuple[dict[str, Any], list[str], list[str]]:
    questions = []
    limitations = []
    rows = []
    total_rows = 0
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        total_rows += 1
        if len(rows) < max_preview_rows:
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                questions.append(
                    f"{path} line {line_number} is not valid JSON ({exc}); how should this datapoint be handled?"
                )
                continue
            rows.append(row)
            questions.extend(_questions_for_row(path, line_number, row))
    if total_rows > max_preview_rows:
        limitations.append(
            f"{path} contains {total_rows} rows; previewed {max_preview_rows}. Increase max_preview_rows for complete inspection."
        )
    return (
        {
            "path": str(path),
            "kind": "jsonl",
            "row_count": total_rows,
            "preview_rows": rows,
        },
        questions,
        limitations,
    )


def _inspect_json(path: Path, max_preview_rows: int) -> tuple[dict[str, Any], list[str], list[str]]:
    data = json.loads(path.read_text(errors="replace"))
    rows = data if isinstance(data, list) else [data]
    questions = []
    for index, row in enumerate(rows[:max_preview_rows], start=1):
        questions.extend(_questions_for_row(path, index, row))
    limitations = []
    if len(rows) > max_preview_rows:
        limitations.append(
            f"{path} contains {len(rows)} JSON records; previewed {max_preview_rows}. Increase max_preview_rows for complete inspection."
        )
    return (
        {
            "path": str(path),
            "kind": "json",
            "row_count": len(rows),
            "preview_rows": rows[:max_preview_rows],
        },
        questions,
        limitations,
    )


def _inspect_csv(path: Path, max_preview_rows: int) -> tuple[dict[str, Any], list[str], list[str]]:
    rows = []
    questions = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            if len(rows) < max_preview_rows:
                rows.append(row)
                questions.extend(_questions_for_row(path, index, row))
    limitations = []
    if reader.line_num - 1 > max_preview_rows:
        limitations.append(
            f"{path} contains {reader.line_num - 1} CSV rows; previewed {max_preview_rows}. Increase max_preview_rows for complete inspection."
        )
    return (
        {
            "path": str(path),
            "kind": "csv",
            "row_count": max(reader.line_num - 1, 0),
            "columns": reader.fieldnames or [],
            "preview_rows": rows,
        },
        questions,
        limitations,
    )


def _inspect_text(path: Path, max_preview_rows: int) -> tuple[dict[str, Any], list[str], list[str]]:
    lines = path.read_text(errors="replace").splitlines()
    limitations = []
    if len(lines) > max_preview_rows:
        limitations.append(
            f"{path} contains {len(lines)} text lines; previewed {max_preview_rows}. Increase max_preview_rows for complete inspection."
        )
    return (
        {
            "path": str(path),
            "kind": "text",
            "line_count": len(lines),
            "preview_lines": lines[:max_preview_rows],
        },
        [],
        limitations,
    )


def _questions_for_row(path: Path, row_number: int, row: Any) -> list[str]:
    questions = []
    if not isinstance(row, dict):
        return questions
    for field in TEXT_FIELDS:
        if field in row:
            value = row.get(field)
            if value is None or value == "":
                questions.append(
                    f"{path} row {row_number} has an empty {field!r} field; what should be done with this datapoint?"
                )
            elif isinstance(value, str) and len(value) > 20000:
                questions.append(
                    f"{path} row {row_number} has a very long {field!r} field ({len(value)} chars); should this datapoint be reviewed?"
                )
    return questions
