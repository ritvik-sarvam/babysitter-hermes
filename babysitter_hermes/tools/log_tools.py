from __future__ import annotations

from pathlib import Path
from typing import Any

from ._common import list_path_arg, ok_payload


LOG_EXTENSIONS = {".log", ".out", ".err", ".txt", ".jsonl", ".json"}
SIGNAL_TERMS = (
    "error",
    "exception",
    "traceback",
    "oom",
    "out of memory",
    "cuda",
    "failed",
    "killed",
    "missing",
    "rate limit",
)


def babysitter_read_logs(args: dict[str, Any], **kwargs: Any) -> str:
    log_paths = list_path_arg(args, "log_paths")
    max_bytes = int(args.get("max_bytes") or 200000)
    max_files = int(args.get("max_files") or 20)
    run_hint = str(args.get("run_hint") or "").lower()

    excerpts = []
    limitations = []
    inspected_files = []
    remaining_budget = max_bytes

    candidate_files = _candidate_log_files(log_paths, run_hint, max_files, limitations)
    for path in candidate_files:
        if remaining_budget <= 0:
            limitations.append(
                "Byte budget exhausted before all configured logs were inspected; increase max_bytes for a complete pass."
            )
            break
        try:
            chunk = _read_tail(path, remaining_budget)
        except Exception as exc:
            limitations.append(f"Could not read {path}: {type(exc).__name__}: {exc}")
            continue

        inspected_files.append(str(path))
        remaining_budget -= len(chunk.encode("utf-8", errors="replace"))
        matched_lines = _signal_lines(chunk)
        excerpts.append(
            {
                "path": str(path),
                "bytes_read": len(chunk.encode("utf-8", errors="replace")),
                "tail": chunk[-4000:],
                "signal_lines": matched_lines,
            }
        )

    return ok_payload(
        inspected_files=inspected_files,
        excerpts=excerpts,
        limitations=limitations,
    )


def _read_tail(path: Path, max_bytes: int) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read(max_bytes).decode("utf-8", errors="replace")


def _candidate_log_files(
    log_paths: list[Path],
    run_hint: str,
    max_files: int,
    limitations: list[str],
) -> list[Path]:
    files = []
    for path in log_paths:
        if not path.exists():
            limitations.append(f"Configured log path does not exist: {path}")
            continue
        if path.is_file():
            files.append(path)
            continue
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and child.suffix.lower() in LOG_EXTENSIONS:
                    if run_hint and run_hint not in child.name.lower() and run_hint not in str(child).lower():
                        continue
                    files.append(child)
            continue
        limitations.append(f"Configured log path is neither file nor directory: {path}")

    files.sort(key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True)
    if len(files) > max_files:
        limitations.append(
            f"Found {len(files)} candidate logs; inspected {max_files}. Increase max_files for complete inspection."
        )
    return files[:max_files]


def _signal_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(term in lowered for term in SIGNAL_TERMS):
            lines.append(line[-1000:])
    return lines[:100]
