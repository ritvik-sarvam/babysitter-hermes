from __future__ import annotations

from pathlib import Path
from typing import Any

from ._common import error_payload, ok_payload, path_arg


CODE_EXTENSIONS = {".py", ".sh", ".toml", ".yaml", ".yml", ".json", ".md"}
DEFAULT_TERMS = (
    "train",
    "wandb",
    "loss",
    "reward",
    "dataset",
    "dataloader",
    "optimizer",
    "checkpoint",
)


def babysitter_inspect_code(args: dict[str, Any], **kwargs: Any) -> str:
    code_dir = path_arg(args, "code_dir")
    if code_dir is None:
        return error_payload("Missing required argument: code_dir")
    if not code_dir.exists() or not code_dir.is_dir():
        return error_payload(f"code_dir must be an existing directory: {code_dir}")

    query = str(args.get("query") or "")
    max_files = int(args.get("max_files") or 40)
    terms = [term.lower() for term in query.split() if term.strip()] or list(DEFAULT_TERMS)
    candidates = _candidate_code_files(code_dir, terms)

    limitations = []
    if len(candidates) > max_files:
        limitations.append(
            f"Found {len(candidates)} candidate code files; inspected {max_files}. Increase max_files for complete inspection."
        )
    inspected = []
    snippets = []
    for path in candidates[:max_files]:
        try:
            text = path.read_text(errors="replace")
        except Exception as exc:
            limitations.append(f"Could not read {path}: {type(exc).__name__}: {exc}")
            continue
        inspected.append(str(path))
        snippet_text = _matching_snippet(text, terms)
        if snippet_text:
            snippets.append(
                {
                    "path": str(path),
                    "text": snippet_text,
                }
            )

    return ok_payload(
        inspected_files=inspected,
        snippets=snippets,
        limitations=limitations,
        focus_terms=terms,
    )


def _candidate_code_files(code_dir: Path, terms: list[str]) -> list[Path]:
    scored = []
    for path in code_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in CODE_EXTENSIONS:
            continue
        rel = str(path.relative_to(code_dir)).lower()
        score = sum(2 for term in terms if term in rel)
        try:
            text = path.read_text(errors="replace").lower()
        except Exception:
            text = ""
        score += sum(1 for term in terms if term in text)
        if score > 0:
            scored.append((score, path))
    scored.sort(key=lambda item: (-item[0], str(item[1])))
    return [path for _, path in scored]


def _matching_snippet(text: str, terms: list[str]) -> str:
    lines = text.splitlines()
    selected = []
    for index, line in enumerate(lines, start=1):
        lowered = line.lower()
        if any(term in lowered for term in terms):
            selected.append(f"{index}: {line}")
        if len(selected) >= 40:
            break
    return "\n".join(selected)
