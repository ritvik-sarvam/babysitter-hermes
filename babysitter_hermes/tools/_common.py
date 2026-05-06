from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ok_payload(**data: Any) -> str:
    payload = {"ok": True}
    payload.update(data)
    return json.dumps(payload, indent=2, default=str)


def error_payload(message: str, **data: Any) -> str:
    payload = {"ok": False, "error": message}
    payload.update(data)
    return json.dumps(payload, indent=2, default=str)


def path_arg(args: dict[str, Any], name: str) -> Path | None:
    value = args.get(name)
    if not value:
        return None
    return Path(str(value)).expanduser()


def list_path_arg(args: dict[str, Any], name: str) -> list[Path]:
    values = args.get(name) or []
    return [Path(str(value)).expanduser() for value in values]


def ensure_within_artifact_dir(
    *,
    artifact_dir: Path | None,
    target_path: Path,
) -> tuple[bool, str | None]:
    if artifact_dir is None:
        return True, None
    artifact_root = artifact_dir.resolve()
    resolved = target_path.resolve()
    if artifact_root == resolved or artifact_root in resolved.parents:
        return True, None
    return (
        False,
        f"Path must be inside artifact_dir: artifact_dir={artifact_root}, path={resolved}",
    )
