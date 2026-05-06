from __future__ import annotations

from typing import Any

from babysitter_hermes.artifacts import write_json

from ._common import error_payload, ok_payload, path_arg


def babysitter_save_artifact(args: dict[str, Any], **kwargs: Any) -> str:
    artifact_dir = path_arg(args, "artifact_dir")
    if artifact_dir is None:
        return error_payload("Missing required argument: artifact_dir")
    output_path = path_arg(args, "output_path")
    if output_path is None:
        return error_payload("Missing required argument: output_path")
    data = args.get("data")
    artifact_root = artifact_dir.resolve()
    resolved_output = output_path.resolve()
    if artifact_root != resolved_output and artifact_root not in resolved_output.parents:
        return error_payload(
            "output_path must be inside artifact_dir",
            artifact_dir=str(artifact_root),
            output_path=str(resolved_output),
        )
    try:
        saved = write_json(resolved_output, data)
    except Exception as exc:
        return error_payload(f"Artifact save failed: {type(exc).__name__}: {exc}")
    return ok_payload(output_path=str(saved))
