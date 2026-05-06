from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import babysitter.kb.retrieve as retrieve_module
from babysitter.wandb_reader import load_snapshot

from ._common import ensure_within_artifact_dir, error_payload, list_path_arg, ok_payload, path_arg


def babysitter_retrieve_kb(args: dict[str, Any], **kwargs: Any) -> str:
    snapshot_path = path_arg(args, "snapshot_path")
    if snapshot_path is None:
        return error_payload("Missing required argument: snapshot_path")

    kb_roots = list_path_arg(args, "kb_roots")
    top_k = int(args.get("top_k") or retrieve_module.DEFAULT_TOP_K_CHUNKS)
    tracking_dir = path_arg(args, "tracking_dir")
    artifact_dir = path_arg(args, "artifact_dir")
    if tracking_dir is not None:
        if artifact_dir is None:
            return error_payload("artifact_dir is required when tracking_dir is provided")
        allowed, error = ensure_within_artifact_dir(
            artifact_dir=artifact_dir,
            target_path=tracking_dir,
        )
        if not allowed:
            return error_payload(error or "tracking_dir is outside artifact_dir")

    try:
        snapshot = load_snapshot(snapshot_path)
        passages = []
        limitations = []
        if kb_roots:
            original_kb_root = retrieve_module._kb_root
            try:
                for kb_root in kb_roots:
                    if not kb_root.exists():
                        limitations.append(f"KB root does not exist: {kb_root}")
                        continue
                    retrieve_module._kb_root = lambda root=kb_root: root
                    passages.extend(
                        asyncio.run(
                            retrieve_module.retrieve_kb_for_snapshot(
                                snapshot,
                                top_k=top_k,
                                tracking_dir=tracking_dir,
                            )
                        )
                    )
            finally:
                retrieve_module._kb_root = original_kb_root
        else:
            passages = asyncio.run(
                retrieve_module.retrieve_kb_for_snapshot(
                    snapshot,
                    top_k=top_k,
                    tracking_dir=tracking_dir,
                )
            )
            limitations = []
    except Exception as exc:
        return error_payload(f"KB retrieval failed: {type(exc).__name__}: {exc}")

    return ok_payload(
        passages=[
            {
                "chunk_id": passage.chunk_id,
                "citation": passage.citation(),
                "parent_doc_id": passage.parent_doc_id,
                "parent_title": passage.parent_title,
                "heading_path": passage.heading_path,
                "text": passage.text,
            }
            for passage in passages
        ],
        limitations=limitations,
    )
