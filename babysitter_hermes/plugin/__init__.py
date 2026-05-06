from __future__ import annotations

from pathlib import Path

from babysitter_hermes.tools import (
    artifact_tools,
    code_tools,
    data_tools,
    graph_tools,
    kb_tools,
    log_tools,
    slack_tools,
    wandb_tools,
)
from babysitter_hermes.tools.schemas import TOOL_SCHEMAS


HANDLERS = {
    "babysitter_wandb_status": wandb_tools.babysitter_wandb_status,
    "babysitter_wandb_snapshot": wandb_tools.babysitter_wandb_snapshot,
    "babysitter_render_graphs": graph_tools.babysitter_render_graphs,
    "babysitter_analyze_graphs": graph_tools.babysitter_analyze_graphs,
    "babysitter_retrieve_kb": kb_tools.babysitter_retrieve_kb,
    "babysitter_read_logs": log_tools.babysitter_read_logs,
    "babysitter_inspect_code": code_tools.babysitter_inspect_code,
    "babysitter_inspect_data": data_tools.babysitter_inspect_data,
    "babysitter_send_slack": slack_tools.babysitter_send_slack,
    "babysitter_save_artifact": artifact_tools.babysitter_save_artifact,
}


def register(ctx) -> None:
    for schema in TOOL_SCHEMAS:
        name = schema["name"]
        ctx.register_tool(
            name=name,
            toolset="babysitter",
            schema=schema,
            handler=HANDLERS[name],
        )

    skills_dir = Path(__file__).parent / "skills"
    for child in sorted(skills_dir.iterdir()):
        skill_md = child / "SKILL.md"
        if child.is_dir() and skill_md.exists():
            ctx.register_skill(child.name, skill_md)
