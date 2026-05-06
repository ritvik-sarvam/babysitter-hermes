from __future__ import annotations


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


BABYSITTER_WANDB_STATUS = _schema(
    "babysitter_wandb_status",
    "Fetch the latest W&B step and run state. Use this for cheap scheduler/status checks.",
    {
        "run_path": {
            "type": "string",
            "description": "Exact W&B run path in entity/project/run_id form.",
        }
    },
    ["run_path"],
)

BABYSITTER_WANDB_SNAPSHOT = _schema(
    "babysitter_wandb_snapshot",
    "Fetch full W&B training rows, system rows, config, summary, and coverage into artifacts.",
    {
        "run_path": {"type": "string"},
        "output_dir": {"type": "string", "description": "Artifact directory for snapshot files."},
        "artifact_dir": {"type": "string", "description": "Root artifact directory allowed for writes."},
    },
    ["run_path", "output_dir", "artifact_dir"],
)

BABYSITTER_RENDER_GRAPHS = _schema(
    "babysitter_render_graphs",
    "Render full, recent, and zoom graph boards from a saved Babysitter snapshot.",
    {
        "snapshot_path": {"type": "string"},
        "output_dir": {"type": "string"},
        "artifact_dir": {"type": "string", "description": "Root artifact directory allowed for writes."},
        "recent_steps": {"type": "integer", "default": 100},
    },
    ["snapshot_path", "output_dir", "artifact_dir"],
)

BABYSITTER_ANALYZE_GRAPHS = _schema(
    "babysitter_analyze_graphs",
    "Analyze rendered graph boards with the existing graph-first diagnosis model.",
    {
        "snapshot_path": {"type": "string"},
        "full_board": {"type": "string"},
        "recent_board": {"type": "string"},
        "zoom_board": {"type": "string"},
        "metadata_path": {"type": "string"},
        "latest_step": {"type": "integer"},
        "suspect_window_rows": {"type": "array", "items": {"type": "object"}},
    },
    ["snapshot_path", "full_board", "recent_board", "zoom_board", "metadata_path"],
)

BABYSITTER_RETRIEVE_KB = _schema(
    "babysitter_retrieve_kb",
    "Retrieve KB passages for a snapshot using symptom tags and configured KB roots.",
    {
        "snapshot_path": {"type": "string"},
        "kb_roots": {"type": "array", "items": {"type": "string"}},
        "top_k": {"type": "integer", "default": 5},
        "tracking_dir": {"type": "string"},
        "artifact_dir": {"type": "string", "description": "Root artifact directory allowed for tracking writes."},
    },
    ["snapshot_path"],
)

BABYSITTER_READ_LOGS = _schema(
    "babysitter_read_logs",
    "Read configured training, Hermes, Slurm, tmux, or local logs and surface runtime signals.",
    {
        "log_paths": {"type": "array", "items": {"type": "string"}},
        "run_hint": {"type": "string"},
        "max_bytes": {"type": "integer", "default": 200000},
        "max_files": {"type": "integer", "default": 20},
    },
    ["log_paths"],
)

BABYSITTER_INSPECT_CODE = _schema(
    "babysitter_inspect_code",
    "Inspect training code read-only for entrypoints, data loading, optimizer/loss/reward logic, and W&B logging.",
    {
        "code_dir": {"type": "string"},
        "query": {"type": "string"},
        "max_files": {"type": "integer", "default": 40},
    },
    ["code_dir"],
)

BABYSITTER_INSPECT_DATA = _schema(
    "babysitter_inspect_data",
    "Inspect configured data files or directories and ask the user about suspicious datapoints instead of recommending removal.",
    {
        "data_paths": {"type": "array", "items": {"type": "string"}},
        "max_preview_rows": {"type": "integer", "default": 50},
    },
    ["data_paths"],
)

BABYSITTER_SEND_SLACK = _schema(
    "babysitter_send_slack",
    "Prepare or send a Slack message for a final Babysitter Hermes diagnosis.",
    {
        "report": {"type": "object"},
        "wandb_run_url": {"type": "string"},
        "latest_step": {"type": "integer"},
        "artifact_dir": {"type": "string"},
        "output_path": {"type": "string"},
        "mode": {"type": "string", "enum": ["dry-run", "send"], "default": "dry-run"},
        "slack_user_id": {"type": "string"},
    },
    ["report", "artifact_dir"],
)

BABYSITTER_SAVE_ARTIFACT = _schema(
    "babysitter_save_artifact",
    "Persist structured Babysitter Hermes data into the configured artifact tree.",
    {
        "artifact_dir": {"type": "string", "description": "Root artifact directory allowed for writes."},
        "output_path": {"type": "string"},
        "data": {"type": "object"},
    },
    ["artifact_dir", "output_path", "data"],
)

TOOL_SCHEMAS = [
    BABYSITTER_WANDB_STATUS,
    BABYSITTER_WANDB_SNAPSHOT,
    BABYSITTER_RENDER_GRAPHS,
    BABYSITTER_ANALYZE_GRAPHS,
    BABYSITTER_RETRIEVE_KB,
    BABYSITTER_READ_LOGS,
    BABYSITTER_INSPECT_CODE,
    BABYSITTER_INSPECT_DATA,
    BABYSITTER_SEND_SLACK,
    BABYSITTER_SAVE_ARTIFACT,
]
