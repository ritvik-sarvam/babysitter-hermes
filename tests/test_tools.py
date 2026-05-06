from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from babysitter.models import MetricCoverage, RunSnapshot

from babysitter_hermes.plugin import HANDLERS
from babysitter_hermes.tools import artifact_tools, code_tools, data_tools, kb_tools, log_tools, slack_tools, wandb_tools


def test_wandb_status_handler_returns_json_error_for_missing_run_path() -> None:
    payload = json.loads(wandb_tools.babysitter_wandb_status({}))

    assert payload["ok"] is False
    assert "run_path" in payload["error"]


def test_wandb_snapshot_handler_preserves_snapshot_artifacts(tmp_path: Path) -> None:
    snapshot = RunSnapshot(
        run_path="entity/project/run",
        state="running",
        training_rows=[{"step": 1, "loss": 3.0}],
        coverage=MetricCoverage(training_rows=1, first_step=1, latest_step=1),
    )

    with patch("babysitter_hermes.tools.wandb_tools.fetch_run_snapshot", return_value=snapshot):
        payload = json.loads(
            wandb_tools.babysitter_wandb_snapshot(
                {
                    "run_path": "entity/project/run",
                    "output_dir": str(tmp_path),
                    "artifact_dir": str(tmp_path),
                }
            )
        )

    assert payload["ok"] is True
    assert Path(payload["snapshot_path"]).exists()
    assert Path(payload["training_rows_path"]).exists()
    assert payload["coverage"]["latest_step"] == 1


def test_log_tool_reports_unreadable_or_missing_paths_as_limitations(tmp_path: Path) -> None:
    missing = tmp_path / "missing.log"

    payload = json.loads(log_tools.babysitter_read_logs({"log_paths": [str(missing)]}))

    assert payload["ok"] is True
    assert payload["excerpts"] == []
    assert any(str(missing) in item for item in payload["limitations"])


def test_log_tool_reads_only_tail_within_byte_budget(tmp_path: Path) -> None:
    log_file = tmp_path / "train.log"
    log_file.write_text("a" * 10000 + "\nERROR final failure\n")

    payload = json.loads(
        log_tools.babysitter_read_logs(
            {"log_paths": [str(log_file)], "max_bytes": 200, "max_files": 1}
        )
    )

    assert payload["ok"] is True
    assert payload["excerpts"][0]["bytes_read"] <= 200
    assert "ERROR final failure" in payload["excerpts"][0]["tail"]


def test_code_tool_finds_relevant_training_files(tmp_path: Path) -> None:
    train_file = tmp_path / "train.py"
    train_file.write_text(
        """
import wandb

def train():
    loss = 1.0
    wandb.log({"loss": loss})
"""
    )

    payload = json.loads(
        code_tools.babysitter_inspect_code(
            {"code_dir": str(tmp_path), "query": "wandb loss", "max_files": 5}
        )
    )

    assert payload["ok"] is True
    assert payload["inspected_files"] == [str(train_file)]
    assert "wandb.log" in payload["snippets"][0]["text"]


def test_data_tool_turns_suspicious_rows_into_user_questions(tmp_path: Path) -> None:
    data_file = tmp_path / "data.jsonl"
    data_file.write_text(
        """
{"prompt": "short", "completion": "ok"}
{"prompt": "", "completion": "missing prompt"}
not-json
""".lstrip()
    )

    payload = json.loads(
        data_tools.babysitter_inspect_data(
            {"data_paths": [str(data_file)], "max_preview_rows": 10}
        )
    )

    assert payload["ok"] is True
    assert payload["questions_for_user"]
    assert not any("skip" in question.lower() for question in payload["questions_for_user"])


def test_slack_tool_saves_dry_run_payload(tmp_path: Path) -> None:
    payload = json.loads(
        slack_tools.babysitter_send_slack(
            {
                "report": {
                    "severity": "warning",
                    "confidence": 0.8,
                    "summary": "Loss spiked.",
                    "requires_user_approval": True,
                },
                "artifact_dir": str(tmp_path),
                "output_path": str(tmp_path / "payload.json"),
                "mode": "dry-run",
                "latest_step": 10,
            }
        )
    )

    assert payload["ok"] is True
    assert (tmp_path / "payload.json").exists()
    assert "dry-run" in payload["status"]


def test_plugin_registers_every_schema_handler() -> None:
    assert "babysitter_wandb_status" in HANDLERS
    assert "babysitter_save_artifact" in HANDLERS


def test_save_artifact_rejects_paths_outside_artifact_dir(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    artifact_dir = tmp_path / "artifacts"

    payload = json.loads(
        artifact_tools.babysitter_save_artifact(
            {
                "artifact_dir": str(artifact_dir),
                "output_path": str(outside),
                "data": {"x": 1},
            }
        )
    )

    assert payload["ok"] is False
    assert not outside.exists()


def test_snapshot_graph_and_slack_tools_reject_outputs_outside_artifacts(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    artifact_dir = tmp_path / "artifacts"

    snapshot_payload = json.loads(
        wandb_tools.babysitter_wandb_snapshot(
            {
                "run_path": "entity/project/run",
                "output_dir": str(outside),
                "artifact_dir": str(artifact_dir),
            }
        )
    )
    graph_payload = json.loads(
        __import__(
            "babysitter_hermes.tools.graph_tools",
            fromlist=["babysitter_render_graphs"],
        ).babysitter_render_graphs(
            {
                "snapshot_path": str(tmp_path / "snapshot.json"),
                "output_dir": str(outside),
                "artifact_dir": str(artifact_dir),
            }
        )
    )
    slack_payload = json.loads(
        slack_tools.babysitter_send_slack(
            {
                "report": {"summary": "x"},
                "artifact_dir": str(artifact_dir),
                "output_path": str(outside / "payload.json"),
            }
        )
    )

    assert snapshot_payload["ok"] is False
    assert graph_payload["ok"] is False
    assert slack_payload["ok"] is False


def test_write_capable_tools_require_artifact_dir(tmp_path: Path) -> None:
    snapshot_payload = json.loads(
        wandb_tools.babysitter_wandb_snapshot(
            {"run_path": "entity/project/run", "output_dir": str(tmp_path)}
        )
    )
    graph_payload = json.loads(
        __import__(
            "babysitter_hermes.tools.graph_tools",
            fromlist=["babysitter_render_graphs"],
        ).babysitter_render_graphs(
            {"snapshot_path": str(tmp_path / "snapshot.json"), "output_dir": str(tmp_path)}
        )
    )

    assert snapshot_payload["ok"] is False
    assert "artifact_dir" in snapshot_payload["error"]
    assert graph_payload["ok"] is False
    assert "artifact_dir" in graph_payload["error"]


def test_kb_tracking_dir_must_stay_inside_artifact_dir(tmp_path: Path) -> None:
    payload = json.loads(
        kb_tools.babysitter_retrieve_kb(
            {
                "snapshot_path": str(tmp_path / "snapshot.json"),
                "tracking_dir": str(tmp_path / "outside"),
                "artifact_dir": str(tmp_path / "artifacts"),
            }
        )
    )

    assert payload["ok"] is False
    assert "artifact_dir" in payload["error"]
