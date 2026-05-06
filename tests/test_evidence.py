from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from babysitter_hermes.evidence import collect_interval_evidence
from babysitter_hermes.models import MetricCoverage, RunSnapshot, WandbStatus


def test_collect_interval_evidence_saves_snapshot_graphs_and_logs(tmp_path: Path) -> None:
    log_file = tmp_path / "train.log"
    log_file.write_text("step 1 ok\nstep 2 loss spike\n")
    snapshot = RunSnapshot(
        run_path="entity/project/run",
        state="running",
        training_rows=[{"step": 1, "loss": 2.0}, {"step": 2, "loss": 3.5}],
        coverage=MetricCoverage(training_rows=2, first_step=1, latest_step=2),
    )

    with patch("babysitter_hermes.evidence.fetch_run_snapshot", return_value=snapshot):
        evidence = collect_interval_evidence(
            run_path="entity/project/run",
            status=WandbStatus(run_path="entity/project/run", latest_step=2, state="running"),
            interval_dir=tmp_path / "interval",
            recent_steps=10,
            log_paths=[log_file],
            max_log_bytes=1000,
        )

    assert Path(evidence["snapshot_path"]).exists()
    assert Path(evidence["coverage_path"]).exists()
    assert Path(evidence["graph_paths"]["full_board"]).exists()
    assert Path(evidence["log_summary_path"]).exists()
    assert evidence["coverage"]["latest_step"] == 2

    log_summary = json.loads(Path(evidence["log_summary_path"]).read_text())
    assert log_summary["ok"] is True
    assert log_summary["excerpts"]
