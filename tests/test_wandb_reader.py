from __future__ import annotations

from unittest.mock import patch

from babysitter_hermes.wandb_reader import fetch_run_snapshot


class FakeRun:
    state = "running"
    config = {}
    summary = {}

    def scan_history(self, page_size=None, keys=None):
        return [{"step": 1, "loss": 1.0}]

    def history(self, stream=None, pandas=False):
        raise RuntimeError("system history unavailable")


class FakeApi:
    def run(self, run_path):
        return FakeRun()


def test_fetch_run_snapshot_surfaces_system_history_failure_as_warning() -> None:
    with patch("wandb.Api", return_value=FakeApi()):
        snapshot = fetch_run_snapshot("entity/project/run")

    assert snapshot.system_rows == []
    assert any("system history" in warning.lower() for warning in snapshot.coverage.warnings)
