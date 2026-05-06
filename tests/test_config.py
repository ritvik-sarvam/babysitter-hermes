from __future__ import annotations

from pathlib import Path

import pytest

from babysitter_hermes.config import load_config
from babysitter_hermes.artifacts import redact_secrets


def test_load_config_resolves_paths_relative_to_config_file(tmp_path: Path) -> None:
    config_path = tmp_path / "babysitter-hermes.yaml"
    config_path.write_text(
        """
project_id: wiki-search-babysitter
workdir: ./runs
hermes:
  command: hermes
  profile: default
  model: anthropic/claude-opus-4.5
run:
  wandb_run_file: ./wandb_run.txt
  training_code_dir: ./code
  kb_roots:
    - ./kb
  log_paths:
    - ./logs
  data_paths:
    - ./data
"""
    )

    config = load_config(config_path)

    assert config.project_id == "wiki-search-babysitter"
    assert config.workdir == tmp_path / "runs"
    assert config.run.wandb_run_file == tmp_path / "wandb_run.txt"
    assert config.run.training_code_dir == tmp_path / "code"
    assert config.run.kb_roots == [tmp_path / "kb"]
    assert config.run.log_paths == [tmp_path / "logs"]
    assert config.run.data_paths == [tmp_path / "data"]
    assert config.monitor.monitor_every_steps == 25
    assert config.analysis.subagents.fix_strategy is True


def test_load_config_requires_exactly_one_wandb_source(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
project_id: bad
workdir: ./runs
run:
  wandb_run: entity/project/run
  wandb_run_file: ./wandb_run.txt
"""
    )

    with pytest.raises(ValueError, match="exactly one"):
        load_config(config_path)


def test_redact_secrets_replaces_sensitive_values() -> None:
    redacted = redact_secrets(
        {
            "WANDB_API_KEY": "wandb-secret",
            "nested": {"slack_bot_token": "xoxb-secret"},
            "safe": "visible",
        }
    )

    assert redacted["WANDB_API_KEY"] == "<redacted>"
    assert redacted["nested"]["slack_bot_token"] == "<redacted>"
    assert redacted["safe"] == "visible"
