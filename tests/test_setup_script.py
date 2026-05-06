from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETUP_SCRIPT = ROOT / "setup_babysitter_hermes.sh"


def test_setup_script_installs_core_runtime_and_hermes() -> None:
    script = SETUP_SCRIPT.read_text()

    assert SETUP_SCRIPT.stat().st_mode & 0o111
    assert "set -euo pipefail" in script
    assert "https://astral.sh/uv/install.sh" in script
    assert "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh" in script
    assert "uv sync" in script
    assert "hermes doctor" in script


def test_setup_script_keeps_claude_default_without_other_provider_requirements() -> None:
    script = SETUP_SCRIPT.read_text()

    assert "anthropic/claude-opus-4-7" in script
    assert "ANTHROPIC_API_KEY" in script
    assert "WANDB_API_KEY" in script
    assert "OPENAI_API_KEY" not in script
    assert "OPENROUTER_API_KEY" not in script


def test_setup_script_installs_and_enables_babysitter_plugin() -> None:
    script = SETUP_SCRIPT.read_text()

    assert "HERMES_PLUGIN_DIR" in script
    assert "babysitter_hermes.plugin" in script
    assert "hermes plugins enable babysitter" in script
    assert "plugin.yaml" in script
