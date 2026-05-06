#!/usr/bin/env bash
set -euo pipefail

# Fill these in on the machine that runs the babysitter.
# Do not commit real secret values.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export SLACK_BOT_TOKEN="${SLACK_BOT_TOKEN:-}"
export SLACK_USER_ID="${SLACK_USER_ID:-}"

# Babysitter Hermes defaults to Claude via Hermes:
#   anthropic/claude-opus-4.5
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"

# Optional model/runtime defaults.
export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-claude-opus-4.5}"
export HERMES_PROFILE="${HERMES_PROFILE:-default}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Pass a config path as the first argument, or set CONFIG_FILE in the environment.
CONFIG_FILE="${1:-${CONFIG_FILE:-config.yaml}}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}" >&2
  echo "Usage: CONFIG_FILE=config.yaml ./run_babysitter_hermes.sh" >&2
  echo "   or: ./run_babysitter_hermes.sh path/to/config.yaml [babysitter-hermes args]" >&2
  exit 1
fi

uv sync
uv run babysitter-hermes --config "${CONFIG_FILE}" "$@"
