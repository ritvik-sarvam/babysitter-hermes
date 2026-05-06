#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERMES_HOME="${HERMES_HOME:-${HOME}/.hermes}"
HERMES_PLUGIN_DIR="${HERMES_PLUGIN_DIR:-${HERMES_HOME}/plugins/babysitter}"
HERMES_MODEL="${HERMES_MODEL:-anthropic/claude-opus-4-7}"
HERMES_PROFILE="${HERMES_PROFILE:-default}"
RUN_HERMES_SETUP="${RUN_HERMES_SETUP:-0}"
SKIP_HERMES_INSTALL="${SKIP_HERMES_INSTALL:-0}"
SKIP_PLUGIN_ENABLE="${SKIP_PLUGIN_ENABLE:-0}"
SKIP_DOCTOR="${SKIP_DOCTOR:-0}"

usage() {
  cat <<'EOF'
Usage: ./setup_babysitter_hermes.sh [options]

Installs the local Babysitter Hermes project, Hermes Agent, and the Hermes
babysitter plugin shim.

Options:
  --run-hermes-setup      Run the interactive `hermes setup` wizard.
  --skip-hermes-install   Do not install Hermes if `hermes` is missing.
  --skip-plugin-enable    Install plugin files but do not run `hermes plugins enable babysitter`.
  --skip-doctor           Do not run `hermes doctor` at the end.
  -h, --help              Show this help.

Environment:
  HERMES_MODEL            Defaults to anthropic/claude-opus-4-7.
  HERMES_PROFILE          Defaults to default.
  HERMES_HOME             Defaults to ~/.hermes.
  HERMES_PLUGIN_DIR       Defaults to ~/.hermes/plugins/babysitter.
  ANTHROPIC_API_KEY       Required by the default Claude model at runtime.
  WANDB_API_KEY           Required for W&B polling at runtime.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-hermes-setup)
      RUN_HERMES_SETUP=1
      ;;
    --skip-hermes-install)
      SKIP_HERMES_INSTALL=1
      ;;
    --skip-plugin-enable)
      SKIP_PLUGIN_ENABLE=1
      ;;
    --skip-doctor)
      SKIP_DOCTOR=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

log() {
  printf '[setup] %s\n' "$*"
}

warn() {
  printf '[setup][warn] %s\n' "$*" >&2
}

die() {
  printf '[setup][error] %s\n' "$*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

ensure_curl() {
  have curl || die "curl is required to install uv/Hermes. Install curl and rerun this script."
}

refresh_path() {
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${HOME}/bin:${PATH}"
}

ensure_uv() {
  refresh_path
  if have uv; then
    log "uv found: $(command -v uv)"
    return
  fi

  ensure_curl
  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  refresh_path
  have uv || die "uv install finished, but uv is still not on PATH. Restart your shell or add ~/.local/bin to PATH."
}

install_python_and_project() {
  log "Ensuring Python 3.11 is available through uv..."
  uv python install 3.11

  log "Installing Babysitter Hermes dependencies with uv sync..."
  (cd "${SCRIPT_DIR}" && uv sync)
}

ensure_hermes() {
  refresh_path
  if have hermes; then
    log "Hermes found: $(command -v hermes)"
    return
  fi

  if [[ "${SKIP_HERMES_INSTALL}" == "1" ]]; then
    die "Hermes is not installed and --skip-hermes-install was provided."
  fi

  ensure_curl
  log "Installing Hermes Agent with the official NousResearch installer..."
  curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
  refresh_path
  have hermes || die "Hermes install finished, but hermes is still not on PATH. Restart your shell and rerun this script."
}

configure_hermes() {
  if [[ "${RUN_HERMES_SETUP}" == "1" ]]; then
    log "Running interactive Hermes setup..."
    hermes setup
  fi

  log "Setting Hermes model to ${HERMES_MODEL}..."
  if ! hermes config set model "${HERMES_MODEL}"; then
    warn "Could not set Hermes model automatically. Run: hermes model"
  fi

  log "Using Hermes profile '${HERMES_PROFILE}' for Babysitter Hermes configs."
}

install_babysitter_plugin() {
  log "Installing Hermes babysitter plugin shim into ${HERMES_PLUGIN_DIR}..."
  mkdir -p "${HERMES_PLUGIN_DIR}"
  cp "${SCRIPT_DIR}/babysitter_hermes/plugin/plugin.yaml" "${HERMES_PLUGIN_DIR}/plugin.yaml"

  cat > "${HERMES_PLUGIN_DIR}/__init__.py" <<EOF
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(r"""${SCRIPT_DIR}""")
_VENV = _REPO_ROOT / ".venv"

for _site_packages in sorted(_VENV.glob("lib/python*/site-packages")):
    _site_packages_str = str(_site_packages)
    if _site_packages_str not in sys.path:
        sys.path.insert(0, _site_packages_str)

_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

from babysitter_hermes.plugin import register as _register


def register(ctx):
    return _register(ctx)
EOF

  if [[ "${SKIP_PLUGIN_ENABLE}" == "1" ]]; then
    warn "Plugin shim installed but not enabled. Enable later with: hermes plugins enable babysitter"
    return
  fi

  log "Enabling Hermes babysitter plugin..."
  if ! hermes plugins enable babysitter; then
    warn "Could not enable plugin automatically. Enable later with: hermes plugins enable babysitter"
  fi
}

check_runtime_environment() {
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    warn "ANTHROPIC_API_KEY is not set. The default Claude model will need it at runtime unless Hermes auth is configured another way."
  fi

  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    warn "WANDB_API_KEY is not set. W&B polling will need it at runtime."
  fi
}

run_doctor() {
  if [[ "${SKIP_DOCTOR}" == "1" ]]; then
    return
  fi

  log "Running hermes doctor..."
  if ! hermes doctor; then
    warn "hermes doctor reported issues. Fix those before relying on long-running babysitter jobs."
  fi
}

main() {
  log "Setting up Babysitter Hermes from ${SCRIPT_DIR}"
  ensure_uv
  install_python_and_project
  ensure_hermes
  configure_hermes
  install_babysitter_plugin
  check_runtime_environment
  run_doctor

  log "Setup finished."
  log "Verify the package with: uv run python -m pytest tests -q"
  log "Start the babysitter with: ./run_babysitter_hermes.sh config.yaml"
}

main
