# Babysitter Hermes

`babysitter-hermes` is a Hermes-native training babysitter. A small scheduler polls
Weights & Biases for cheap run status, then dispatches Hermes at configured step
intervals or terminal states. Hermes owns the agentic loop and uses the
`babysitter_*` tools to gather metrics, graphs, KB evidence, logs, code context,
data context, and Slack-ready user messages.

The first version monitors existing W&B runs. Training launch can be added later
as another Hermes tool so the agent can reason about launch and status in the
same loop.

## Current Status

This repository is standalone. It should work from a fresh clone with `uv sync`;
it does not need to live inside the old `optimus_training` monorepo.

The default model is Claude through Hermes:

```yaml
hermes:
  model: anthropic/claude-opus-4-7
```

Set `ANTHROPIC_API_KEY` before running, or override `hermes.model` in your config
if you intentionally want a different Claude model.

## Architecture

The scheduler is deliberately small and non-agentic:

1. Resolve a W&B run from `run.wandb_run` or `run.wandb_run_file`.
2. Poll cheap W&B status: latest step and terminal state.
3. Dispatch Hermes only when `monitor_every_steps` is reached or the run is terminal.
4. Persist interval artifacts and Hermes output under `workdir`.

Hermes is responsible for reasoning. The scheduler prompt tells Hermes to use the
`babysitter_*` tools and delegate specialist subagents for graphs, KB, runtime
logs, code, data, config, and fix strategy.

Important package areas:

- `babysitter_hermes/scheduler.py`: polling loop and Hermes dispatch.
- `babysitter_hermes/config.py`: YAML config schema and path resolution.
- `babysitter_hermes/prompts.py`: parent prompt, subagent scopes, final JSON schema.
- `babysitter_hermes/tools/`: Hermes tool handlers.
- `babysitter_hermes/plugin/`: Hermes plugin registration and bundled skill.
- `babysitter_hermes/kb_retrieve.py`: current lightweight KB retrieval.
- `tests/`: focused unit tests for config, scheduler, prompts, tools, and safety.

## Install

```bash
git clone https://github.com/ritvik-sarvam/babysitter-hermes.git
cd babysitter-hermes
./setup_babysitter_hermes.sh
```

The setup script installs or verifies:

- `uv` and Python 3.11 through `uv`.
- This package's Python dependencies via `uv sync`.
- Hermes Agent through the official NousResearch installer if `hermes` is missing.
- The local `babysitter` Hermes plugin shim in `~/.hermes/plugins/babysitter`.
- The Claude default model, `anthropic/claude-opus-4-7`.

If you want the interactive Hermes wizard during setup:

```bash
./setup_babysitter_hermes.sh --run-hermes-setup
```

## Verify

```bash
uv run python -c "import babysitter_hermes; print(babysitter_hermes.__version__)"
uv run python -m pytest tests -q
```

## Run

Create a config file. For local BabyLM autoresearch, start from the checked-in
`config.yaml` and update the W&B pointer:

```yaml
run:
  wandb_run_file: ../babylm-autoresearch/runs/<run_name>_wandb_run.txt
```

The BabyLM training script prints the pointer path:

```text
WandB run pointer: runs/<run_name>_wandb_run.txt
```

Minimal config shape:

```yaml
project_id: demo
workdir: ./babysitter_hermes_runs
hermes:
  model: anthropic/claude-opus-4-7
run:
  wandb_run: entity/project/run_id
  training_code_dir: ./training-code
  kb_roots:
    - ./kb
  log_paths:
    - ./logs
  data_paths:
    - ./data
```

Then start the scheduler:

```bash
uv run babysitter-hermes --config config.yaml
```

Or use the helper script and put your environment variables there:

```bash
chmod +x run_babysitter_hermes.sh
./run_babysitter_hermes.sh config.yaml
```

The scheduler logs each W&B poll, dispatch decision, Hermes analysis interval,
artifact directory, and sleep period. Increase verbosity with:

```bash
./run_babysitter_hermes.sh config.yaml --log-level DEBUG
```

Logs are also written into the configured `workdir`:

```text
babysitter_hermes_runs/<project_id>/0_scheduler_logs/scheduler.log
babysitter_hermes_runs/<project_id>/runs/<run>/intervals/<interval>/7_final_synthesis/hermes_logs/
babysitter_hermes_runs/<project_id>/runs/<run>/intervals/<interval>/7_final_synthesis/hermes_stream/
```

The interval `hermes_logs/` directory is a copy of Hermes' local `~/.hermes/logs`
files at the time that interval finishes.

During an analysis interval, Hermes stdout/stderr are streamed live to the
terminal. The same live stream is also saved as `hermes_stream/stdout.log` and
`hermes_stream/stderr.log`, while the final captured subprocess result is saved
in `7_final_synthesis/hermes_result.json`.

Babysitter Hermes invokes Hermes in non-interactive chat mode:

```bash
hermes chat --model <model> --toolsets <toolsets> -q "<scheduler prompt>"
```

The package is self-contained; it does not require being cloned inside the old
`optimus_training` monorepo.

## Environment Variables

Required for W&B:

```bash
export WANDB_API_KEY="..."
```

Required for the default Claude setup:

```bash
export ANTHROPIC_API_KEY="..."
```

Only needed if Slack send mode is enabled:

```bash
export SLACK_BOT_TOKEN="..."
export SLACK_USER_ID="..."
```

The helper script `run_babysitter_hermes.sh` is intentionally a template. Fill
the variables on the machine that runs the babysitter, but do not commit real
secret values.

## Setup Script Options

```bash
./setup_babysitter_hermes.sh --help
./setup_babysitter_hermes.sh --run-hermes-setup
./setup_babysitter_hermes.sh --skip-hermes-install
./setup_babysitter_hermes.sh --skip-plugin-enable
./setup_babysitter_hermes.sh --skip-doctor
```

The script prints warnings if `ANTHROPIC_API_KEY` or `WANDB_API_KEY` are missing.
It does not write real secret values into the repository.

## Safety Rules

- Do not skip datapoints. If data looks suspicious, ask the user what to do.
- Do not mutate code, data, configs, or training state without explicit approval.
- Tool handlers should return JSON payloads, including failures.
- Write-capable tools must stay inside the configured artifact directory.
- Keep W&B polling cheap; expensive analysis should happen inside Hermes intervals.

## Known Follow-Up TODOs

These are intentionally documented in code as TODO comments:

- Replace shallow KB markdown scanning with symptom-grounded retrieval.
- Return JSON errors for invalid Slack report input.
- Return JSON errors/limitations for malformed data files.
