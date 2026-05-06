# Babysitter Hermes

`babysitter-hermes` is a Hermes-native training babysitter. A small scheduler polls
Weights & Biases for cheap run status, then dispatches Hermes at configured step
intervals or terminal states. Hermes owns the agentic loop and uses the
`babysitter_*` tools to gather metrics, graphs, KB evidence, logs, code context,
data context, and Slack-ready user messages.

The first version monitors existing W&B runs. Training launch can be added later
as another Hermes tool so the agent can reason about launch and status in the
same loop.

## Install

```bash
git clone https://github.com/ritvik-sarvam/babysitter-hermes.git
cd babysitter-hermes
uv sync
```

## Verify

```bash
uv run python -c "import babysitter_hermes; print(babysitter_hermes.__version__)"
uv run python -m pytest tests -q
```

## Run

Create a config file:

```yaml
project_id: demo
workdir: ./babysitter_hermes_runs
hermes:
  model: anthropic/claude-opus-4.5
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

The package is self-contained; it does not require being cloned inside the old
`optimus_training` monorepo.

By default, Babysitter Hermes invokes Hermes with Claude:
`anthropic/claude-opus-4.5`. Set `ANTHROPIC_API_KEY` before running, or override
`hermes.model` in your config if you intentionally want a different Claude model.
