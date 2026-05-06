# Agent Guide

## Project Purpose

`babysitter-hermes` monitors ML training runs by polling W&B cheaply and
dispatching Hermes for deeper agentic diagnosis at configured intervals.

Hermes owns reasoning and delegation. Python should expose deterministic tools,
save artifacts, and keep the scheduler simple.

## Main Flow

1. `babysitter_hermes.scheduler` resolves a W&B run.
2. It polls latest W&B step/state.
3. It creates an interval artifact directory when analysis is due.
4. It prompts Hermes to use `babysitter_*` tools.
5. Hermes may delegate graph, KB, runtime log, code, data, config, and fix-strategy subagents.

## Commands

```bash
./setup_babysitter_hermes.sh
uv sync
uv run python -m pytest tests -q
uv run babysitter-hermes --config config.yaml
```

## Invariants

- Never skip or drop datapoints. Suspicious data becomes a question for the user.
- Never mutate code, data, configs, launch settings, or training state without explicit approval.
- Tool handlers should return JSON strings for both success and failure.
- Write-capable tools must constrain outputs to the artifact directory.
- Keep scheduler logic deterministic; put reasoning in Hermes prompts/tools.

## Defaults

- Hermes model: `anthropic/claude-opus-4-7`.
- Required runtime key for default model: `ANTHROPIC_API_KEY`.
- Required W&B key: `WANDB_API_KEY`.
- Slack keys are only required for `slack_mode: send`.
- Setup script: `./setup_babysitter_hermes.sh`.

## Known TODOs

- Replace shallow KB markdown retrieval with symptom-grounded retrieval.
- Harden Slack and data-inspection tool error handling so invalid inputs always return JSON.
