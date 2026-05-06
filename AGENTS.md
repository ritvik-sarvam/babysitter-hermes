# Agent Guide

## Project Purpose

`babysitter-hermes` launches optional user-provided training scripts in tmux,
monitors ML training runs by polling W&B cheaply, runs Claude triage, and
dispatches Hermes only for deeper agentic diagnosis when triage says action is
needed.

Claude owns the cheap first-pass health classification. Hermes owns deeper
reasoning and delegation. Python should launch, gather evidence, save artifacts,
and keep the scheduler deterministic.

## Main Flow

1. `babysitter_hermes.training_launch` optionally starts a tmux session from `run.launch.script_path`.
2. `babysitter_hermes.scheduler` resolves a W&B run.
3. It polls latest W&B step/state.
4. It creates an interval artifact directory when analysis is due.
5. It fetches W&B history, renders graphs, reads configured logs, and runs Claude triage.
6. It sends a message for non-healthy triage.
7. It invokes Hermes only when triage says `needs_action`.

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
- Keep scheduler logic deterministic; put cheap classification in Claude triage and deep reasoning in Hermes.

## Defaults

- Hermes model: `anthropic/claude-opus-4-7`.
- Required runtime key for default model: `ANTHROPIC_API_KEY`.
- Required W&B key: `WANDB_API_KEY`.
- Slack keys are only required for `slack_mode: send`.
- Setup script: `./setup_babysitter_hermes.sh`.

## Known TODOs

- Replace shallow KB markdown retrieval with symptom-grounded retrieval.
- Harden Slack and data-inspection tool error handling so invalid inputs always return JSON.
