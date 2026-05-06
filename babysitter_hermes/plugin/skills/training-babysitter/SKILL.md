---
name: training-babysitter
description: Monitor and diagnose ML training runs using Babysitter Hermes tools.
---

# Training Babysitter

Use this skill when asked to monitor, diagnose, or explain a training run.

## Workflow

1. Start with `babysitter_wandb_snapshot` to fetch complete W&B evidence for the current interval.
2. Use `babysitter_render_graphs` and `babysitter_analyze_graphs` for graph-first diagnosis.
3. Use `babysitter_retrieve_kb` to ground symptoms in the configured knowledge base.
4. Use `babysitter_read_logs` to inspect runtime, Hermes, tmux, Slurm, or local logs.
5. Use `babysitter_inspect_code` when code paths could explain the symptoms.
6. Use `babysitter_inspect_data` when data schema, formatting, or examples could explain the symptoms.
7. Delegate specialist subagents for graph, KB, runtime logs, code, data, config, and fix strategy when an issue is plausible.
8. Send or prepare Slack only after final synthesis.

## Rules

- Do not mutate code, data, configs, launch settings, or training state.
- Never recommend skipping datapoints. Ask the user what to do with any suspicious datapoint.
- Any proposed mutation requires explicit user approval.
- Cite artifact paths and evidence sources in the final answer.
- If a tool reports limitations, include them in the final diagnosis.
