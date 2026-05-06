# Claude Context

This repo is a standalone Hermes plugin and scheduler for training babysitting.
Do not assume it is installed inside `optimus_training`.

## How To Work Here

- Read `AGENTS.md` first for the project flow and invariants.
- Prefer small, focused changes with tests for scheduler, config, prompts, and tools.
- Preserve Claude as the default Hermes model unless the user explicitly asks to change it.
- Do not add OpenAI/OpenRouter requirements for the default path.
- Do not skip datapoints or silently ignore malformed data. Ask the user what to do.
- Do not implement mutations to training code, configs, data, or launch state without user approval.

## Verification

```bash
uv run python -m pytest tests -q
```

For config-only changes, also check that YAML loads through `BabysitterConfig.load`.
For machine setup, use `./setup_babysitter_hermes.sh`.

## Key Files

- `babysitter_hermes/config.py`: config schema and path resolution.
- `babysitter_hermes/scheduler.py`: cheap polling and Hermes dispatch.
- `babysitter_hermes/prompts.py`: agent prompt contracts.
- `babysitter_hermes/tools/`: Hermes tool implementations.
- `babysitter_hermes/plugin/skills/training-babysitter/SKILL.md`: Hermes-facing skill.
