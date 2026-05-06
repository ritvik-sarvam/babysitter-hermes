# Gemini Context

Follow `AGENTS.md` as the source of truth for this repository.

Important reminders:

- `babysitter-hermes` is standalone and uses Claude triage before Hermes escalation.
- The scheduler should stay deterministic and cheap.
- The default Hermes model is `anthropic/claude-opus-4-7`.
- Do not add non-Claude provider requirements to the default path.
- Do not skip datapoints. Ask the user before dropping, filtering, or ignoring any data.
- Do not mutate training code, configs, data, or launch state without explicit user approval.
- Launch training only through the configured tmux bash script; Hermes must not launch training.
- Healthy triage sends no message. Suspicious/error/unknown sends a message. Needs_action sends a message and invokes Hermes.

Run tests with:

```bash
uv run python -m pytest tests -q
```

Set up a fresh machine with:

```bash
./setup_babysitter_hermes.sh
```
