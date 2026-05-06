# Babysitter Hermes

`babysitter-hermes` is a Hermes-native training babysitter. A small scheduler polls
Weights & Biases for cheap run status, then dispatches Hermes at configured step
intervals or terminal states. Hermes owns the agentic loop and uses the
`babysitter_*` tools to gather metrics, graphs, KB evidence, logs, code context,
data context, and Slack-ready user messages.

The first version monitors existing W&B runs. Training launch can be added later
as another Hermes tool so the agent can reason about launch and status in the
same loop.
