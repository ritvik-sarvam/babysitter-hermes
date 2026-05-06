from __future__ import annotations


SUBAGENT_TITLES = {
    "graph": "Graph Subagent",
    "kb": "KB Subagent",
    "runtime_logs": "Runtime Logs Subagent",
    "code": "Code Subagent",
    "data": "Data Subagent",
    "config": "Config Subagent",
    "fix_strategy": "Fix Strategy Subagent",
}

SUBAGENT_TOOL_SCOPES = {
    "graph": ["babysitter_render_graphs", "babysitter_analyze_graphs"],
    "kb": ["babysitter_retrieve_kb"],
    "runtime_logs": ["babysitter_read_logs"],
    "code": ["babysitter_inspect_code"],
    "data": ["babysitter_inspect_data"],
    "config": ["babysitter_wandb_snapshot", "babysitter_inspect_code"],
    "fix_strategy": ["babysitter_save_artifact"],
}


def build_scheduler_prompt(
    *,
    run_path: str,
    latest_step: int | None,
    artifact_dir: str,
    notify_severity: str,
    context: dict[str, object] | None = None,
) -> str:
    step = latest_step if latest_step is not None else "unknown"
    context_text = _context_text(context or {})
    return f"""You are Babysitter Hermes. Analyze W&B run {run_path} at step {step}.

Use the babysitter tools in this order unless evidence says a step is unnecessary:
1. Call babysitter_wandb_snapshot for full W&B rows, config, summary, and system metrics.
2. Call babysitter_render_graphs and use babysitter_analyze_graphs for graph-first diagnosis.
3. Call babysitter_retrieve_kb with the configured KB roots.
4. Call babysitter_read_logs for runtime logs.
5. Use babysitter_inspect_code and babysitter_inspect_data when symptoms point to code or data.
6. Use delegate_task for graph, KB, runtime logs, code, data, config, and fix-strategy subagents when an issue is plausible.
7. If severity is at least {notify_severity}, prepare a user message with babysitter_send_slack.

Artifact directory: {artifact_dir}

Configured evidence context:
{context_text}

Do not mutate code, data, configs, or training state.
Never recommend skipping datapoints; ask the user what to do instead.
Any code, data, hyperparameter, or training mutation requires explicit user approval.

Return a final concise diagnosis with severity, confidence, evidence, subagent findings,
questions for the user, proposed next actions, and whether user approval is required.

When delegating with delegate_task, use these specialist prompts exactly:
{all_subagent_prompts()}

After the final synthesis, call babysitter_save_artifact to save a structured diagnosis JSON
inside the artifact directory. Use this schema:
{final_diagnosis_schema_text()}
"""


def subagent_prompt(agent_name: str) -> str:
    title = SUBAGENT_TITLES.get(agent_name, f"{agent_name.replace('_', ' ').title()} Subagent")
    tools = SUBAGENT_TOOL_SCOPES.get(agent_name, [])
    tool_list = ", ".join(tools) if tools else "none"
    return f"""{title}

You are a specialist inside Babysitter Hermes. Stay within your scope and return only JSON
with these fields:
- agent_name
- severity: info, warning, or critical
- confidence: float from 0 to 1
- summary
- findings
- evidence
- questions_for_user
- proposed_next_actions
- requires_user_approval
- limitations

Allowed babysitter tools: {tool_list}

Never invent missing evidence.
Never recommend skipping datapoints; if a datapoint, batch, or example is suspicious,
add a clear question_for_user asking what to do.
Any mutation to code, data, config, hyperparameters, launch settings, or training state
requires user approval.
"""


def all_subagent_prompts() -> str:
    sections = []
    for agent_name in SUBAGENT_TOOL_SCOPES:
        sections.append(f"## {agent_name}\n{subagent_prompt(agent_name)}")
    return "\n\n".join(sections)


def final_diagnosis_schema_text() -> str:
    return """{
  "severity": "info|warning|critical",
  "confidence": 0.0,
  "summary": "user-facing diagnosis",
  "reasoning": "why the evidence supports this",
  "subagent_reports": [],
  "questions_for_user": [],
  "proposed_next_actions": [],
  "requires_user_approval": false
}"""


def _context_text(context: dict[str, object]) -> str:
    if not context:
        return "- None provided"
    lines = []
    for key, value in context.items():
        if isinstance(value, list):
            if value:
                rendered = ", ".join(str(item) for item in value)
            else:
                rendered = "none"
        elif isinstance(value, dict):
            rendered = ", ".join(f"{sub_key}: {sub_value}" for sub_key, sub_value in value.items())
        else:
            rendered = str(value)
        lines.append(f"- {key}: {rendered}")
    return "\n".join(lines)
