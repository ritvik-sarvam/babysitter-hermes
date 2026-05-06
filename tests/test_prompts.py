from __future__ import annotations

from babysitter_hermes.prompts import SUBAGENT_TOOL_SCOPES, build_scheduler_prompt, subagent_prompt
from babysitter_hermes.synthesis import diagnosis_to_markdown, save_final_diagnosis
from babysitter_hermes.tools.schemas import TOOL_SCHEMAS


def test_scheduler_prompt_instructs_hermes_to_use_tools_and_not_mutate() -> None:
    prompt = build_scheduler_prompt(
        run_path="entity/project/run",
        latest_step=125,
        artifact_dir="/tmp/artifacts",
        notify_severity="warning",
    )

    assert "entity/project/run" in prompt
    assert "step 125" in prompt
    assert "Do not mutate code, data, configs, or training state" in prompt
    assert "Never recommend skipping datapoints" in prompt
    assert "babysitter_wandb_snapshot" in prompt
    assert "delegate_task" in prompt
    assert "When delegating with delegate_task" in prompt
    assert "Fix Strategy Subagent" in prompt
    assert "babysitter_save_artifact" in prompt


def test_subagent_prompt_defines_json_contract_and_scope() -> None:
    prompt = subagent_prompt("data")

    assert "Data Subagent" in prompt
    assert "Allowed babysitter tools" in prompt
    assert "babysitter_inspect_data" in prompt
    assert "questions_for_user" in prompt
    assert "requires_user_approval" in prompt
    assert "Never recommend skipping" in prompt


def test_all_expected_subagents_have_tool_scopes() -> None:
    assert set(SUBAGENT_TOOL_SCOPES) == {
        "graph",
        "kb",
        "runtime_logs",
        "code",
        "data",
        "config",
        "fix_strategy",
    }


def test_tool_schemas_expose_expected_babysitter_tools() -> None:
    names = {schema["name"] for schema in TOOL_SCHEMAS}

    assert {
        "babysitter_wandb_status",
        "babysitter_wandb_snapshot",
        "babysitter_render_graphs",
        "babysitter_retrieve_kb",
        "babysitter_read_logs",
        "babysitter_inspect_code",
        "babysitter_inspect_data",
        "babysitter_send_slack",
    }.issubset(names)


def test_final_diagnosis_markdown_includes_subagents_and_questions() -> None:
    markdown = diagnosis_to_markdown(
        {
            "severity": "warning",
            "confidence": 0.7,
            "summary": "Reward collapsed after step 50.",
            "subagent_reports": [
                {
                    "agent_name": "data",
                    "severity": "warning",
                    "confidence": 0.8,
                    "summary": "Some prompts are empty.",
                    "questions_for_user": ["A row has an empty prompt; what should be done?"],
                    "requires_user_approval": True,
                }
            ],
            "questions_for_user": ["Should we inspect the source dataset?"],
            "proposed_next_actions": ["Ask user before changing data handling."],
            "requires_user_approval": True,
        },
        artifact_dir="/tmp/artifacts",
    )

    assert "Reward collapsed after step 50." in markdown
    assert "data: Some prompts are empty." in markdown
    assert "Should we inspect the source dataset?" in markdown
    assert "Requires user approval before mutation: True" in markdown
    assert "/tmp/artifacts" in markdown


def test_save_final_diagnosis_writes_json_and_markdown(tmp_path) -> None:
    paths = save_final_diagnosis(
        {
            "severity": "info",
            "confidence": 0.5,
            "summary": "Run is healthy.",
            "proposed_next_actions": [],
        },
        output_dir=tmp_path,
        artifact_dir=str(tmp_path),
    )

    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert "Run is healthy." in paths["markdown"].read_text()
