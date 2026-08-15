from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from finsight.agent import LLMFinSightAgent, OpenAICompatibleActor
from finsight.audit import evaluate_result
from finsight.business_cases import load_business_cases
from finsight.environment import ActionName, AgentAction
from finsight.models import AgentState
from finsight.training.teacher import TeacherError


class RecordingPolicy:
    model_name = "recording-live-model"
    prompt_version = "test-live-prompt"
    schema_repairs = 0
    api_calls = 0
    total_latency_seconds = 0.0
    last_usage: dict[str, Any] = {}

    def __init__(self, actions: Sequence[AgentAction]) -> None:
        self.actions = list(actions)
        self.messages_seen: list[list[Mapping[str, Any]]] = []

    def next_action(self, messages: Sequence[Mapping[str, Any]]) -> AgentAction:
        self.messages_seen.append(list(messages))
        self.api_calls += 1
        return self.actions.pop(0)


def _action(name: ActionName, **arguments: Any) -> AgentAction:
    return AgentAction(name, arguments)


def test_live_actor_generates_program_without_hidden_program_in_initial_context():
    task = load_business_cases()[0]
    generated_program = "divide(multiply(subtract(1250,1100),const_100),1100)"
    policy = RecordingPolicy(
        [
            _action(ActionName.SEARCH, query=task.question, top_k=2),
            _action(ActionName.OPEN_EVIDENCE, evidence_id="income_statement"),
            _action(ActionName.SELECT_EVIDENCE, evidence_ids=["income_statement"]),
            _action(ActionName.EMIT_PROGRAM, program=generated_program),
            _action(ActionName.CALCULATE),
            _action(ActionName.VERIFY),
            _action(ActionName.DELIVER),
        ]
    )

    result = LLMFinSightAgent(policy).run(task)

    assert result.state == AgentState.DELIVERED
    assert result.verified
    assert result.formula == generated_program
    assert result.formula != task.program
    assert task.program not in str(policy.messages_seen[0])
    assert "expected_evidence_ids" not in str(policy.messages_seen[0])
    assert result.run_metadata["execution_mode"] == "live_llm_agent"
    assert result.run_metadata["api_calls"] == 7
    assert result.run_metadata["hidden_gold_exposed_to_actor"] is False
    assert evaluate_result(task, result)["trajectory_panel"]["state_path_valid"]


def test_live_actor_can_refuse_after_searching_for_missing_evidence():
    task = load_business_cases()[3]
    policy = RecordingPolicy(
        [
            _action(ActionName.SEARCH, query="2023 average inventory", top_k=2),
            _action(ActionName.ABSTAIN, reason_code="missing_evidence"),
        ]
    )

    result = LLMFinSightAgent(policy).run(task)

    assert result.state == AgentState.REFUSED
    assert result.answer is None
    assert result.run_metadata["termination_reason"] == "abstained:missing_evidence"
    evaluation = evaluate_result(task, result)
    assert evaluation["outcome_panel"]["correct_refusal"]
    assert evaluation["trajectory_panel"]["state_path_valid"]


def test_user_connection_builds_session_scoped_actor():
    actor = OpenAICompatibleActor.from_connection(
        provider_name="Custom Provider",
        base_url="https://provider.example/v1/",
        api_key="session-secret",
        model_name="custom-model",
        temperature=None,
        json_mode=False,
    )

    assert actor.provider_name == "Custom Provider"
    assert actor.model_name == "custom-model"
    assert actor._client.base_url == "https://provider.example/v1"
    assert actor._client.response_format is None
    assert actor._client.last_usage == {}


@pytest.mark.parametrize(
    ("base_url", "api_key", "model_name"),
    [
        ("", "secret", "model"),
        ("https://provider.example/v1", "", "model"),
        ("https://provider.example/v1", "secret", ""),
        ("provider.example/v1", "secret", "model"),
    ],
)
def test_user_connection_rejects_incomplete_or_invalid_settings(
    base_url: str,
    api_key: str,
    model_name: str,
):
    with pytest.raises(TeacherError):
        OpenAICompatibleActor.from_connection(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
        )
