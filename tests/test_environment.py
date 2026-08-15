from finsight.business_cases import load_business_cases
from finsight.environment import ActionName, AgentAction, FinSightEnvironment
from finsight.models import EventStatus


def _action(name: ActionName, **arguments) -> AgentAction:
    return AgentAction(name=name, arguments=arguments)


def test_environment_executes_a_replayable_verified_trajectory():
    task = load_business_cases()[0]
    environment = FinSightEnvironment(task)
    initial = environment.initial_observation()

    assert "expected_evidence_ids" not in initial.payload
    assert "program" not in initial.payload

    search = environment.step(_action(ActionName.SEARCH, query=task.question, top_k=2))
    assert search.status == EventStatus.COMPLETED
    assert "expected_evidence_ids" not in search.payload

    environment.step(_action(ActionName.OPEN_EVIDENCE, evidence_id="income_statement"))
    environment.step(
        _action(ActionName.SELECT_EVIDENCE, evidence_ids=["income_statement"])
    )
    environment.step(_action(ActionName.EMIT_PROGRAM, program=task.program))
    calculation = environment.step(_action(ActionName.CALCULATE))
    verification = environment.step(_action(ActionName.VERIFY))
    delivery = environment.step(_action(ActionName.DELIVER))

    assert calculation.payload["value"] > 0
    assert verification.payload["passed"]
    assert delivery.done
    assert delivery.payload["evidence_ids"] == ["income_statement"]
    trajectory = environment.trajectory()
    assert trajectory["termination_reason"] == "delivered"
    assert trajectory["context"]["estimated_tokens_used"] > 0
    assert not trajectory["context"]["context_overflow"]
    assert [event["event_id"] for event in trajectory["events"]] == [
        f"e{index:04d}" for index in range(1, 8)
    ]


def test_action_guard_blocks_delivery_before_verification():
    environment = FinSightEnvironment(load_business_cases()[0])
    observation = environment.step(_action(ActionName.DELIVER))

    assert observation.status == EventStatus.REJECTED
    assert not observation.done
    assert observation.payload["reason_code"] == "action_rejected"


def test_invalid_action_schema_is_rejected_without_running_code():
    environment = FinSightEnvironment(load_business_cases()[0])
    observation = environment.step({"name": "shell", "arguments": {"cmd": "pwd"}})

    assert observation.status == EventStatus.REJECTED
    assert observation.payload["reason_code"] == "invalid_action"


def test_exact_repeat_terminates_no_progress_loop():
    environment = FinSightEnvironment(load_business_cases()[0])
    action = _action(ActionName.SEARCH, query="irrelevant tokens", top_k=1)
    environment.step(action)
    repeated = environment.step(action)

    assert repeated.done
    assert repeated.payload["reason_code"] == "repeat_loop"
    assert environment.termination_reason == "repeat_loop"


def test_missing_evidence_task_can_abstain_with_whitelisted_reason():
    environment = FinSightEnvironment(load_business_cases()[3])
    observation = environment.step(
        _action(ActionName.ABSTAIN, reason_code="missing_evidence")
    )

    assert observation.done
    assert environment.termination_reason == "abstained:missing_evidence"


def test_environment_accepts_finqa_rounded_expected_value():
    task = load_business_cases()[0]
    environment = FinSightEnvironment(task, expected_value=13.636)
    actions = [
        _action(ActionName.SEARCH, query=task.question, top_k=2),
        _action(ActionName.OPEN_EVIDENCE, evidence_id="income_statement"),
        _action(ActionName.SELECT_EVIDENCE, evidence_ids=["income_statement"]),
        _action(ActionName.EMIT_PROGRAM, program=task.program),
        _action(ActionName.CALCULATE),
        _action(ActionName.VERIFY),
    ]
    observations = environment.replay(actions)

    assert observations[-1].payload["passed"]
