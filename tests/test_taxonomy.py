from finsight.audit import diagnose_trajectory
from finsight.environment import ActionName, AgentAction, FinSightEnvironment
from finsight.business_cases import load_business_cases


def test_taxonomy_prioritizes_context_overflow():
    trajectory = {
        "termination_reason": "context_overflow",
        "events": [
            {
                "event_id": "e0001",
                "status": "rejected",
                "observation": {"reason_code": "action_rejected"},
            },
            {
                "event_id": "e0002",
                "status": "failed",
                "observation": {"reason_code": "context_overflow"},
            },
        ],
    }

    diagnosis = diagnose_trajectory(
        trajectory,
        rejection_reasons=["answer_mismatch", "not_verified_delivery"],
    )

    assert diagnosis.primary_error == "context_overflow"
    assert diagnosis.primary_category == "context"
    assert diagnosis.failing_event_ids == ("e0002",)


def test_taxonomy_marks_delivered_clean_trajectory_valid():
    task = load_business_cases()[0]
    environment = FinSightEnvironment(task)
    actions = [
        AgentAction(ActionName.SEARCH, {"query": task.question, "top_k": 2}),
        AgentAction(ActionName.OPEN_EVIDENCE, {"evidence_id": "income_statement"}),
        AgentAction(ActionName.SELECT_EVIDENCE, {"evidence_ids": ["income_statement"]}),
        AgentAction(ActionName.EMIT_PROGRAM, {"program": task.program}),
        AgentAction(ActionName.CALCULATE),
        AgentAction(ActionName.VERIFY),
        AgentAction(ActionName.DELIVER),
    ]
    environment.replay(actions)

    diagnosis = diagnose_trajectory(environment.trajectory())

    assert diagnosis.valid
    assert diagnosis.primary_error is None
