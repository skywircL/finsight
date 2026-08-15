from finsight.business_cases import load_business_cases
from finsight.environment import FinSightEnvironment
from finsight.environment.projection import estimate_tokens, project_payload


def test_projection_truncates_long_text_within_observation_budget():
    payload = {"evidence": {"evidence_id": "e1", "text": "revenue " * 500}}

    projected, audit = project_payload(
        payload,
        observation_limit=80,
        context_remaining=1_000,
    )

    assert audit.truncated
    assert not audit.overflow
    assert audit.emitted_tokens_estimate <= 80
    assert estimate_tokens(projected) <= 80
    assert projected["evidence"]["evidence_id"] == "e1"


def test_environment_fails_closed_when_context_is_exhausted():
    environment = FinSightEnvironment(
        load_business_cases()[0],
        max_observation_tokens=10,
        max_context_tokens=1,
    )

    initial = environment.initial_observation()

    assert initial.done
    assert initial.audit["overflow"]
    assert environment.trajectory()["context"]["context_overflow"]
    assert environment.termination_reason == "context_overflow"
