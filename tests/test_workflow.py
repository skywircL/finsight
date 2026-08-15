from finsight.agent import FinSightAgent
from finsight.business_cases import load_business_cases
from finsight.models import AgentState


def test_three_normal_cases_are_delivered():
    for task in load_business_cases()[:3]:
        result = FinSightAgent().run(task)
        assert result.state == AgentState.DELIVERED
        assert result.verified
        assert "原始依据" in result.report_markdown
        assert [event.event_id for event in result.trace] == [
            f"e{index:04d}" for index in range(1, 7)
        ]
        assert all(event.action for event in result.trace)


def test_missing_evidence_case_refuses():
    result = FinSightAgent().run(load_business_cases()[3])
    assert result.state == AgentState.REFUSED
    assert not result.verified
    assert result.answer is None
    assert "证据不足" in result.report_markdown
