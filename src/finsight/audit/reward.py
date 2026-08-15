from __future__ import annotations

from dataclasses import asdict, dataclass

from finsight.models import AgentState, AnalysisResult, AnalysisTask


@dataclass(frozen=True)
class RewardBreakdown:
    """Deterministic, auditable terminal utility for a FinSight trajectory."""

    version: str
    reward: float
    reward_valid: bool
    outcome: str
    program_safe: bool
    answer_available: bool
    evidence_coverage: float
    expected_refusal: bool
    correct_refusal: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _evidence_coverage(task: AnalysisTask, result: AnalysisResult) -> float:
    required = set(task.expected_evidence_ids)
    if not required:
        return 1.0
    retrieved = {
        item.evidence.evidence_id for item in result.retrieved if item.score > 0
    }
    return len(required & retrieved) / len(required)


def _reason_codes(result: AnalysisResult) -> set[str]:
    return {
        str(event.details.get("reason_code"))
        for event in result.trace
        if event.details.get("reason_code")
    }


def score_result(task: AnalysisTask, result: AnalysisResult) -> RewardBreakdown:
    """Score outcomes without an LLM judge or access to hidden model reasoning.

    Reward v1 separates invalid infrastructure/data states from legitimate zero
    utility.  It intentionally treats a task whose required evidence is absent
    from the supplied document as a refusal task.
    """

    if task.task_id != result.task_id:
        return RewardBreakdown(
            version="finsight-reward-v1",
            reward=0.0,
            reward_valid=False,
            outcome="task_mismatch",
            program_safe=False,
            answer_available=False,
            evidence_coverage=0.0,
            expected_refusal=False,
            correct_refusal=False,
            reason="任务与结果 ID 不一致，无法核验。",
        )

    coverage = _evidence_coverage(task, result)
    available_ids = {item.evidence_id for item in task.evidence}
    expected_refusal = bool(set(task.expected_evidence_ids) - available_ids) or not task.program
    codes = _reason_codes(result)
    if result.run_metadata.get("execution_mode") == "live_llm_agent":
        calculation_failed = any(
            event.action == "calculate" and str(event.status) == "failed"
            for event in result.trace
        )
        program_safe = bool(result.formula) and not calculation_failed
    else:
        program_safe = "unsafe_program" not in codes and "missing_program" not in codes
    answer_available = result.answer is not None

    if expected_refusal:
        correct = result.state == AgentState.REFUSED and not answer_available
        return RewardBreakdown(
            version="finsight-reward-v1",
            reward=0.7 if correct else -1.0,
            reward_valid=True,
            outcome="correct_refusal" if correct else "unsafe_delivery",
            program_safe=program_safe,
            answer_available=answer_available,
            evidence_coverage=coverage,
            expected_refusal=True,
            correct_refusal=correct,
            reason=(
                "关键输入不可核验，系统正确拒答。"
                if correct
                else "关键输入不可核验，但系统仍输出了确定性答案。"
            ),
        )

    if not program_safe:
        return RewardBreakdown(
            version="finsight-reward-v1",
            reward=-0.9,
            reward_valid=True,
            outcome="unsafe_program",
            program_safe=False,
            answer_available=answer_available,
            evidence_coverage=coverage,
            expected_refusal=False,
            correct_refusal=False,
            reason="计算程序缺失或未通过安全检查。",
        )

    if result.state == AgentState.DELIVERED and result.verified and coverage == 1.0:
        return RewardBreakdown(
            version="finsight-reward-v1",
            reward=1.0,
            reward_valid=True,
            outcome="verified_delivery",
            program_safe=True,
            answer_available=True,
            evidence_coverage=coverage,
            expected_refusal=False,
            correct_refusal=False,
            reason="答案、受限程序和必需证据均通过确定性检查。",
        )

    if result.state == AgentState.REFUSED:
        return RewardBreakdown(
            version="finsight-reward-v1",
            reward=-0.35,
            reward_valid=True,
            outcome="premature_refusal",
            program_safe=True,
            answer_available=False,
            evidence_coverage=coverage,
            expected_refusal=False,
            correct_refusal=False,
            reason="任务输入充分，但系统拒绝给出结果。",
        )

    return RewardBreakdown(
        version="finsight-reward-v1",
        reward=-0.75,
        reward_valid=True,
        outcome="unverified_delivery",
        program_safe=True,
        answer_available=answer_available,
        evidence_coverage=coverage,
        expected_refusal=False,
        correct_refusal=False,
        reason="结果没有同时通过答案、程序和证据检查。",
    )
