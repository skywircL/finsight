from __future__ import annotations

from collections import Counter

from finsight.audit.reward import score_result
from finsight.models import AgentState, AnalysisResult, AnalysisTask, EventStatus


EXPECTED_PATHS = {
    AgentState.DELIVERED: (
        AgentState.INGESTED,
        AgentState.PLANNED,
        AgentState.RETRIEVED,
        AgentState.CALCULATED,
        AgentState.VERIFIED,
        AgentState.DELIVERED,
    ),
}

ALLOWED_REFUSAL_PATHS = {
    (
        AgentState.INGESTED,
        AgentState.PLANNED,
        AgentState.RETRIEVED,
        AgentState.REFUSED,
    ),
    (
        AgentState.INGESTED,
        AgentState.PLANNED,
        AgentState.RETRIEVED,
        AgentState.CALCULATED,
        AgentState.REFUSED,
    ),
}


def _trajectory_panel(result: AnalysisResult) -> dict:
    states = tuple(event.state for event in result.trace)
    ids = [event.event_id for event in result.trace]
    terminal_consistent = bool(states) and states[-1] == result.state
    if result.run_metadata.get("execution_mode") == "live_llm_agent":
        state_path_valid = _live_state_path_valid(states, result.state)
    elif result.state == AgentState.REFUSED:
        state_path_valid = states in ALLOWED_REFUSAL_PATHS
    else:
        state_path_valid = states == EXPECTED_PATHS.get(result.state)
    action_counts = Counter(event.action for event in result.trace)
    rejected = [event.event_id for event in result.trace if event.status == EventStatus.REJECTED]
    return {
        "event_count": len(result.trace),
        "event_ids_unique": len(ids) == len(set(ids)) and all(ids),
        "terminal_consistent": terminal_consistent,
        "state_path_valid": state_path_valid,
        "actions": dict(action_counts),
        "rejected_event_ids": rejected,
        "repeat_actions": sum(max(0, count - 1) for count in action_counts.values()),
    }


def _live_state_path_valid(states: tuple[AgentState, ...], final: AgentState) -> bool:
    """Validate required milestones while allowing visible LLM recovery actions."""

    if len(states) < 4 or states[:2] != (AgentState.INGESTED, AgentState.PLANNED):
        return False
    if states[-1] != final:
        return False
    allowed = {
        AgentState.INGESTED,
        AgentState.PLANNED,
        AgentState.RETRIEVED,
        AgentState.CALCULATED,
        AgentState.VERIFIED,
        AgentState.DELIVERED,
        AgentState.REFUSED,
    }
    if any(state not in allowed for state in states):
        return False
    if final == AgentState.REFUSED:
        return AgentState.RETRIEVED in states[2:-1]
    if final != AgentState.DELIVERED:
        return False
    cursor = 2
    for milestone in (
        AgentState.RETRIEVED,
        AgentState.CALCULATED,
        AgentState.VERIFIED,
        AgentState.DELIVERED,
    ):
        try:
            cursor = states.index(milestone, cursor) + 1
        except ValueError:
            return False
    return True


def evaluate_result(task: AnalysisTask, result: AnalysisResult) -> dict:
    """Build four separate panels instead of hiding trade-offs in one score."""

    reward = score_result(task, result)
    cited_ids = [item.evidence.evidence_id for item in result.retrieved if item.score > 0]
    required = list(task.expected_evidence_ids)
    missing = sorted(set(required) - set(cited_ids))
    return {
        "schema_version": "finsight-evaluation-v1",
        "task_id": task.task_id,
        "outcome_panel": {
            **reward.to_dict(),
            "terminal_state": result.state,
            "verified": result.verified,
        },
        "evidence_panel": {
            "required_evidence_ids": required,
            "retrieved_evidence_ids": cited_ids,
            "missing_evidence_ids": missing,
            "coverage": reward.evidence_coverage,
            "all_required_covered": not missing,
        },
        "trajectory_panel": _trajectory_panel(result),
        "deterministic_panel": {
            "program_present": bool(
                result.formula
                if result.run_metadata.get("execution_mode") == "live_llm_agent"
                else task.program
            ),
            "program_safe": reward.program_safe,
            "answer_available": reward.answer_available,
            "arbitrary_code_execution": False,
            "infrastructure_valid": reward.reward_valid,
        },
    }


def compare_retrieval_runs(
    baseline: dict,
    candidate: dict,
    *,
    baseline_name: str,
    candidate_name: str,
) -> dict:
    """Pair two fixed-denominator retrieval results without inventing a total score."""

    if baseline.get("dataset") != candidate.get("dataset"):
        raise ValueError("检索实验的数据集不一致。")
    if baseline.get("samples") != candidate.get("samples"):
        raise ValueError("检索实验的固定分母不一致。")
    if baseline.get("top_k") != candidate.get("top_k"):
        raise ValueError("检索实验的 top_k 不一致。")
    metrics = (
        "recall_at_k",
        "mrr",
        "gold_evidence_coverage",
        "table_evidence_recall",
        "text_evidence_recall",
    )
    return {
        "schema_version": "finsight-retrieval-comparison-v1",
        "dataset": baseline["dataset"],
        "fixed_denominator": baseline["samples"],
        "top_k": baseline["top_k"],
        "baseline": baseline_name,
        "candidate": candidate_name,
        "metrics": {
            metric: {
                "baseline": baseline[metric],
                "candidate": candidate[metric],
                "delta": candidate[metric] - baseline[metric],
            }
            for metric in metrics
        },
    }
