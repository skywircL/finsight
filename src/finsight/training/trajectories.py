from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isclose
from typing import Any, Iterable

from finsight.audit import diagnose_trajectory
from finsight.environment import ActionName, AgentAction, FinSightEnvironment
from finsight.models import AnalysisTask, EventStatus, FinQASample


SYSTEM_INSTRUCTION = """你是 FinSight 财务分析 Agent。只能根据当前 Observation 选择一个
白名单动作；必须先检索、打开并选择证据，再生成受限计算程序。只有 calculate 和 verify
成功后才能 deliver。证据不足时使用 abstain，禁止编造数字或引用不可见证据。"""


@dataclass(frozen=True)
class TrajectoryAcceptance:
    accepted: bool
    rejection_reasons: tuple[str, ...]
    answer_match: bool
    evidence_coverage: float
    all_actions_completed: bool
    termination_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["rejection_reasons"] = list(self.rejection_reasons)
        return value


@dataclass(frozen=True)
class TrajectoryBuildResult:
    accepted: bool
    sft_row: dict[str, Any] | None
    audit: dict[str, Any]


def sample_to_task(sample: FinQASample) -> AnalysisTask:
    return AnalysisTask(
        task_id=sample.sample_id,
        title=f"FinQA {sample.sample_id}",
        question=sample.question,
        evidence=sample.evidences,
        program=None,
        expected_evidence_ids=sample.gold_evidence_ids,
        unit="",
    )


def numeric_answer(sample: FinQASample) -> float | None:
    if isinstance(sample.answer, bool):
        return None
    try:
        return float(sample.answer)
    except (TypeError, ValueError):
        return None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _append_action(
    environment: FinSightEnvironment,
    messages: list[dict[str, Any]],
    action: AgentAction,
) -> dict[str, Any]:
    observation = environment.step(action)
    messages.append(
        {
            "role": "assistant",
            "content": _canonical_json(action.to_dict()),
            "trainable": True,
        }
    )
    messages.append(
        {
            "role": "tool",
            "name": action.name,
            "content": _canonical_json(observation.to_dict()),
            "trainable": False,
        }
    )
    return observation.to_dict()


def _accept_trajectory(
    sample: FinQASample,
    environment: FinSightEnvironment,
    expected_answer: float,
) -> TrajectoryAcceptance:
    reasons: list[str] = []
    events = environment.events
    all_completed = bool(events) and all(
        event.status == EventStatus.COMPLETED for event in events
    )
    if not all_completed:
        reasons.append("action_not_completed")
    if not environment.done or environment.termination_reason != "delivered":
        reasons.append("not_verified_delivery")

    final_payload = events[-1].observation if events else {}
    delivered_answer = final_payload.get("answer")
    answer_match = isinstance(delivered_answer, (int, float)) and isclose(
        float(delivered_answer), expected_answer, rel_tol=1e-3, abs_tol=1e-3
    )
    if not answer_match:
        reasons.append("answer_mismatch")

    verification_events = [event for event in events if event.action.name == ActionName.VERIFY]
    coverage = (
        float(verification_events[-1].observation.get("evidence_coverage", 0.0))
        if verification_events
        else 0.0
    )
    if coverage < 1.0:
        reasons.append("incomplete_gold_evidence")

    selection_events = [
        event for event in events if event.action.name == ActionName.SELECT_EVIDENCE
    ]
    selected_ids = (
        set(selection_events[-1].observation.get("selected_evidence_ids", []))
        if selection_events
        else set()
    )
    if not set(sample.gold_evidence_ids) <= selected_ids:
        reasons.append("gold_evidence_not_selected")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return TrajectoryAcceptance(
        accepted=not unique_reasons,
        rejection_reasons=unique_reasons,
        answer_match=answer_match,
        evidence_coverage=coverage,
        all_actions_completed=all_completed,
        termination_reason=environment.termination_reason,
    )


def build_oracle_trajectory(sample: FinQASample) -> TrajectoryBuildResult:
    """Build one gold-assisted action trace and verify it by real execution.

    This is an Oracle data builder, not a model benchmark. Gold annotations are
    used only to propose actions and to validate the final result. The emitted
    SFT messages contain exactly the observations that an Actor would see.
    """

    expected_answer = numeric_answer(sample)
    if expected_answer is None:
        acceptance = TrajectoryAcceptance(
            accepted=False,
            rejection_reasons=("unsupported_non_numeric_answer",),
            answer_match=False,
            evidence_coverage=0.0,
            all_actions_completed=False,
            termination_reason=None,
        )
        return TrajectoryBuildResult(
            accepted=False,
            sft_row=None,
            audit={
                "schema_version": "finsight-trajectory-audit-v1",
                "task_id": sample.sample_id,
                "builder": "oracle_gold_replay_v1",
                "acceptance": acceptance.to_dict(),
                "trajectory": None,
                "diagnosis": diagnose_trajectory(
                    None,
                    rejection_reasons=acceptance.rejection_reasons,
                ).to_dict(),
            },
        )

    task = sample_to_task(sample)
    max_steps = max(12, 2 * len(sample.gold_evidence_ids) + 10)
    environment = FinSightEnvironment(
        task,
        table=sample.table,
        expected_value=expected_answer,
        max_steps=max_steps,
        max_search_top_k=10,
    )
    initial = environment.initial_observation()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTION, "trainable": False},
        {
            "role": "user",
            "content": _canonical_json(
                {
                    "question": sample.question,
                    "observation": initial.to_dict(),
                }
            ),
            "trainable": False,
        },
    ]

    first_search = _append_action(
        environment,
        messages,
        AgentAction(ActionName.SEARCH, {"query": sample.question, "top_k": 5}),
    )
    visible_ids = {
        item["evidence_id"]
        for item in first_search.get("payload", {}).get("results", [])
        if item.get("score", 0.0) > 0
    }
    opened_ids: set[str] = set()
    for evidence_id in sample.gold_evidence_ids:
        if evidence_id in visible_ids:
            opened = _append_action(
                environment,
                messages,
                AgentAction(ActionName.OPEN_EVIDENCE, {"evidence_id": evidence_id}),
            )
            if opened["status"] == EventStatus.COMPLETED:
                opened_ids.add(evidence_id)

    evidence_by_id = {item.evidence_id: item for item in sample.evidences}
    for evidence_id in sample.gold_evidence_ids:
        if evidence_id in opened_ids or environment.done:
            continue
        evidence = evidence_by_id[evidence_id]
        refined_search = _append_action(
            environment,
            messages,
            AgentAction(
                ActionName.SEARCH,
                {"query": evidence.text, "evidence_kind": evidence.kind, "top_k": 5},
            ),
        )
        refined_ids = {
            item["evidence_id"]
            for item in refined_search.get("payload", {}).get("results", [])
            if item.get("score", 0.0) > 0
        }
        if evidence_id in refined_ids and not environment.done:
            opened = _append_action(
                environment,
                messages,
                AgentAction(ActionName.OPEN_EVIDENCE, {"evidence_id": evidence_id}),
            )
            if opened["status"] == EventStatus.COMPLETED:
                opened_ids.add(evidence_id)

    if set(sample.gold_evidence_ids) <= opened_ids and not environment.done:
        _append_action(
            environment,
            messages,
            AgentAction(
                ActionName.SELECT_EVIDENCE,
                {"evidence_ids": list(sample.gold_evidence_ids)},
            ),
        )
        _append_action(
            environment,
            messages,
            AgentAction(ActionName.EMIT_PROGRAM, {"program": sample.program}),
        )
        _append_action(environment, messages, AgentAction(ActionName.CALCULATE))
        verification = _append_action(environment, messages, AgentAction(ActionName.VERIFY))
        if verification.get("payload", {}).get("passed") and not environment.done:
            _append_action(environment, messages, AgentAction(ActionName.DELIVER))

    acceptance = _accept_trajectory(sample, environment, expected_answer)
    trajectory = environment.trajectory()
    audit = {
        "schema_version": "finsight-trajectory-audit-v1",
        "task_id": sample.sample_id,
        "builder": "oracle_gold_replay_v1",
        "acceptance": acceptance.to_dict(),
        "trajectory": trajectory,
        "diagnosis": diagnose_trajectory(
            trajectory,
            rejection_reasons=acceptance.rejection_reasons,
        ).to_dict(),
    }
    if not acceptance.accepted:
        return TrajectoryBuildResult(accepted=False, sft_row=None, audit=audit)

    row = {
        "schema_version": "finsight-action-sft-v1",
        "task_id": sample.sample_id,
        "messages": messages,
        "metadata": {
            "builder": "oracle_gold_replay_v1",
            "data_normalization": "finqa-normalizer-v2-table-zero",
            "environment": "finsight-environment-v1",
            "loss_policy": "assistant_actions_only",
            "event_count": len(environment.events),
        },
    }
    row["metadata"]["row_sha256"] = sha256(_canonical_json(row).encode()).hexdigest()
    return TrajectoryBuildResult(accepted=True, sft_row=row, audit=audit)


def build_oracle_dataset(
    samples: Iterable[FinQASample],
    *,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    primary_error_counts: Counter[str] = Counter()
    processed = 0
    for sample in samples:
        if limit is not None and processed >= limit:
            break
        processed += 1
        result = build_oracle_trajectory(sample)
        audits.append(result.audit)
        if result.accepted and result.sft_row is not None:
            rows.append(result.sft_row)
        else:
            rejection_counts.update(result.audit["acceptance"]["rejection_reasons"])
            primary = result.audit["diagnosis"]["primary_error"]
            if primary:
                primary_error_counts[primary] += 1
    summary = {
        "schema_version": "finsight-oracle-build-summary-v1",
        "builder": "oracle_gold_replay_v1",
        "data_normalization": "finqa-normalizer-v2-table-zero",
        "processed": processed,
        "accepted": len(rows),
        "rejected": processed - len(rows),
        "acceptance_rate": len(rows) / processed if processed else 0.0,
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "primary_errors": dict(sorted(primary_error_counts.items())),
        "loss_policy": "assistant_actions_only",
    }
    return rows, audits, summary
