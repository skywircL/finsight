from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

from finsight.environment.protocol import (
    ActionName,
    AgentAction,
    EnvironmentEvent,
    EnvironmentObservation,
)
from finsight.environment.projection import project_payload
from finsight.models import AnalysisTask, EventStatus, RetrievedEvidence
from finsight.retrieval import BM25Retriever, TableAwareBM25Retriever
from finsight.tools import CalculationError, execute_program
from finsight.verifier import verify_result


class ActionRejected(ValueError):
    pass


class FinSightEnvironment:
    """Replayable tool environment with a strict actor-visible boundary.

    Gold evidence IDs remain inside the verifier. They are never included in an
    observation, so a policy must discover and open supporting evidence itself.
    """

    ABSTAIN_REASONS = {
        "missing_evidence",
        "ambiguous_question",
        "unsupported_operation",
        "unsafe_request",
    }

    def __init__(
        self,
        task: AnalysisTask,
        *,
        table: Sequence[Sequence[str]] = (),
        expected_value: float | None = None,
        answer_rel_tol: float = 1e-3,
        answer_abs_tol: float = 1e-3,
        max_steps: int = 12,
        max_no_progress: int = 3,
        max_search_top_k: int = 10,
        retrieval_method: str = "table_aware",
        max_observation_tokens: int = 2_048,
        max_context_tokens: int = 24_576,
    ) -> None:
        if (
            max_steps < 1
            or max_no_progress < 1
            or max_observation_tokens < 1
            or max_context_tokens < 1
        ):
            raise ValueError("步数和上下文限制必须为正整数。")
        self.task = task
        self.table = tuple(tuple(cell for cell in row) for row in table)
        self._expected_value = expected_value
        self._answer_rel_tol = answer_rel_tol
        self._answer_abs_tol = answer_abs_tol
        self.max_steps = max_steps
        self.max_no_progress = max_no_progress
        self.max_search_top_k = max_search_top_k
        if retrieval_method not in {"bm25", "table_aware"}:
            raise ValueError("retrieval_method 只能是 bm25 或 table_aware。")
        self.retrieval_method = retrieval_method
        self.max_observation_tokens = max_observation_tokens
        self.max_context_tokens = max_context_tokens
        self.events: list[EnvironmentEvent] = []
        self.done = False
        self.termination_reason: str | None = None
        self._last_search_ids: list[str] = []
        self._seen_ids: set[str] = set()
        self._opened_ids: set[str] = set()
        self._selected_ids: list[str] = []
        self._retrieved: dict[str, RetrievedEvidence] = {}
        self._program: str | None = None
        self._calculation_value: float | None = None
        self._calculation_steps: tuple[str, ...] = ()
        self._verified = False
        self._no_progress = 0
        self._last_signature: str | None = None
        self._context_tokens_used = 0
        self._observation_truncations = 0
        self._context_overflow = False

    def initial_observation(self) -> EnvironmentObservation:
        """Return task information safe for the policy to see."""

        return self._observation(
            EventStatus.COMPLETED,
            "财务分析任务已载入。",
            {
                "task_id": self.task.task_id,
                "question": self.task.question,
                "allowed_actions": [action.value for action in ActionName],
                "max_steps": self.max_steps,
                "retrieval_method": self.retrieval_method,
            },
            event_id="e0000",
        )

    def step(self, action: AgentAction | Mapping[str, Any]) -> EnvironmentObservation:
        if self.done:
            return self._observation(
                EventStatus.REJECTED,
                "Episode 已终止，不能继续执行动作。",
                {"reason_code": "episode_done"},
            )

        if len(self.events) >= self.max_steps:
            self.done = True
            self.termination_reason = "max_steps"
            return self._observation(
                EventStatus.REJECTED,
                "达到最大环境步数。",
                {"reason_code": "max_steps"},
            )

        if not isinstance(action, AgentAction):
            try:
                action = AgentAction.from_dict(action)
            except ValueError as exc:
                return self._record_invalid_action(str(exc))

        signature = json.dumps(action.to_dict(), ensure_ascii=False, sort_keys=True, default=str)
        exact_repeat = signature == self._last_signature
        self._last_signature = signature

        try:
            payload, message, progress, terminal = self._execute(action)
            status = EventStatus.COMPLETED
        except ActionRejected as exc:
            payload = {"reason_code": "action_rejected"}
            message = str(exc)
            progress = False
            terminal = False
            status = EventStatus.REJECTED
        except CalculationError as exc:
            payload = {"reason_code": "calculation_error"}
            message = f"计算程序未通过安全执行：{exc}"
            progress = False
            terminal = False
            status = EventStatus.FAILED

        self._no_progress = 0 if progress else self._no_progress + 1
        if exact_repeat:
            self.done = True
            self.termination_reason = "repeat_loop"
            payload = {**payload, "reason_code": "repeat_loop"}
            message = f"{message} 检测到连续重复动作，Episode 已终止。"
            terminal = True
        elif self._no_progress >= self.max_no_progress:
            self.done = True
            self.termination_reason = "no_progress_loop"
            payload = {**payload, "reason_code": "no_progress_loop"}
            message = f"{message} 连续动作没有产生有效进展，Episode 已终止。"
            terminal = True

        if terminal:
            self.done = True
        return self._record(action, status, message, payload, new_evidence=progress)

    def replay(self, actions: Iterable[AgentAction | Mapping[str, Any]]) -> list[EnvironmentObservation]:
        observations = []
        for action in actions:
            observations.append(self.step(action))
            if self.done:
                break
        return observations

    def trajectory(self) -> dict[str, Any]:
        return {
            "schema_version": "finsight-trajectory-v1",
            "task_id": self.task.task_id,
            "done": self.done,
            "termination_reason": self.termination_reason,
            "context": {
                "estimated_tokens_used": self._context_tokens_used,
                "max_context_tokens": self.max_context_tokens,
                "max_observation_tokens": self.max_observation_tokens,
                "observation_truncations": self._observation_truncations,
                "context_overflow": self._context_overflow,
            },
            "retrieval_method": self.retrieval_method,
            "events": [event.to_dict() for event in self.events],
        }

    def _execute(self, action: AgentAction) -> tuple[dict, str, bool, bool]:
        args = dict(action.arguments)
        if action.name == ActionName.SEARCH:
            return self._search(args)
        if action.name == ActionName.OPEN_EVIDENCE:
            return self._open_evidence(args)
        if action.name == ActionName.SELECT_EVIDENCE:
            return self._select_evidence(args)
        if action.name == ActionName.EMIT_PROGRAM:
            return self._emit_program(args)
        if action.name == ActionName.CALCULATE:
            return self._calculate(args)
        if action.name == ActionName.VERIFY:
            return self._verify(args)
        if action.name == ActionName.DELIVER:
            return self._deliver(args)
        if action.name == ActionName.ABSTAIN:
            return self._abstain(args)
        raise ActionRejected("动作不在白名单中。")

    def _search(self, args: dict) -> tuple[dict, str, bool, bool]:
        self._require_keys(args, required={"query"}, optional={"evidence_kind", "top_k"})
        query = args["query"]
        kind = args.get("evidence_kind")
        top_k = args.get("top_k", 5)
        if not isinstance(query, str) or not query.strip():
            raise ActionRejected("search.query 必须是非空字符串。")
        if kind not in {None, "text", "table"}:
            raise ActionRejected("evidence_kind 只能是 text 或 table。")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= self.max_search_top_k:
            raise ActionRejected(f"top_k 必须在 1 到 {self.max_search_top_k} 之间。")
        candidates = [item for item in self.task.evidence if kind is None or item.kind == kind]
        retriever = (
            TableAwareBM25Retriever(candidates)
            if self.retrieval_method == "table_aware"
            else BM25Retriever(candidates)
        )
        ranked = retriever.search(query.strip(), top_k=top_k)
        self._last_search_ids = [item.evidence.evidence_id for item in ranked]
        for item in ranked:
            self._retrieved[item.evidence.evidence_id] = item
        new_ids = set(self._last_search_ids) - self._seen_ids
        self._seen_ids.update(self._last_search_ids)
        results = [
            {
                "evidence_id": item.evidence.evidence_id,
                "kind": item.evidence.kind,
                "score": round(item.score, 6),
                "preview": item.evidence.text[:240],
            }
            for item in ranked
        ]
        return {"results": results}, "证据检索完成。", bool(new_ids), False

    def _open_evidence(self, args: dict) -> tuple[dict, str, bool, bool]:
        self._require_keys(args, required={"evidence_id"})
        evidence_id = args["evidence_id"]
        if not isinstance(evidence_id, str) or evidence_id not in self._last_search_ids:
            raise ActionRejected("只能打开最近一次搜索结果中的 evidence_id。")
        item = self._retrieved[evidence_id].evidence
        progress = evidence_id not in self._opened_ids
        self._opened_ids.add(evidence_id)
        return {
            "evidence": {
                "evidence_id": item.evidence_id,
                "kind": item.kind,
                "text": item.text,
                "page": item.page,
                "source": item.source,
            }
        }, "证据已打开。", progress, False

    def _select_evidence(self, args: dict) -> tuple[dict, str, bool, bool]:
        self._require_keys(args, required={"evidence_ids"})
        evidence_ids = args["evidence_ids"]
        if not isinstance(evidence_ids, list) or not evidence_ids or not all(
            isinstance(item, str) for item in evidence_ids
        ):
            raise ActionRejected("evidence_ids 必须是非空字符串数组。")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ActionRejected("evidence_ids 不能重复。")
        unopened = set(evidence_ids) - self._opened_ids
        if unopened:
            raise ActionRejected("只能选择已经打开核验的证据。")
        progress = evidence_ids != self._selected_ids
        self._selected_ids = list(evidence_ids)
        self._verified = False
        return {"selected_evidence_ids": self._selected_ids}, "证据选择已更新。", progress, False

    def _emit_program(self, args: dict) -> tuple[dict, str, bool, bool]:
        self._require_keys(args, required={"program"})
        program = args["program"]
        if not self._selected_ids:
            raise ActionRejected("生成程序前必须先选择已核验的证据。")
        if not isinstance(program, str) or not program.strip() or len(program) > 2_000:
            raise ActionRejected("program 必须是长度不超过 2000 的非空字符串。")
        normalized = program.strip()
        progress = normalized != self._program
        self._program = normalized
        self._calculation_value = None
        self._calculation_steps = ()
        self._verified = False
        return {"program": normalized}, "结构化计算程序已记录。", progress, False

    def _calculate(self, args: dict) -> tuple[dict, str, bool, bool]:
        self._require_keys(args)
        if not self._program:
            raise ActionRejected("尚未生成结构化计算程序。")
        result = execute_program(self._program, table=self.table)
        if not isinstance(result.value, float):
            raise ActionRejected("当前环境只接受数值型经营分析结果。")
        progress = result.value != self._calculation_value
        self._calculation_value = result.value
        self._calculation_steps = result.steps
        self._verified = False
        return {
            "value": result.value,
            "steps": list(result.steps),
        }, "白名单计算程序执行完成。", progress, False

    def _verify(self, args: dict) -> tuple[dict, str, bool, bool]:
        self._require_keys(args)
        if self._calculation_value is None:
            raise ActionRejected("验证前必须先完成计算。")
        selected = [self._retrieved[evidence_id] for evidence_id in self._selected_ids]
        verification = verify_result(
            value=self._calculation_value,
            retrieved=selected,
            required_evidence_ids=self.task.expected_evidence_ids,
            expected_value=self._expected_value,
            expected_rel_tol=self._answer_rel_tol,
            expected_abs_tol=self._answer_abs_tol,
        )
        progress = verification.passed and not self._verified
        self._verified = verification.passed
        return {
            "passed": verification.passed,
            "evidence_coverage": verification.evidence_coverage,
            "reason": verification.reason,
        }, verification.reason, progress, False

    def _deliver(self, args: dict) -> tuple[dict, str, bool, bool]:
        self._require_keys(args)
        if not self._verified or self._calculation_value is None:
            raise ActionRejected("只有通过验证的结果才能交付。")
        self.termination_reason = "delivered"
        return {
            "answer": self._calculation_value,
            "unit": self.task.unit,
            "program": self._program,
            "evidence_ids": list(self._selected_ids),
        }, "已交付通过验证的经营分析结果。", True, True

    def _abstain(self, args: dict) -> tuple[dict, str, bool, bool]:
        self._require_keys(args, required={"reason_code"})
        reason = args["reason_code"]
        if reason not in self.ABSTAIN_REASONS:
            raise ActionRejected("拒答 reason_code 不在白名单中。")
        self.termination_reason = f"abstained:{reason}"
        return {"reason_code": reason}, "Agent 已选择拒答。", True, True

    @staticmethod
    def _require_keys(
        args: dict,
        *,
        required: set[str] | None = None,
        optional: set[str] | None = None,
    ) -> None:
        required = required or set()
        optional = optional or set()
        missing = required - set(args)
        extra = set(args) - required - optional
        if missing:
            raise ActionRejected(f"动作缺少参数：{', '.join(sorted(missing))}。")
        if extra:
            raise ActionRejected(f"动作包含未声明参数：{', '.join(sorted(extra))}。")

    def _record_invalid_action(self, message: str) -> EnvironmentObservation:
        self._no_progress += 1
        if self._no_progress >= self.max_no_progress:
            self.done = True
            self.termination_reason = "no_progress_loop"
            message = f"{message} 连续非法动作没有产生进展，Episode 已终止。"
        placeholder = AgentAction(ActionName.ABSTAIN, {"reason_code": "invalid_action"})
        return self._record(
            placeholder,
            EventStatus.REJECTED,
            message,
            {"reason_code": "invalid_action"},
            new_evidence=False,
        )

    def _record(
        self,
        action: AgentAction,
        status: EventStatus,
        message: str,
        payload: Mapping[str, Any],
        *,
        new_evidence: bool,
    ) -> EnvironmentObservation:
        event_id = f"e{len(self.events) + 1:04d}"
        observation = self._observation(
            status,
            message,
            payload,
            event_id=event_id,
        )
        if action.name == ActionName.SEARCH:
            self._last_search_ids = [
                item["evidence_id"]
                for item in observation.payload.get("results", [])
                if isinstance(item, Mapping) and "evidence_id" in item
            ]
        event = EnvironmentEvent(
            event_id=event_id,
            action=action,
            status=status,
            message=message,
            observation=dict(observation.payload),
            observation_audit=dict(observation.audit),
            new_evidence=new_evidence,
            done=observation.done,
        )
        self.events.append(event)
        return observation

    def _observation(
        self,
        status: EventStatus,
        message: str,
        payload: Mapping[str, Any],
        *,
        event_id: str | None = None,
    ) -> EnvironmentObservation:
        remaining = self.max_context_tokens - self._context_tokens_used
        projected, projection_audit = project_payload(
            payload,
            observation_limit=self.max_observation_tokens,
            context_remaining=remaining,
        )
        audit = projection_audit.to_dict()
        self._context_tokens_used += projection_audit.emitted_tokens_estimate
        if projection_audit.truncated:
            self._observation_truncations += 1
        if projection_audit.overflow:
            self._context_overflow = True
            self.done = True
            self.termination_reason = "context_overflow"
        audit["context_used_after"] = self._context_tokens_used
        audit["context_limit"] = self.max_context_tokens
        return EnvironmentObservation(
            event_id=event_id or f"e{len(self.events):04d}",
            status=status,
            done=self.done,
            message=message,
            payload=projected,
            audit=audit,
        )
