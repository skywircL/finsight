from __future__ import annotations

from finsight.models import AgentState, AnalysisResult, AnalysisTask, EventStatus, TraceEvent
from finsight.report import build_report
from finsight.retrieval import BM25Retriever
from finsight.tools import CalculationError, execute_program
from finsight.verifier import verify_result


class FinSightAgent:
    """A deterministic fallback workflow with explicit, auditable states."""

    def run(self, task: AnalysisTask, *, top_k: int = 5) -> AnalysisResult:
        trace: list[TraceEvent] = []
        self._append_event(
            trace,
            AgentState.INGESTED,
            "分析资料已载入",
            action="ingest_task",
            details={"evidence": len(task.evidence)},
        )
        self._append_event(
            trace,
            AgentState.PLANNED,
            "已生成有限步骤计划：检索、计算、验证、交付",
            action="plan_analysis",
            details={"question": task.question},
        )

        retrieved = BM25Retriever(task.evidence).search(task.question, top_k=top_k)
        self._append_event(
            trace,
            AgentState.RETRIEVED,
            "证据检索完成",
            action="retrieve_evidence",
            details={
                "top_k": top_k,
                "evidence_ids": [item.evidence.evidence_id for item in retrieved],
            },
        )

        if not task.program:
            reason = "任务没有可验证的结构化计算程序。"
            self._append_event(
                trace,
                AgentState.REFUSED,
                reason,
                action="refuse_analysis",
                status=EventStatus.REJECTED,
                details={"reason_code": "missing_program"},
            )
            return self._refusal(task, retrieved, trace, reason)

        try:
            calculation = execute_program(task.program)
        except CalculationError as exc:
            reason = f"计算程序未通过安全检查：{exc}"
            self._append_event(
                trace,
                AgentState.REFUSED,
                reason,
                action="refuse_analysis",
                status=EventStatus.REJECTED,
                details={"reason_code": "unsafe_program", "error": str(exc)},
            )
            return self._refusal(task, retrieved, trace, reason)

        if not isinstance(calculation.value, float):
            reason = "当前经营分析任务要求数值结果。"
            self._append_event(
                trace,
                AgentState.REFUSED,
                reason,
                action="refuse_analysis",
                status=EventStatus.REJECTED,
                details={"reason_code": "non_numeric_result"},
            )
            return self._refusal(task, retrieved, trace, reason)

        self._append_event(
            trace,
            AgentState.CALCULATED,
            "白名单计算工具执行完成",
            action="execute_program",
            details={"value": calculation.value, "steps": list(calculation.steps)},
        )
        verification = verify_result(
            value=calculation.value,
            retrieved=retrieved,
            required_evidence_ids=task.expected_evidence_ids,
        )
        self._append_event(
            trace,
            AgentState.VERIFIED if verification.passed else AgentState.REFUSED,
            verification.reason,
            action="verify_result" if verification.passed else "refuse_analysis",
            status=EventStatus.COMPLETED if verification.passed else EventStatus.REJECTED,
            details={
                "evidence_coverage": verification.evidence_coverage,
                "reason_code": "verified" if verification.passed else "missing_evidence",
            },
        )
        if not verification.passed:
            return self._refusal(task, retrieved, trace, verification.reason)

        report = build_report(
            task,
            answer=calculation.value,
            formula=calculation.program,
            retrieved=retrieved,
            verified=True,
        )
        self._append_event(
            trace,
            AgentState.DELIVERED,
            "经营分析报告已生成",
            action="deliver_report",
        )
        return AnalysisResult(
            task_id=task.task_id,
            state=AgentState.DELIVERED,
            answer=calculation.value,
            unit=task.unit,
            formula=calculation.program,
            retrieved=retrieved,
            verified=True,
            refusal_reason=None,
            report_markdown=report,
            trace=trace,
            run_metadata={
                "execution_mode": "deterministic_baseline",
                "model": None,
                "api_calls": 0,
                "schema_repairs": 0,
            },
        )

    @staticmethod
    def _append_event(
        trace: list[TraceEvent],
        state: AgentState,
        message: str,
        *,
        action: str,
        status: EventStatus = EventStatus.COMPLETED,
        details: dict | None = None,
    ) -> None:
        trace.append(
            TraceEvent(
                state=state,
                message=message,
                details=details or {},
                event_id=f"e{len(trace) + 1:04d}",
                action=action,
                status=status,
            )
        )

    @staticmethod
    def _refusal(task, retrieved, trace, reason) -> AnalysisResult:
        return AnalysisResult(
            task_id=task.task_id,
            state=AgentState.REFUSED,
            answer=None,
            unit=task.unit,
            formula=task.program,
            retrieved=list(retrieved),
            verified=False,
            refusal_reason=reason,
            report_markdown=build_report(
                task,
                answer=None,
                formula=task.program,
                retrieved=list(retrieved),
                verified=False,
                refusal_reason=reason,
            ),
            trace=trace,
            run_metadata={
                "execution_mode": "deterministic_baseline",
                "model": None,
                "api_calls": 0,
                "schema_repairs": 0,
            },
        )
