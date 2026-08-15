from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from finsight.environment import ActionName, AgentAction, FinSightEnvironment
from finsight.models import (
    AgentState,
    AnalysisResult,
    AnalysisTask,
    EventStatus,
    RetrievedEvidence,
    TraceEvent,
)
from finsight.report import build_report
from finsight.tools import CalculationError, execute_program
from finsight.training.teacher import (
    OpenAICompatibleTeacher,
    TeacherError,
    load_teacher_config_file,
)


LIVE_ACTOR_SYSTEM_PROMPT = """你是 FinSight 实时财务分析 Agent 的动作策略。
每次只输出以下八种 JSON 动作之一，不得增加字段，不得输出 Markdown 或解释：
{"name":"search","arguments":{"query":"非空检索词","top_k":5}}
{"name":"open_evidence","arguments":{"evidence_id":"最近搜索结果中的一个 ID"}}
{"name":"select_evidence","arguments":{"evidence_ids":["已经打开的 ID"]}}
{"name":"emit_program","arguments":{"program":"白名单 DSL"}}
{"name":"calculate","arguments":{}}
{"name":"verify","arguments":{}}
{"name":"deliver","arguments":{}}
{"name":"abstain","arguments":{"reason_code":"missing_evidence"}}
你只能依据用户问题和环境返回的 Observation 决策，不能假设未出现的数字、证据 ID、
标准公式或答案。动作会经过严格 Schema、白名单计算器和确定性验证器。

执行规则：
1. 先用问题中的指标和年份 search；open_evidence 只能使用最近一次搜索结果中的 ID。
   必须打开并选择支撑全部数字、年份和口径的证据，不能把无关文本当作数值证据。
2. select_evidence 后才能 emit_program。计算只允许 FinQA DSL：add、subtract、multiply、
   divide、exp、greater、average、sum、max、min；多步用逗号分隔并用 #0、#1 引用。
   常量 100 写作 const_100，严禁中缀运算符和任意代码。
3. 百分比变化通常为 multiply(divide(subtract(新值,旧值),旧值),const_100)。
   calculate、verify、deliver 的 arguments 必须是 {}。verify 通过后立即 deliver。
4. 若 verify 失败，依据回执补搜证据或修正程序；不要重复同一无效动作。
5. 已搜索但关键输入在资料中确实不存在时，使用
   {"name":"abstain","arguments":{"reason_code":"missing_evidence"}}，不得补造数字。"""
LIVE_ACTOR_PROMPT_VERSION = "finsight-live-actor-v2-exact-schema"


class ActionPolicy(Protocol):
    model_name: str
    prompt_version: str
    schema_repairs: int
    api_calls: int
    total_latency_seconds: float

    def next_action(self, messages: Sequence[Mapping[str, Any]]) -> AgentAction: ...


@dataclass
class OpenAICompatibleActor:
    """Online action policy backed by an OpenAI-compatible chat endpoint."""

    _client: OpenAICompatibleTeacher
    provider_name: str = "OpenAI-compatible"

    @classmethod
    def from_connection(
        cls,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        provider_name: str = "OpenAI-compatible",
        temperature: float | None = 0.0,
        json_mode: bool = True,
        extra_body: dict[str, Any] | None = None,
        timeout_seconds: float = 60.0,
    ) -> OpenAICompatibleActor:
        """Create a live actor from user-supplied, session-scoped API settings."""

        cleaned_base_url = base_url.strip().rstrip("/")
        cleaned_api_key = api_key.strip()
        cleaned_model = model_name.strip()
        if not cleaned_base_url or not cleaned_api_key or not cleaned_model:
            raise TeacherError("Base URL、API Key 和模型 ID 均不能为空。")
        if not cleaned_base_url.startswith(("https://", "http://")):
            raise TeacherError("Base URL 必须以 https:// 或 http:// 开头。")
        return cls(
            OpenAICompatibleTeacher(
                base_url=cleaned_base_url,
                api_key=cleaned_api_key,
                model_name=cleaned_model,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                response_format={"type": "json_object"} if json_mode else None,
                extra_body=dict(extra_body or {}),
                prompt_version=LIVE_ACTOR_PROMPT_VERSION,
                system_prompt=LIVE_ACTOR_SYSTEM_PROMPT,
            ),
            provider_name=provider_name.strip() or "OpenAI-compatible",
        )

    @classmethod
    def from_config_file(cls, path: str | Path) -> OpenAICompatibleActor:
        config = load_teacher_config_file(path)
        return cls(
            OpenAICompatibleTeacher(
                base_url=config["OPENAI_BASE_URL"],
                api_key=config["OPENAI_API_KEY"],
                model_name=config["TEACHER_MODEL"],
                prompt_version=LIVE_ACTOR_PROMPT_VERSION,
                system_prompt=LIVE_ACTOR_SYSTEM_PROMPT,
            ),
            provider_name="本地 OpenAI-compatible 配置",
        )

    @property
    def model_name(self) -> str:
        return self._client.model_name

    @property
    def prompt_version(self) -> str:
        return self._client.prompt_version

    @property
    def schema_repairs(self) -> int:
        return self._client.schema_repairs

    @property
    def api_calls(self) -> int:
        return self._client.api_calls

    @property
    def total_latency_seconds(self) -> float:
        return self._client.total_latency_seconds

    @property
    def last_usage(self) -> dict[str, Any]:
        return dict(self._client.last_usage)

    def next_action(self, messages: Sequence[Mapping[str, Any]]) -> AgentAction:
        return self._client.next_action(messages)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _expected_value_for_verifier(task: AnalysisTask) -> float | None:
    """Compute hidden business-case gold for the verifier, never for actor messages."""

    if not task.program:
        return None
    try:
        value = execute_program(task.program).value
    except CalculationError:
        return None
    return value if isinstance(value, float) else None


def _state_for_action(name: ActionName, observation: Mapping[str, Any]) -> AgentState:
    if name in {ActionName.SEARCH, ActionName.OPEN_EVIDENCE, ActionName.SELECT_EVIDENCE}:
        return AgentState.RETRIEVED
    if name in {ActionName.EMIT_PROGRAM, ActionName.CALCULATE}:
        return AgentState.CALCULATED
    if name == ActionName.VERIFY:
        return AgentState.VERIFIED if observation.get("passed") else AgentState.CALCULATED
    if name == ActionName.DELIVER:
        return AgentState.DELIVERED
    return AgentState.REFUSED


def _collect_retrieved(
    task: AnalysisTask,
    environment: FinSightEnvironment,
) -> list[RetrievedEvidence]:
    scores: dict[str, tuple[float, int]] = {}
    opened: list[str] = []
    selected: list[str] = []
    for event in environment.events:
        if event.action.name == ActionName.SEARCH:
            for rank, item in enumerate(event.observation.get("results", []), start=1):
                if not isinstance(item, Mapping) or "evidence_id" not in item:
                    continue
                evidence_id = str(item["evidence_id"])
                score = float(item.get("score", 0.0))
                if evidence_id not in scores or score > scores[evidence_id][0]:
                    scores[evidence_id] = (score, rank)
        elif (
            event.action.name == ActionName.OPEN_EVIDENCE
            and event.status == EventStatus.COMPLETED
        ):
            evidence_id = str(event.action.arguments.get("evidence_id", ""))
            if evidence_id and evidence_id not in opened:
                opened.append(evidence_id)
        elif (
            event.action.name == ActionName.SELECT_EVIDENCE
            and event.status == EventStatus.COMPLETED
        ):
            selected = [str(item) for item in event.action.arguments.get("evidence_ids", [])]

    chosen = selected if environment.termination_reason == "delivered" else opened
    evidence_by_id = {item.evidence_id: item for item in task.evidence}
    retrieved = []
    for fallback_rank, evidence_id in enumerate(chosen, start=1):
        if evidence_id not in evidence_by_id:
            continue
        score, rank = scores.get(evidence_id, (1e-12, fallback_rank))
        retrieved.append(
            RetrievedEvidence(
                evidence=evidence_by_id[evidence_id],
                score=max(score, 1e-12),
                rank=rank,
            )
        )
    return retrieved


class LLMFinSightAgent:
    """Run a live LLM policy through the guarded FinSight environment."""

    def __init__(self, policy: ActionPolicy, *, max_steps: int = 12) -> None:
        self.policy = policy
        self.max_steps = max_steps

    def run(self, task: AnalysisTask, *, top_k: int = 5) -> AnalysisResult:
        del top_k  # The actor chooses top_k inside the guarded action schema.
        started = time.perf_counter()
        calls_before = int(getattr(self.policy, "api_calls", 0))
        repairs_before = int(getattr(self.policy, "schema_repairs", 0))
        policy_latency_before = float(getattr(self.policy, "total_latency_seconds", 0.0))
        environment = FinSightEnvironment(
            task,
            expected_value=_expected_value_for_verifier(task),
            max_steps=self.max_steps,
        )
        initial = environment.initial_observation()
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": _canonical_json(
                    {"question": task.question, "observation": initial.to_dict()}
                ),
            }
        ]
        actor_error: str | None = None
        for _ in range(self.max_steps):
            if environment.done:
                break
            try:
                action = self.policy.next_action(messages)
            except TeacherError as exc:
                actor_error = str(exc)
                break
            messages.append(
                {"role": "assistant", "content": _canonical_json(action.to_dict())}
            )
            observation = environment.step(action)
            messages.append(
                {
                    "role": "tool",
                    "name": action.name,
                    "content": _canonical_json(observation.to_dict()),
                }
            )

        retrieved = _collect_retrieved(task, environment)
        generated_program: str | None = None
        answer: float | None = None
        for event in environment.events:
            if event.action.name == ActionName.EMIT_PROGRAM and event.status == EventStatus.COMPLETED:
                generated_program = str(event.observation.get("program", "")) or generated_program
            if event.action.name == ActionName.DELIVER and event.status == EventStatus.COMPLETED:
                delivered = event.observation.get("answer")
                answer = float(delivered) if isinstance(delivered, (int, float)) else None

        delivered = environment.termination_reason == "delivered" and answer is not None
        refusal_reason = None
        if not delivered:
            if actor_error:
                refusal_reason = f"实时模型调用失败，系统已关闭式停止：{actor_error}"
            elif environment.events:
                refusal_reason = environment.events[-1].message
            else:
                refusal_reason = "实时模型未产生可执行动作，系统已关闭式停止。"

        trace = [
            TraceEvent(
                state=AgentState.INGESTED,
                message="分析资料已载入，隐藏验收字段未暴露给模型",
                event_id="e0001",
                action="ingest_task",
                details={"evidence": len(task.evidence)},
            ),
            TraceEvent(
                state=AgentState.PLANNED,
                message="实时 LLM Actor 已接管结构化动作决策",
                event_id="e0002",
                action="start_llm_actor",
                details={
                    "model": self.policy.model_name,
                    "prompt_version": self.policy.prompt_version,
                },
            ),
        ]
        for event in environment.events:
            trace.append(
                TraceEvent(
                    state=_state_for_action(event.action.name, event.observation),
                    message=event.message,
                    event_id=f"e{len(trace) + 1:04d}",
                    action=event.action.name,
                    status=event.status,
                    details={
                        "environment_event_id": event.event_id,
                        "arguments": dict(event.action.arguments),
                        "observation": dict(event.observation),
                        "observation_audit": dict(event.observation_audit),
                        "new_evidence": event.new_evidence,
                    },
                )
            )
        final_state = AgentState.DELIVERED if delivered else AgentState.REFUSED
        if not trace or trace[-1].state != final_state:
            trace.append(
                TraceEvent(
                    state=final_state,
                    message=refusal_reason or "实时分析已终止。",
                    event_id=f"e{len(trace) + 1:04d}",
                    action="fail_closed",
                    status=EventStatus.REJECTED,
                    details={"reason_code": "actor_or_environment_failure"},
                )
            )

        elapsed = time.perf_counter() - started
        run_metadata = {
            "execution_mode": "live_llm_agent",
            "provider": getattr(self.policy, "provider_name", "OpenAI-compatible"),
            "model": self.policy.model_name,
            "prompt_version": self.policy.prompt_version,
            "api_calls": int(getattr(self.policy, "api_calls", 0)) - calls_before,
            "schema_repairs": int(getattr(self.policy, "schema_repairs", 0)) - repairs_before,
            "llm_latency_seconds": round(
                float(getattr(self.policy, "total_latency_seconds", 0.0))
                - policy_latency_before,
                4,
            ),
            "elapsed_seconds": round(elapsed, 4),
            "environment_steps": len(environment.events),
            "termination_reason": environment.termination_reason or "actor_error",
            "hidden_gold_exposed_to_actor": False,
            "actor_error": actor_error,
        }
        last_usage = getattr(self.policy, "last_usage", None)
        if isinstance(last_usage, Mapping) and last_usage:
            run_metadata["last_usage"] = dict(last_usage)

        report = build_report(
            task,
            answer=answer,
            formula=generated_program,
            retrieved=retrieved,
            verified=delivered,
            refusal_reason=refusal_reason,
        )
        return AnalysisResult(
            task_id=task.task_id,
            state=final_state,
            answer=answer if delivered else None,
            unit=task.unit,
            formula=generated_program,
            retrieved=retrieved,
            verified=delivered,
            refusal_reason=refusal_reason,
            report_markdown=report,
            trace=trace,
            run_metadata=run_metadata,
        )
