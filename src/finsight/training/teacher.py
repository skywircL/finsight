from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from finsight.environment import ActionName, AgentAction, validate_action_schema
from finsight.models import FinQASample


TEACHER_SYSTEM_PROMPT = """你是 FinSight 财务分析 Agent 的动作策略。
每次只输出一个 JSON 对象，且只能包含 name 和 arguments：
{"name":"search|open_evidence|select_evidence|emit_program|calculate|verify|deliver|abstain","arguments":{}}
只能依据对话中 Actor 实际看到的 Observation 决策。禁止输出 Markdown，禁止引用隐藏
Gold evidence、Gold program 或 Gold answer。动作会经过严格 Schema 和环境 Guard。

工作规则：
1. 先 search；只能 open_evidence 最近一次搜索结果中的 evidence_id。打开所有支撑分子、
   分母、年份和口径的证据，再 select_evidence。
2. emit_program 只允许 FinQA DSL：add(a,b)、subtract(a,b)、multiply(a,b)、divide(a,b)、
   exp(a,b)、greater(a,b)，以及 average/sum/max/min；多步用逗号分隔并用 #0、#1 引用。
   常量 100 写作 const_100。严禁使用 + - * / 等中缀运算符，也没有 percent() 操作。
3. FinQA 数据中“what percent/fraction of X is Y”通常要求比率 divide(Y,X)，不要擅自
   乘 100；文本中的 12.5% 可直接作为 12.5% 使用，它在 DSL 中表示 0.125。
4. calculate、verify、deliver 的 arguments 必须是 {}。calculate 失败时根据回执修正 DSL；
   verify 通过后立即 deliver。若 verify 提示证据不完整，必须重新 search 相关口径，打开
   尚未核验的结果并更新 select_evidence，不能只反复 verify。
5. 仅在确实无法获得证据或操作不受支持时 abstain，不要猜测 evidence_id。"""
TEACHER_PROMPT_VERSION = "finsight-teacher-prompt-v2-dsl-recovery"


class TeacherError(RuntimeError):
    pass


class TeacherPolicy(Protocol):
    model_name: str
    prompt_version: str
    schema_repairs: int
    api_calls: int
    total_latency_seconds: float

    def next_action(self, messages: Sequence[Mapping[str, Any]]) -> AgentAction: ...


def load_teacher_config_file(path: str | Path) -> dict[str, str]:
    """Read a small KEY=VALUE file without mutating or logging process secrets."""

    source = Path(path)
    values: dict[str, str] = {}
    allowed = {"OPENAI_BASE_URL", "OPENAI_API_KEY", "TEACHER_MODEL"}
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped.removeprefix("export ").lstrip()
        if "=" not in stripped:
            raise TeacherError(f"{source} 第 {line_number} 行缺少等号。")
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in allowed:
            raise TeacherError(f"{source} 第 {line_number} 行包含不允许的配置名。")
        if not value:
            raise TeacherError(f"{source} 第 {line_number} 行配置值为空。")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if not value:
            raise TeacherError(f"{source} 第 {line_number} 行配置值为空。")
        values[key] = value
    missing = allowed - set(values)
    if missing:
        raise TeacherError(f"教师配置文件缺少：{', '.join(sorted(missing))}。")
    return values


def parse_teacher_action(content: Any) -> AgentAction:
    if isinstance(content, Mapping):
        value = content
    elif isinstance(content, str):
        try:
            value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise TeacherError("教师返回的内容不是合法 JSON。") from exc
    else:
        raise TeacherError("教师返回必须是 JSON 对象或 JSON 字符串。")
    if not isinstance(value, Mapping):
        raise TeacherError("教师动作必须是 JSON 对象。")
    try:
        action = AgentAction.from_dict(value)
        validate_action_schema(action)
        return action
    except ValueError as exc:
        raise TeacherError(f"教师动作不符合 FinSight Schema：{exc}") from exc


def _message_for_api(message: Mapping[str, Any]) -> dict[str, str]:
    role = str(message["role"])
    content = message.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    if role == "tool":
        # Plain OpenAI-compatible chat endpoints commonly require tool_call_id
        # for native tool messages. The policy only needs the visible content,
        # so encode observations as user turns for broad compatibility.
        return {"role": "user", "content": f"TOOL_OBSERVATION\n{content}"}
    return {"role": role, "content": content}


@dataclass
class OpenAICompatibleTeacher:
    base_url: str
    api_key: str
    model_name: str
    timeout_seconds: float = 60.0
    max_retries: int = 2
    temperature: float | None = 0.0
    response_format: dict[str, Any] | None = field(
        default_factory=lambda: {"type": "json_object"}
    )
    extra_body: dict[str, Any] = field(default_factory=dict)
    prompt_version: str = TEACHER_PROMPT_VERSION
    system_prompt: str = TEACHER_SYSTEM_PROMPT
    schema_repairs: int = 0
    api_calls: int = 0
    total_latency_seconds: float = 0.0
    last_usage: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        *,
        base_url_variable: str = "OPENAI_BASE_URL",
        api_key_variable: str = "OPENAI_API_KEY",
        model_variable: str = "TEACHER_MODEL",
    ) -> OpenAICompatibleTeacher:
        base_url = os.environ.get(base_url_variable, "").strip()
        api_key = os.environ.get(api_key_variable, "").strip()
        model_name = os.environ.get(model_variable, "").strip()
        if not base_url or not api_key or not model_name:
            raise TeacherError(
                f"必须配置 {base_url_variable}、{api_key_variable} 和 {model_variable}。"
            )
        return cls(base_url=base_url, api_key=api_key, model_name=model_name)

    @classmethod
    def from_config_file(cls, path: str | Path) -> OpenAICompatibleTeacher:
        config = load_teacher_config_file(path)
        return cls(
            base_url=config["OPENAI_BASE_URL"],
            api_key=config["OPENAI_API_KEY"],
            model_name=config["TEACHER_MODEL"],
        )

    def next_action(self, messages: Sequence[Mapping[str, Any]]) -> AgentAction:
        endpoint = self.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        last_error: Exception | None = None
        correction: str | None = None
        for attempt in range(self.max_retries + 1):
            api_messages = [
                {"role": "system", "content": self.system_prompt},
                *[_message_for_api(message) for message in messages],
            ]
            if correction:
                api_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "上一输出未通过动作 Schema，未发送给环境。错误："
                            f"{correction} 请只返回修正后的单个 JSON 动作。"
                        ),
                    }
                )
            payload: dict[str, Any] = {
                "model": self.model_name,
                "messages": api_messages,
            }
            if self.temperature is not None:
                payload["temperature"] = self.temperature
            if self.response_format is not None:
                payload["response_format"] = dict(self.response_format)
            payload.update(self.extra_body)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                endpoint,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            request_started = time.perf_counter()
            self.api_calls += 1
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                usage = data.get("usage", {})
                self.last_usage = dict(usage) if isinstance(usage, Mapping) else {}
                content = data["choices"][0]["message"]["content"]
                return parse_teacher_action(content)
            except TeacherError as exc:
                last_error = exc
                correction = str(exc)
                self.schema_repairs += 1
            except (KeyError, IndexError, TypeError, ValueError, urllib.error.URLError) as exc:
                last_error = exc
            finally:
                self.total_latency_seconds += time.perf_counter() - request_started
            if attempt < self.max_retries:
                time.sleep(2**attempt)
        raise TeacherError(f"教师接口调用失败：{last_error}") from last_error


@dataclass
class ScriptedTeacher:
    """Deterministic offline teacher for collector integration tests."""

    actions: Sequence[AgentAction | Mapping[str, Any]]
    model_name: str = "scripted-teacher-v1"
    prompt_version: str = "scripted-teacher-v1"
    schema_repairs: int = 0

    def __post_init__(self) -> None:
        self._cursor = 0

    def next_action(self, messages: Sequence[Mapping[str, Any]]) -> AgentAction:
        del messages
        if self._cursor >= len(self.actions):
            raise TeacherError("脚本教师动作已经耗尽。")
        value = self.actions[self._cursor]
        self._cursor += 1
        return value if isinstance(value, AgentAction) else parse_teacher_action(value)


def scripted_oracle_teacher(sample: FinQASample) -> ScriptedTeacher:
    """Gold-assisted policy used only to test the collector without network calls."""

    evidence_by_id = {item.evidence_id: item for item in sample.evidences}
    actions: list[AgentAction] = []
    for evidence_id in sample.gold_evidence_ids:
        evidence = evidence_by_id[evidence_id]
        actions.extend(
            [
                AgentAction(
                    ActionName.SEARCH,
                    {
                        "query": f"{evidence.text} {evidence_id}",
                        "evidence_kind": evidence.kind,
                        "top_k": 5,
                    },
                ),
                AgentAction(ActionName.OPEN_EVIDENCE, {"evidence_id": evidence_id}),
            ]
        )
    actions.extend(
        [
            AgentAction(
                ActionName.SELECT_EVIDENCE,
                {"evidence_ids": list(sample.gold_evidence_ids)},
            ),
            AgentAction(ActionName.EMIT_PROGRAM, {"program": sample.program}),
            AgentAction(ActionName.CALCULATE),
            AgentAction(ActionName.VERIFY),
            AgentAction(ActionName.DELIVER),
        ]
    )
    return ScriptedTeacher(actions, model_name="scripted-oracle-teacher-v1")
