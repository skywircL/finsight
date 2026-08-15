from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from finsight.models import EventStatus


class ActionName(StrEnum):
    SEARCH = "search"
    OPEN_EVIDENCE = "open_evidence"
    SELECT_EVIDENCE = "select_evidence"
    EMIT_PROGRAM = "emit_program"
    CALCULATE = "calculate"
    VERIFY = "verify"
    DELIVER = "deliver"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class AgentAction:
    name: ActionName
    arguments: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AgentAction:
        if set(value) - {"name", "arguments"}:
            raise ValueError("动作只允许 name 和 arguments 字段。")
        try:
            name = ActionName(value["name"])
        except (KeyError, ValueError) as exc:
            raise ValueError("未知或缺失的动作名称。") from exc
        arguments = value.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise ValueError("arguments 必须是对象。")
        return cls(name=name, arguments=dict(arguments))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": dict(self.arguments)}


def validate_action_schema(action: AgentAction, *, max_search_top_k: int = 10) -> None:
    """Validate the public action schema without depending on environment state."""

    arguments = dict(action.arguments)
    required: set[str] = set()
    optional: set[str] = set()
    if action.name == ActionName.SEARCH:
        required = {"query"}
        optional = {"evidence_kind", "top_k"}
    elif action.name == ActionName.OPEN_EVIDENCE:
        required = {"evidence_id"}
    elif action.name == ActionName.SELECT_EVIDENCE:
        required = {"evidence_ids"}
    elif action.name == ActionName.EMIT_PROGRAM:
        required = {"program"}
    elif action.name == ActionName.ABSTAIN:
        required = {"reason_code"}
    missing = required - set(arguments)
    extra = set(arguments) - required - optional
    if missing:
        raise ValueError(f"动作缺少参数：{', '.join(sorted(missing))}。")
    if extra:
        raise ValueError(f"动作包含多余参数：{', '.join(sorted(extra))}。")

    if action.name == ActionName.SEARCH:
        query = arguments["query"]
        evidence_kind = arguments.get("evidence_kind")
        top_k = arguments.get("top_k", 5)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search.query 必须是非空字符串。")
        if evidence_kind not in {None, "text", "table"}:
            raise ValueError("search.evidence_kind 只能是 text 或 table。")
        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not 1 <= top_k <= max_search_top_k
        ):
            raise ValueError(f"search.top_k 必须在 1 到 {max_search_top_k} 之间。")
    elif action.name == ActionName.OPEN_EVIDENCE:
        evidence_id = arguments["evidence_id"]
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError("open_evidence.evidence_id 必须是非空字符串。")
    elif action.name == ActionName.SELECT_EVIDENCE:
        evidence_ids = arguments["evidence_ids"]
        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or not all(isinstance(item, str) and item for item in evidence_ids)
            or len(evidence_ids) != len(set(evidence_ids))
        ):
            raise ValueError("select_evidence.evidence_ids 必须是无重复的非空字符串数组。")
    elif action.name == ActionName.EMIT_PROGRAM:
        program = arguments["program"]
        if not isinstance(program, str) or not program.strip() or len(program) > 2_000:
            raise ValueError("emit_program.program 必须是长度不超过 2000 的非空字符串。")
    elif action.name == ActionName.ABSTAIN:
        allowed_reasons = {
            "missing_evidence",
            "ambiguous_question",
            "unsupported_operation",
            "unsafe_request",
        }
        if arguments["reason_code"] not in allowed_reasons:
            raise ValueError("abstain.reason_code 不在白名单中。")


@dataclass(frozen=True)
class EnvironmentObservation:
    event_id: str
    status: EventStatus
    done: bool
    message: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    audit: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnvironmentEvent:
    event_id: str
    action: AgentAction
    status: EventStatus
    message: str
    observation: Mapping[str, Any]
    observation_audit: Mapping[str, Any]
    new_evidence: bool
    done: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
