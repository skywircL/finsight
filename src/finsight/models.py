from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class AgentState(StrEnum):
    INGESTED = "INGESTED"
    PLANNED = "PLANNED"
    RETRIEVED = "RETRIEVED"
    CALCULATED = "CALCULATED"
    VERIFIED = "VERIFIED"
    DELIVERED = "DELIVERED"
    REFUSED = "REFUSED"


class EventStatus(StrEnum):
    """Deterministic execution status used by trajectory audits."""

    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    text: str
    kind: str = "text"
    page: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedEvidence:
    evidence: Evidence
    score: float
    rank: int


@dataclass(frozen=True)
class FinQASample:
    sample_id: str
    question: str
    table: tuple[tuple[str, ...], ...]
    evidences: tuple[Evidence, ...]
    gold_evidence_ids: tuple[str, ...]
    program: str
    answer: float | str


@dataclass(frozen=True)
class AnalysisTask:
    task_id: str
    title: str
    question: str
    evidence: tuple[Evidence, ...]
    program: str | None
    expected_evidence_ids: tuple[str, ...]
    unit: str = "%"
    risk_note: str = ""
    required_terms: tuple[str, ...] = ()


@dataclass
class TraceEvent:
    state: AgentState
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    event_id: str = ""
    action: str = ""
    status: EventStatus = EventStatus.COMPLETED


@dataclass
class AnalysisResult:
    task_id: str
    state: AgentState
    answer: float | None
    unit: str
    formula: str | None
    retrieved: list[RetrievedEvidence]
    verified: bool
    refusal_reason: str | None
    report_markdown: str
    trace: list[TraceEvent]
    run_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
