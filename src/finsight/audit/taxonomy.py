from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping


class ErrorCategory(StrEnum):
    NONE = "none"
    TASK = "task"
    PROTOCOL = "protocol"
    RETRIEVAL = "retrieval"
    EVIDENCE = "evidence"
    PROGRAM = "program"
    CALCULATION = "calculation"
    VERIFICATION = "verification"
    TERMINATION = "termination"
    CONTEXT = "context"
    INFRASTRUCTURE = "infrastructure"


@dataclass(frozen=True)
class TrajectoryDiagnosis:
    version: str
    valid: bool
    primary_error: str | None
    secondary_errors: tuple[str, ...]
    primary_category: ErrorCategory
    failing_event_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["primary_category"] = self.primary_category.value
        value["secondary_errors"] = list(self.secondary_errors)
        value["failing_event_ids"] = list(self.failing_event_ids)
        return value


ERROR_PRIORITY = (
    "infrastructure_error",
    "task_mismatch",
    "unsupported_non_numeric_answer",
    "context_overflow",
    "max_steps",
    "repeat_loop",
    "no_progress_loop",
    "invalid_action",
    "action_rejected",
    "calculation_error",
    "unsafe_program",
    "missing_program",
    "gold_evidence_not_selected",
    "incomplete_gold_evidence",
    "answer_mismatch",
    "not_verified_delivery",
    "premature_refusal",
)


ERROR_CATEGORIES = {
    "infrastructure_error": ErrorCategory.INFRASTRUCTURE,
    "task_mismatch": ErrorCategory.TASK,
    "unsupported_non_numeric_answer": ErrorCategory.TASK,
    "context_overflow": ErrorCategory.CONTEXT,
    "max_steps": ErrorCategory.TERMINATION,
    "repeat_loop": ErrorCategory.TERMINATION,
    "no_progress_loop": ErrorCategory.TERMINATION,
    "invalid_action": ErrorCategory.PROTOCOL,
    "action_rejected": ErrorCategory.PROTOCOL,
    "action_not_completed": ErrorCategory.PROTOCOL,
    "calculation_error": ErrorCategory.CALCULATION,
    "unsafe_program": ErrorCategory.PROGRAM,
    "missing_program": ErrorCategory.PROGRAM,
    "gold_evidence_not_selected": ErrorCategory.EVIDENCE,
    "incomplete_gold_evidence": ErrorCategory.RETRIEVAL,
    "answer_mismatch": ErrorCategory.VERIFICATION,
    "not_verified_delivery": ErrorCategory.TERMINATION,
    "premature_refusal": ErrorCategory.TERMINATION,
}


def _event_value(event: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = event.get(key, default)
    return value.value if hasattr(value, "value") else value


def diagnose_trajectory(
    trajectory: Mapping[str, Any] | None,
    *,
    rejection_reasons: tuple[str, ...] | list[str] = (),
) -> TrajectoryDiagnosis:
    """Classify failures with a frozen deterministic priority, never an LLM."""

    if trajectory is None:
        errors = list(dict.fromkeys(rejection_reasons))
        primary = next((code for code in ERROR_PRIORITY if code in errors), None)
        primary = primary or (errors[0] if errors else "infrastructure_error")
        secondary = tuple(code for code in errors if code != primary)
        return TrajectoryDiagnosis(
            version="finsight-error-taxonomy-v1",
            valid=False,
            primary_error=primary,
            secondary_errors=secondary,
            primary_category=ERROR_CATEGORIES.get(primary, ErrorCategory.INFRASTRUCTURE),
            failing_event_ids=(),
        )

    events = trajectory.get("events", [])
    event_errors: list[tuple[str, str]] = []
    for event in events:
        event_id = str(event.get("event_id", ""))
        status = _event_value(event, "status")
        observation = event.get("observation", {})
        reason = observation.get("reason_code") if isinstance(observation, Mapping) else None
        if reason:
            event_errors.append((str(reason), event_id))
        elif status in {"failed", "rejected"}:
            event_errors.append(("action_not_completed", event_id))

    errors = list(dict.fromkeys([code for code, _ in event_errors] + list(rejection_reasons)))
    primary = next((code for code in ERROR_PRIORITY if code in errors), None)
    if primary is None and trajectory.get("termination_reason") not in {None, "delivered"}:
        termination = str(trajectory["termination_reason"]).split(":", 1)[0]
        primary = termination if termination in ERROR_CATEGORIES else "premature_refusal"
        errors.append(primary)
    valid = not errors and trajectory.get("termination_reason") == "delivered"
    if valid:
        return TrajectoryDiagnosis(
            version="finsight-error-taxonomy-v1",
            valid=True,
            primary_error=None,
            secondary_errors=(),
            primary_category=ErrorCategory.NONE,
            failing_event_ids=(),
        )
    primary = primary or "not_verified_delivery"
    secondary = tuple(code for code in errors if code != primary)
    failing_ids = tuple(event_id for code, event_id in event_errors if code == primary)
    return TrajectoryDiagnosis(
        version="finsight-error-taxonomy-v1",
        valid=False,
        primary_error=primary,
        secondary_errors=secondary,
        primary_category=ERROR_CATEGORIES.get(primary, ErrorCategory.INFRASTRUCTURE),
        failing_event_ids=failing_ids,
    )
