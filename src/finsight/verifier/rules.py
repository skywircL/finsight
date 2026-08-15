from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Sequence

from finsight.models import RetrievedEvidence


@dataclass(frozen=True)
class Verification:
    passed: bool
    reason: str
    evidence_coverage: float


def verify_result(
    *,
    value: float,
    retrieved: Sequence[RetrievedEvidence],
    required_evidence_ids: Sequence[str],
    expected_value: float | None = None,
    minimum_score: float = 0.0,
    expected_rel_tol: float = 1e-5,
    expected_abs_tol: float = 1e-6,
) -> Verification:
    if not retrieved or max(item.score for item in retrieved) <= minimum_score:
        return Verification(False, "未检索到与任务相关的可靠证据。", 0.0)

    retrieved_ids = {item.evidence.evidence_id for item in retrieved if item.score > minimum_score}
    required = set(required_evidence_ids)
    coverage = len(required & retrieved_ids) / len(required) if required else 1.0
    if required and coverage < 1.0:
        return Verification(False, "关键证据未被完整召回。", coverage)
    if expected_value is not None and not isclose(
        value,
        expected_value,
        rel_tol=expected_rel_tol,
        abs_tol=expected_abs_tol,
    ):
        return Verification(False, "计算结果与验证值不一致。", coverage)
    return Verification(True, "数字、公式和证据检查通过。", coverage)
