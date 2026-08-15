from __future__ import annotations

import json
from pathlib import Path

from finsight.models import AnalysisTask, Evidence


DEFAULT_CASES = Path(__file__).parents[2] / "data" / "business_eval" / "cases.json"


def load_business_cases(path: str | Path = DEFAULT_CASES) -> list[AnalysisTask]:
    with Path(path).open(encoding="utf-8") as handle:
        raw_cases = json.load(handle)
    cases = []
    for item in raw_cases:
        cases.append(
            AnalysisTask(
                task_id=item["task_id"],
                title=item["title"],
                question=item["question"],
                evidence=tuple(Evidence(**entry) for entry in item["evidence"]),
                program=item.get("program"),
                expected_evidence_ids=tuple(item.get("expected_evidence_ids", [])),
                unit=item.get("unit", "%"),
                risk_note=item.get("risk_note", ""),
                required_terms=tuple(item.get("required_terms", [])),
            )
        )
    return cases

