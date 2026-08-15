from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from finsight.models import Evidence, FinQASample


_SPACE_RE = re.compile(r"\s+")


def clean_text(value: Any) -> str:
    """Normalize spacing without discarding financial signs, units, or punctuation."""
    return _SPACE_RE.sub(" ", str(value).replace("\u00a0", " ")).strip()


def serialize_table_row(header: list[Any], row: list[Any]) -> str:
    """Serialize a FinQA table row while preserving row/column relationships."""
    if not row:
        return ""
    row_name = clean_text(row[0])
    parts = []
    for index, value in enumerate(row[1:], start=1):
        column = clean_text(header[index]) if index < len(header) else f"column_{index}"
        parts.append(f"the {row_name} of {column} is {clean_text(value)}")
    return f"{clean_text(header[0])} " + " ; ".join(parts) + " ;"


def record_evidences(record: dict[str, Any]) -> tuple[Evidence, ...]:
    evidence: list[Evidence] = []
    text_rows = [*record.get("pre_text", []), *record.get("post_text", [])]
    for index, text in enumerate(text_rows):
        cleaned = clean_text(text)
        if cleaned:
            evidence.append(
                Evidence(
                    evidence_id=f"text_{index}",
                    text=cleaned,
                    kind="text",
                    source=record.get("id"),
                )
            )

    table = record.get("table", [])
    if table:
        header = table[0]
        # FinQA indexes every original table row from table_0. Some source tables
        # have no explicit header, so dropping row 0 makes valid Gold evidence
        # impossible to retrieve. Serializing the first row against itself also
        # matches FinQA's published table-to-text convention.
        for index, row in enumerate(table):
            serialized = serialize_table_row(header, row)
            if serialized:
                evidence.append(
                    Evidence(
                        evidence_id=f"table_{index}",
                        text=serialized,
                        kind="table",
                        source=record.get("id"),
                        metadata={
                            "table_index": index,
                            "row": [clean_text(value) for value in row],
                            "row_label": clean_text(row[0]) if row else "",
                            "values": [clean_text(value) for value in row[1:]],
                            "header": [clean_text(value) for value in header],
                        },
                    )
                )
    return tuple(evidence)


def normalize_finqa_record(record: dict[str, Any]) -> FinQASample:
    qa = record["qa"]
    return FinQASample(
        sample_id=str(record["id"]),
        question=clean_text(qa["question"]),
        table=tuple(tuple(clean_text(cell) for cell in row) for row in record.get("table", [])),
        evidences=record_evidences(record),
        gold_evidence_ids=tuple(qa.get("gold_inds", {}).keys()),
        program=clean_text(qa.get("program", "")),
        answer=qa.get("exe_ans", qa.get("answer", "")),
    )


def load_finqa(path: str | Path, *, limit: int | None = None) -> list[FinQASample]:
    with Path(path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("FinQA file must contain a JSON list")
    selected: Iterable[dict[str, Any]] = records if limit is None else records[:limit]
    return [normalize_finqa_record(record) for record in selected]
