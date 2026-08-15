from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from finsight.models import FinQASample


def audit_teacher_records(
    records: Iterable[Mapping[str, Any]],
    samples: Mapping[str, FinQASample],
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    failed_action_counts: Counter[str] = Counter()
    program_attempts = 0
    searches = 0
    schema_repairs = 0
    for record in records:
        task_id = str(record.get("task_id", ""))
        sample = samples.get(task_id)
        trajectory = record.get("trajectory", {})
        events = trajectory.get("events", []) if isinstance(trajectory, Mapping) else []
        actions: list[str] = []
        failed_events: list[dict[str, Any]] = []
        programs: list[str] = []
        search_queries: list[str] = []
        selected_history: list[list[str]] = []
        verification_history: list[dict[str, Any]] = []
        for event in events:
            action = event.get("action", {})
            name = str(action.get("name", "unknown"))
            arguments = action.get("arguments", {})
            actions.append(name)
            action_counts[name] += 1
            if name == "search":
                searches += 1
                query = arguments.get("query") if isinstance(arguments, Mapping) else None
                search_queries.append(str(query) if query is not None else "")
            elif name == "emit_program":
                program_attempts += 1
                program = arguments.get("program") if isinstance(arguments, Mapping) else None
                if program is not None:
                    programs.append(str(program))
            elif name == "select_evidence" and isinstance(arguments, Mapping):
                selected = arguments.get("evidence_ids")
                if isinstance(selected, list):
                    selected_history.append([str(item) for item in selected])
            elif name == "verify":
                observation = event.get("observation", {})
                if isinstance(observation, Mapping):
                    verification_history.append(dict(observation))
            if str(event.get("status")) != "completed":
                failed_action_counts[name] += 1
                failed_events.append(
                    {
                        "event_id": event.get("event_id"),
                        "action": action,
                        "status": event.get("status"),
                        "message": event.get("message"),
                    }
                )

        accepted = record.get("accepted") is True
        answer_match = record.get("answer_match") is True
        fully_grounded = record.get("evidence_coverage") == 1.0
        delivered = trajectory.get("termination_reason") == "delivered"
        schema_repairs += int(record.get("schema_repairs", 0))
        if accepted:
            verdict = "clean_sft_candidate"
        elif delivered and answer_match and fully_grounded:
            verdict = "task_succeeded_but_trace_unclean"
        elif record.get("teacher_error"):
            verdict = "infrastructure_failure"
        else:
            verdict = "task_failure"
        cases.append(
            {
                "task_id": task_id,
                "question": sample.question if sample else None,
                "verdict": verdict,
                "accepted": accepted,
                "delivered": delivered,
                "answer_match": answer_match,
                "evidence_coverage": record.get("evidence_coverage"),
                "steps": len(events),
                "termination_reason": trajectory.get("termination_reason"),
                "primary_error": record.get("diagnosis", {}).get("primary_error"),
                "rejection_reasons": record.get("rejection_reasons", []),
                "schema_repairs": record.get("schema_repairs", 0),
                "search_queries": search_queries,
                "selected_history": selected_history,
                "programs": programs,
                "verification_history": verification_history,
                "failed_events": failed_events,
            }
        )

    total = len(cases)
    verdicts = Counter(case["verdict"] for case in cases)
    return {
        "schema_version": "finsight-teacher-quality-audit-v1",
        "samples": total,
        "clean_sft_candidates": verdicts["clean_sft_candidate"],
        "clean_sft_rate": verdicts["clean_sft_candidate"] / total if total else 0.0,
        "task_successes": sum(case["delivered"] and case["answer_match"] for case in cases),
        "task_success_rate": (
            sum(case["delivered"] and case["answer_match"] for case in cases) / total
            if total
            else 0.0
        ),
        "fully_grounded": sum(case["evidence_coverage"] == 1.0 for case in cases),
        "fully_grounded_rate": (
            sum(case["evidence_coverage"] == 1.0 for case in cases) / total
            if total
            else 0.0
        ),
        "verdicts": dict(sorted(verdicts.items())),
        "actions": sum(action_counts.values()),
        "action_distribution": dict(sorted(action_counts.items())),
        "failed_action_distribution": dict(sorted(failed_action_counts.items())),
        "searches": searches,
        "program_attempts": program_attempts,
        "schema_repairs": schema_repairs,
        "cases": cases,
    }


def audit_teacher_output(
    output_dir: str | Path,
    samples: Iterable[FinQASample],
) -> dict[str, Any]:
    destination = Path(output_dir)
    raw_path = destination / "raw.jsonl"
    records = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = audit_teacher_records(records, {sample.sample_id: sample for sample in samples})
    report_path = destination / "quality_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
