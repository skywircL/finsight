from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from math import isclose
from pathlib import Path
from typing import Any, Iterable

from finsight.audit import diagnose_trajectory
from finsight.environment import ActionName, FinSightEnvironment
from finsight.models import EventStatus, FinQASample
from finsight.training.teacher import TeacherError, TeacherPolicy
from finsight.training.trajectories import SYSTEM_INSTRUCTION, numeric_answer, sample_to_task


@dataclass(frozen=True)
class TeacherRolloutResult:
    task_id: str
    accepted: bool
    raw: dict[str, Any]
    sft_row: dict[str, Any] | None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _evaluate_teacher_rollout(
    sample: FinQASample,
    environment: FinSightEnvironment,
) -> tuple[bool, list[str], bool, float]:
    reasons: list[str] = []
    expected = numeric_answer(sample)
    if expected is None:
        reasons.append("unsupported_non_numeric_answer")
    events = environment.events
    if any(event.status != EventStatus.COMPLETED for event in events):
        reasons.append("action_not_completed")
    if not environment.done or environment.termination_reason != "delivered":
        reasons.append("not_verified_delivery")
    final = events[-1].observation if events else {}
    delivered = final.get("answer")
    answer_match = (
        expected is not None
        and isinstance(delivered, (int, float))
        and isclose(float(delivered), expected, rel_tol=1e-3, abs_tol=1e-3)
    )
    if not answer_match:
        reasons.append("answer_mismatch")
    verifications = [event for event in events if event.action.name == ActionName.VERIFY]
    coverage = (
        float(verifications[-1].observation.get("evidence_coverage", 0.0))
        if verifications
        else 0.0
    )
    if coverage < 1.0:
        reasons.append("incomplete_gold_evidence")
    return not reasons, list(dict.fromkeys(reasons)), answer_match, coverage


def collect_teacher_trajectory(
    sample: FinQASample,
    teacher: TeacherPolicy,
    *,
    max_steps: int | None = None,
) -> TeacherRolloutResult:
    max_steps = max_steps or max(20, 2 * len(sample.gold_evidence_ids) + 8)
    expected = numeric_answer(sample)
    environment = FinSightEnvironment(
        sample_to_task(sample),
        table=sample.table,
        expected_value=expected,
        max_steps=max_steps,
    )
    initial = environment.initial_observation()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTION, "trainable": False},
        {
            "role": "user",
            "content": _canonical_json(
                {"question": sample.question, "observation": initial.to_dict()}
            ),
            "trainable": False,
        },
    ]
    teacher_error: str | None = None
    for _ in range(max_steps):
        if environment.done:
            break
        try:
            action = teacher.next_action(messages)
        except TeacherError as exc:
            teacher_error = str(exc)
            break
        messages.append(
            {
                "role": "assistant",
                "content": _canonical_json(action.to_dict()),
                "trainable": True,
            }
        )
        observation = environment.step(action)
        messages.append(
            {
                "role": "tool",
                "name": action.name,
                "content": _canonical_json(observation.to_dict()),
                "trainable": False,
            }
        )

    accepted, reasons, answer_match, coverage = _evaluate_teacher_rollout(
        sample, environment
    )
    if teacher_error:
        accepted = False
        reasons.insert(0, "infrastructure_error")
    trajectory = environment.trajectory()
    diagnosis = diagnose_trajectory(trajectory, rejection_reasons=reasons).to_dict()
    raw = {
        "schema_version": "finsight-teacher-rollout-v1",
        "task_id": sample.sample_id,
        "teacher_model": teacher.model_name,
        "teacher_prompt_version": teacher.prompt_version,
        "schema_repairs": teacher.schema_repairs,
        "accepted": accepted,
        "rejection_reasons": reasons,
        "teacher_error": teacher_error,
        "answer_match": answer_match,
        "evidence_coverage": coverage,
        "messages": messages,
        "trajectory": trajectory,
        "diagnosis": diagnosis,
    }
    if not accepted:
        return TeacherRolloutResult(sample.sample_id, False, raw, None)
    row = {
        "schema_version": "finsight-action-sft-v1",
        "task_id": sample.sample_id,
        "messages": messages,
        "metadata": {
            "builder": "teacher_environment_replay_v1",
            "teacher_model": teacher.model_name,
            "teacher_prompt_version": teacher.prompt_version,
            "schema_repairs": teacher.schema_repairs,
            "data_normalization": "finqa-normalizer-v2-table-zero",
            "environment": "finsight-environment-v1",
            "loss_policy": "assistant_actions_only",
            "event_count": len(environment.events),
        },
    }
    row["metadata"]["row_sha256"] = sha256(_canonical_json(row).encode()).hexdigest()
    return TeacherRolloutResult(sample.sample_id, True, raw, row)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            completed.add(str(json.loads(line)["task_id"]))
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"raw.jsonl 第 {line_number} 行损坏，拒绝断点续跑。") from exc
    return completed


def rebuild_teacher_artifacts(output_dir: str | Path) -> dict[str, Any]:
    destination = Path(output_dir)
    raw_path = destination / "raw.jsonl"
    records = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if raw_path.exists() else []
    accepted = [record for record in records if record.get("accepted")]
    rejected = [record for record in records if not record.get("accepted")]
    accepted_path = destination / "accepted.jsonl"
    rejected_path = destination / "rejected.jsonl"
    accepted_rows = []
    for record in accepted:
        row = {
            "schema_version": "finsight-action-sft-v1",
            "task_id": record["task_id"],
            "messages": record["messages"],
            "metadata": {
                "builder": "teacher_environment_replay_v1",
                "teacher_model": record["teacher_model"],
                "teacher_prompt_version": record.get("teacher_prompt_version", "unknown"),
                "schema_repairs": record.get("schema_repairs", 0),
                "data_normalization": "finqa-normalizer-v2-table-zero",
                "environment": "finsight-environment-v1",
                "loss_policy": "assistant_actions_only",
                "event_count": len(record["trajectory"]["events"]),
            },
        }
        row["metadata"]["row_sha256"] = sha256(_canonical_json(row).encode()).hexdigest()
        accepted_rows.append(row)
    accepted_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n"
            for row in accepted_rows
        ),
        encoding="utf-8",
    )
    rejected_path.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": record["task_id"],
                    "primary_error": record["diagnosis"]["primary_error"],
                    "rejection_reasons": record["rejection_reasons"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ) + "\n"
            for record in rejected
        ),
        encoding="utf-8",
    )
    primary = Counter(
        record["diagnosis"]["primary_error"] for record in rejected
    )
    delivered = [
        record
        for record in records
        if record.get("trajectory", {}).get("termination_reason") == "delivered"
    ]
    answer_matched = [record for record in records if record.get("answer_match") is True]
    fully_grounded = [
        record for record in records if record.get("evidence_coverage") == 1.0
    ]
    prompt_versions = Counter(
        record.get("teacher_prompt_version", "unknown") for record in records
    )
    summary = {
        "schema_version": "finsight-teacher-collection-summary-v1",
        "processed": len(records),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": len(accepted) / len(records) if records else 0.0,
        "task_success": {
            "delivered": len(delivered),
            "delivered_rate": len(delivered) / len(records) if records else 0.0,
            "answer_matched": len(answer_matched),
            "answer_match_rate": len(answer_matched) / len(records) if records else 0.0,
            "fully_grounded": len(fully_grounded),
            "fully_grounded_rate": len(fully_grounded) / len(records) if records else 0.0,
        },
        "schema_repairs": sum(int(record.get("schema_repairs", 0)) for record in records),
        "teacher_prompt_versions": dict(sorted(prompt_versions.items())),
        "primary_errors": dict(sorted(primary.items())),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def collect_teacher_dataset(
    samples: Iterable[FinQASample],
    teacher_factory,
    *,
    output_dir: str | Path,
    limit: int | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / "raw.jsonl"
    completed = _load_completed(raw_path)
    attempted = 0
    for sample in samples:
        if sample.sample_id in completed:
            continue
        if limit is not None and attempted >= limit:
            break
        teacher = teacher_factory(sample)
        result = collect_teacher_trajectory(sample, teacher, max_steps=max_steps)
        _append_jsonl(raw_path, result.raw)
        completed.add(sample.sample_id)
        attempted += 1
    summary = rebuild_teacher_artifacts(destination)
    summary["attempted_this_run"] = attempted
    summary["resumable_source"] = raw_path.name
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
