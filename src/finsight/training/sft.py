from __future__ import annotations

import importlib.util
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping

from finsight.environment import AgentAction, validate_action_schema
from finsight.environment.projection import estimate_tokens
from finsight.training.splits import file_sha256


SFT_SCHEMA_VERSION = "finsight-action-sft-v1"
SFT_MANIFEST_VERSION = "finsight-sft-dataset-manifest-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def compute_sft_row_sha256(row: Mapping[str, Any]) -> str:
    """Recompute the hash stored before metadata.row_sha256 was attached."""

    canonical_row = dict(row)
    metadata = dict(canonical_row.get("metadata", {}))
    metadata.pop("row_sha256", None)
    canonical_row["metadata"] = metadata
    return sha256(_canonical_json(canonical_row).encode()).hexdigest()


@dataclass(frozen=True)
class SFTRowAudit:
    task_id: str | None
    valid: bool
    errors: tuple[str, ...]
    action_names: tuple[str, ...]
    estimated_tokens: int
    builder: str | None
    teacher_model: str | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["errors"] = list(self.errors)
        value["action_names"] = list(self.action_names)
        return value


def validate_sft_row(
    row: Any,
    *,
    max_context_tokens: int = 24_576,
) -> SFTRowAudit:
    """Validate one accepted trajectory without trusting its stored metadata."""

    errors: list[str] = []
    actions: list[str] = []
    if not isinstance(row, Mapping):
        return SFTRowAudit(None, False, ("row_not_object",), (), 0, None, None)

    task_id_value = row.get("task_id")
    task_id = task_id_value if isinstance(task_id_value, str) and task_id_value else None
    if task_id is None:
        errors.append("invalid_task_id")
    if row.get("schema_version") != SFT_SCHEMA_VERSION:
        errors.append("invalid_schema_version")

    metadata_value = row.get("metadata")
    metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
    if not metadata:
        errors.append("invalid_metadata")
    if metadata.get("loss_policy") != "assistant_actions_only":
        errors.append("invalid_loss_policy")

    messages_value = row.get("messages")
    messages = messages_value if isinstance(messages_value, list) else []
    if not messages:
        errors.append("invalid_messages")
    elif len(messages) < 4:
        errors.append("trajectory_too_short")

    expected_roles = ("system", "user")
    for index, role in enumerate(expected_roles):
        if index >= len(messages) or not isinstance(messages[index], Mapping):
            errors.append(f"missing_{role}_message")
            continue
        if messages[index].get("role") != role:
            errors.append(f"invalid_initial_role_{index}")

    parsed_actions: list[AgentAction] = []
    for index, message_value in enumerate(messages):
        if not isinstance(message_value, Mapping):
            errors.append(f"message_{index}_not_object")
            continue
        role = message_value.get("role")
        trainable = message_value.get("trainable")
        if role not in {"system", "user", "assistant", "tool"}:
            errors.append(f"message_{index}_invalid_role")
        if trainable is not (role == "assistant"):
            errors.append(f"message_{index}_invalid_trainable_flag")
        content = message_value.get("content")
        if not isinstance(content, str) or not content:
            errors.append(f"message_{index}_invalid_content")
            continue
        if index >= 2:
            expected_role = "assistant" if index % 2 == 0 else "tool"
            if role != expected_role:
                errors.append(f"message_{index}_invalid_turn_order")
        if role != "assistant":
            continue
        try:
            payload = json.loads(content)
            if not isinstance(payload, Mapping):
                raise ValueError("动作必须是对象。")
            action = AgentAction.from_dict(payload)
            validate_action_schema(action)
        except (json.JSONDecodeError, ValueError, TypeError):
            errors.append(f"message_{index}_invalid_action")
            continue
        parsed_actions.append(action)
        actions.append(str(action.name))
        if index + 1 >= len(messages) or not isinstance(messages[index + 1], Mapping):
            errors.append(f"message_{index}_missing_tool_result")
        elif messages[index + 1].get("name") != str(action.name):
            errors.append(f"message_{index + 1}_tool_name_mismatch")

    for index, message_value in enumerate(messages):
        if not isinstance(message_value, Mapping) or message_value.get("role") != "tool":
            continue
        try:
            observation = json.loads(str(message_value.get("content", "")))
            if not isinstance(observation, Mapping):
                raise ValueError("Observation 必须是对象。")
        except (json.JSONDecodeError, ValueError, TypeError):
            errors.append(f"message_{index}_invalid_observation")
            continue
        if observation.get("status") != "completed":
            errors.append(f"message_{index}_action_not_completed")

    if not parsed_actions:
        errors.append("missing_trainable_action")
    if len(messages) % 2:
        errors.append("incomplete_action_tool_pair")
    if parsed_actions and parsed_actions[-1].name not in {"deliver", "abstain"}:
        errors.append("invalid_terminal_action")
    if messages and isinstance(messages[-1], Mapping) and messages[-1].get("role") == "tool":
        try:
            terminal_observation = json.loads(str(messages[-1].get("content", "")))
        except json.JSONDecodeError:
            terminal_observation = {}
        if not isinstance(terminal_observation, Mapping) or terminal_observation.get("done") is not True:
            errors.append("terminal_observation_not_done")

    if metadata.get("event_count") != len(parsed_actions):
        errors.append("event_count_mismatch")
    stored_hash = metadata.get("row_sha256")
    if not isinstance(stored_hash, str) or stored_hash != compute_sft_row_sha256(row):
        errors.append("row_sha256_mismatch")

    token_estimate = estimate_tokens(messages)
    if token_estimate > max_context_tokens:
        errors.append("context_estimate_exceeded")
    unique_errors = tuple(dict.fromkeys(errors))
    return SFTRowAudit(
        task_id=task_id,
        valid=not unique_errors,
        errors=unique_errors,
        action_names=tuple(actions),
        estimated_tokens=token_estimate,
        builder=str(metadata["builder"]) if metadata.get("builder") else None,
        teacher_model=(
            str(metadata["teacher_model"]) if metadata.get("teacher_model") else None
        ),
    )


def _read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON。") from exc
    return rows


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * percentile + 0.999999) - 1))
    return ordered[index]


def _audit_rows(
    rows: Iterable[Any],
    *,
    max_context_tokens: int,
) -> tuple[list[Mapping[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    valid_rows: list[Mapping[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    builder_counts: Counter[str] = Counter()
    teacher_counts: Counter[str] = Counter()
    tokens: list[int] = []
    task_ids: set[str] = set()
    duplicate_task_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        audit = validate_sft_row(row, max_context_tokens=max_context_tokens)
        if audit.task_id in task_ids:
            duplicate_task_ids.add(str(audit.task_id))
        elif audit.task_id:
            task_ids.add(audit.task_id)
        if not audit.valid:
            invalid.append({"line": line_number, **audit.to_dict()})
            continue
        valid_rows.append(row)
        action_counts.update(audit.action_names)
        builder_counts.update([audit.builder or "unknown"])
        if audit.teacher_model:
            teacher_counts.update([audit.teacher_model])
        tokens.append(audit.estimated_tokens)

    if duplicate_task_ids:
        invalid.append(
            {
                "line": None,
                "task_id": None,
                "valid": False,
                "errors": ["duplicate_task_ids"],
                "duplicate_task_ids": sorted(duplicate_task_ids),
            }
        )
    stats = {
        "rows": len(valid_rows),
        "task_ids": len(task_ids),
        "actions": sum(action_counts.values()),
        "action_distribution": dict(sorted(action_counts.items())),
        "builders": dict(sorted(builder_counts.items())),
        "teacher_models": dict(sorted(teacher_counts.items())),
        "estimated_tokens": {
            "total": sum(tokens),
            "mean": round(sum(tokens) / len(tokens), 2) if tokens else 0.0,
            "p50": _percentile(tokens, 0.50),
            "p95": _percentile(tokens, 0.95),
            "max": max(tokens, default=0),
        },
    }
    return valid_rows, stats, invalid


def probe_training_runtime() -> dict[str, Any]:
    packages = ("torch", "transformers", "peft", "accelerate")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None

    accelerator = "unavailable"
    if importlib.util.find_spec("torch"):
        import torch

        if torch.cuda.is_available():
            accelerator = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            accelerator = "mps"
        else:
            accelerator = "cpu"
    teacher_config = {
        "endpoint_present": bool(os.environ.get("OPENAI_BASE_URL")),
        "api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
        "model_present": bool(os.environ.get("TEACHER_MODEL")),
    }
    return {
        "packages": versions,
        "dependencies_ready": all(versions.values()),
        "accelerator": accelerator,
        "accelerator_ready": accelerator in {"cuda", "mps"},
        "teacher_config": teacher_config,
        "teacher_config_ready": all(teacher_config.values()),
    }


def build_sft_dataset(
    train_source: str | Path,
    validation_source: str | Path,
    *,
    output_dir: str | Path,
    max_context_tokens: int = 24_576,
    allow_oracle: bool = False,
) -> dict[str, Any]:
    """Fail closed, then freeze canonical train/validation Action-only JSONL."""

    train_path = Path(train_source)
    validation_path = Path(validation_source)
    sources = {"train": train_path, "validation": validation_path}
    audited: dict[str, tuple[list[Mapping[str, Any]], dict[str, Any]]] = {}
    invalid_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, path in sources.items():
        rows, stats, invalid = _audit_rows(
            _read_jsonl(path), max_context_tokens=max_context_tokens
        )
        audited[split] = (rows, stats)
        if not rows:
            invalid.append(
                {
                    "line": None,
                    "task_id": None,
                    "valid": False,
                    "errors": ["empty_split"],
                }
            )
        if invalid:
            invalid_by_split[split] = invalid
    if invalid_by_split:
        counts = {split: len(errors) for split, errors in invalid_by_split.items()}
        raise ValueError(f"SFT 数据校验失败，错误记录数：{counts}")

    train_ids = {str(row["task_id"]) for row in audited["train"][0]}
    validation_ids = {str(row["task_id"]) for row in audited["validation"][0]}
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        raise ValueError(f"SFT train/validation 有 {len(overlap)} 个 task_id 重叠。")

    all_builders = {
        builder
        for _, stats in audited.values()
        for builder in stats["builders"]
    }
    oracle_present = any(builder.startswith("oracle_") for builder in all_builders)
    if oracle_present and not allow_oracle:
        raise ValueError(
            "检测到 Oracle Gold Replay；仅可用 --allow-oracle 明确生成管线冒烟数据。"
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    split_manifests: dict[str, Any] = {}
    for split, source_path in sources.items():
        rows, stats = audited[split]
        output_path = destination / f"{split}.jsonl"
        output_path.write_text(
            "".join(_canonical_json(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        split_manifests[split] = {
            **stats,
            "source": str(source_path),
            "source_sha256": file_sha256(source_path),
            "file": output_path.name,
            "sha256": file_sha256(output_path),
        }

    warnings: list[str] = []
    if oracle_present:
        warnings.append("oracle_data_is_pipeline_smoke_only")
    runtime = probe_training_runtime()
    if not runtime["accelerator_ready"]:
        warnings.append("no_supported_training_accelerator_detected")
    if not runtime["teacher_config_ready"]:
        warnings.append("real_teacher_config_missing")
    manifest = {
        "schema_version": SFT_MANIFEST_VERSION,
        "loss_policy": "assistant_actions_only",
        "max_context_tokens": max_context_tokens,
        "task_overlap": 0,
        "oracle_present": oracle_present,
        "formal_training_data_ready": not oracle_present,
        "splits": split_manifests,
        "runtime": runtime,
        "warnings": warnings,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _message_header(message: Mapping[str, Any]) -> str:
    role = str(message["role"])
    if role == "tool":
        return f"<|tool:{message.get('name', 'unknown')}|>\n"
    return f"<|{role}|>\n"


def encode_action_only(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
) -> dict[str, Any]:
    """Tokenize one trace and mask every target except assistant action JSON.

    The function deliberately rejects overflow instead of truncating a trajectory,
    because silent truncation can detach an action from its Observation.
    """

    audit = validate_sft_row(row, max_context_tokens=max_length * 4)
    non_context_errors = tuple(
        error for error in audit.errors if error != "context_estimate_exceeded"
    )
    if non_context_errors:
        raise ValueError(f"无效 SFT 行：{', '.join(non_context_errors)}")
    input_ids: list[int] = []
    labels: list[int] = []
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if bos_token_id is not None:
        input_ids.append(int(bos_token_id))
        labels.append(-100)

    for message in row["messages"]:
        header_ids = list(tokenizer.encode(_message_header(message), add_special_tokens=False))
        content_ids = list(
            tokenizer.encode(str(message["content"]) + "\n", add_special_tokens=False)
        )
        input_ids.extend(header_ids)
        labels.extend([-100] * len(header_ids))
        input_ids.extend(content_ids)
        if message["role"] == "assistant":
            labels.extend(content_ids)
            if eos_token_id is not None:
                input_ids.append(int(eos_token_id))
                labels.append(int(eos_token_id))
        else:
            labels.extend([-100] * len(content_ids))

    if len(input_ids) > max_length:
        raise ValueError(
            f"tokenized_length={len(input_ids)} 超过 max_length={max_length}，拒绝静默截断。"
        )
    target_count = sum(label != -100 for label in labels)
    if target_count == 0:
        raise ValueError("SFT 行没有可训练的 Assistant Action token。")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "tokenized_length": len(input_ids),
        "target_token_count": target_count,
    }
