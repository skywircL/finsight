from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping


_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def estimate_tokens(value: Any) -> int:
    """Dependency-free conservative estimate for mixed Chinese/English JSON."""

    text = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    cjk_count = len(_CJK.findall(text))
    non_cjk_count = len(text) - cjk_count
    return cjk_count + math.ceil(non_cjk_count / 4)


@dataclass(frozen=True)
class ProjectionAudit:
    original_tokens_estimate: int
    emitted_tokens_estimate: int
    observation_limit: int
    context_remaining_before: int
    truncated: bool
    overflow: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _truncate_strings(value: Any, cap: int) -> Any:
    if isinstance(value, str):
        if len(value) <= cap:
            return value
        if cap <= 0:
            return ""
        marker = "…[truncated]"
        if cap <= len(marker):
            return marker[:cap]
        return value[: cap - len(marker)] + marker
    if isinstance(value, list):
        return [_truncate_strings(item, cap) for item in value]
    if isinstance(value, tuple):
        return [_truncate_strings(item, cap) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _truncate_strings(item, cap) for key, item in value.items()}
    return value


def _max_string_length(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (list, tuple)):
        return max((_max_string_length(item) for item in value), default=0)
    if isinstance(value, Mapping):
        return max((_max_string_length(item) for item in value.values()), default=0)
    return 0


def project_payload(
    payload: Mapping[str, Any],
    *,
    observation_limit: int,
    context_remaining: int,
) -> tuple[dict[str, Any], ProjectionAudit]:
    if observation_limit < 1:
        raise ValueError("observation_limit 必须为正整数。")
    original = dict(payload)
    original_tokens = estimate_tokens(original)
    allowance = max(0, min(observation_limit, context_remaining))
    if original_tokens <= allowance:
        return original, ProjectionAudit(
            original_tokens_estimate=original_tokens,
            emitted_tokens_estimate=original_tokens,
            observation_limit=observation_limit,
            context_remaining_before=max(0, context_remaining),
            truncated=False,
            overflow=False,
        )

    if allowance == 0:
        return {}, ProjectionAudit(
            original_tokens_estimate=original_tokens,
            emitted_tokens_estimate=0,
            observation_limit=observation_limit,
            context_remaining_before=0,
            truncated=True,
            overflow=True,
        )

    low = 0
    high = _max_string_length(original)
    best: dict[str, Any] | None = None
    while low <= high:
        middle = (low + high) // 2
        candidate = _truncate_strings(original, middle)
        if estimate_tokens(candidate) <= allowance:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1

    overflow = False
    if best is None:
        minimal = {"projection": "omitted"}
        if estimate_tokens(minimal) <= allowance:
            best = minimal
        else:
            best = {}
            overflow = True
    emitted_tokens = estimate_tokens(best) if best else 0
    return best, ProjectionAudit(
        original_tokens_estimate=original_tokens,
        emitted_tokens_estimate=emitted_tokens,
        observation_limit=observation_limit,
        context_remaining_before=max(0, context_remaining),
        truncated=True,
        overflow=overflow,
    )
