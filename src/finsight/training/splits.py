from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping


DEFAULT_TRAIN_COUNTS = {
    "sft_train": 4_000,
    "sft_validation": 500,
    "grpo_train": 1_500,
    "grpo_validation": 251,
}


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_task_ids(path: str | Path) -> list[str]:
    with Path(path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{path} 必须包含 JSON 数组。")
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} 内存在重复 task_id。")
    return ids


def _stable_order(task_ids: list[str], seed: str) -> list[str]:
    return sorted(
        task_ids,
        key=lambda task_id: (sha256(f"{seed}\0{task_id}".encode()).hexdigest(), task_id),
    )


def _write_task_ids(path: Path, task_ids: list[str]) -> None:
    path.write_text(
        "".join(
            json.dumps({"task_id": task_id}, ensure_ascii=False, sort_keys=True) + "\n"
            for task_id in task_ids
        ),
        encoding="utf-8",
    )


def _assert_pairwise_disjoint(partitions: Mapping[str, list[str]]) -> None:
    owners: dict[str, str] = {}
    for partition, task_ids in partitions.items():
        for task_id in task_ids:
            if task_id in owners:
                raise ValueError(
                    f"task_id={task_id} 同时出现在 {owners[task_id]} 和 {partition}。"
                )
            owners[task_id] = partition


def build_split_artifacts(
    *,
    train_path: str | Path,
    dev_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    seed: str = "finsight-finqa-v1",
    train_counts: Mapping[str, int] = DEFAULT_TRAIN_COUNTS,
) -> dict[str, Any]:
    """Freeze task-disjoint candidate pools with reproducible file hashes."""

    required_names = tuple(DEFAULT_TRAIN_COUNTS)
    if set(train_counts) != set(required_names):
        raise ValueError(f"训练划分必须且只能包含：{', '.join(required_names)}。")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in train_counts.values()):
        raise ValueError("每个训练划分数量必须是非负整数。")

    source_paths = {
        "train": Path(train_path),
        "dev": Path(dev_path),
        "test": Path(test_path),
    }
    source_ids = {name: _load_task_ids(path) for name, path in source_paths.items()}
    _assert_pairwise_disjoint(source_ids)
    if sum(train_counts.values()) != len(source_ids["train"]):
        raise ValueError(
            "训练划分数量之和必须等于 FinQA train 总数，避免未归属或重复任务。"
        )

    ordered = _stable_order(source_ids["train"], seed)
    partitions: dict[str, list[str]] = {}
    cursor = 0
    for name in required_names:
        next_cursor = cursor + train_counts[name]
        partitions[name] = ordered[cursor:next_cursor]
        cursor = next_cursor
    partitions["checkpoint_selection"] = list(source_ids["dev"])
    partitions["frozen_test"] = list(source_ids["test"])
    _assert_pairwise_disjoint(partitions)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    partition_metadata: dict[str, dict[str, Any]] = {}
    for name, task_ids in partitions.items():
        file_path = destination / f"{name}.jsonl"
        _write_task_ids(file_path, task_ids)
        partition_metadata[name] = {
            "file": file_path.name,
            "count": len(task_ids),
            "sha256": file_sha256(file_path),
            "source_split": (
                "train"
                if name in train_counts
                else "dev"
                if name == "checkpoint_selection"
                else "test"
            ),
        }

    manifest = {
        "schema_version": "finsight-finqa-splits-v1",
        "frozen": True,
        "seed": seed,
        "algorithm": "sha256(seed\\0task_id), ascending fixed slices",
        "sources": {
            name: {
                "path": str(path),
                "count": len(source_ids[name]),
                "sha256": file_sha256(path),
            }
            for name, path in source_paths.items()
        },
        "partitions": partition_metadata,
        "checks": {
            "source_task_ids_unique": True,
            "source_splits_disjoint": True,
            "partitions_disjoint": True,
            "train_fully_partitioned": cursor == len(source_ids["train"]),
            "frozen_test_used_for_training": False,
        },
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_partition_ids(manifest_path: str | Path, partition: str) -> list[str]:
    """Load a partition and fail closed if count or content hash changed."""

    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "finsight-finqa-splits-v1":
        raise ValueError("不支持的数据划分 manifest 版本。")
    try:
        metadata = manifest["partitions"][partition]
    except KeyError as exc:
        raise ValueError(f"未知数据划分：{partition}。") from exc
    partition_path = manifest_file.parent / metadata["file"]
    if file_sha256(partition_path) != metadata["sha256"]:
        raise ValueError(f"{partition} 文件 SHA-256 与冻结 manifest 不一致。")
    rows = [
        json.loads(line)
        for line in partition_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(row["task_id"]) for row in rows]
    if len(ids) != metadata["count"] or len(ids) != len(set(ids)):
        raise ValueError(f"{partition} 的行数或 task_id 唯一性检查失败。")
    return ids
