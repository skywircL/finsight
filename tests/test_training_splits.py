import json
from pathlib import Path

import pytest

from finsight.training.splits import build_split_artifacts, load_partition_ids


def _write_source(path: Path, prefix: str, count: int) -> None:
    path.write_text(
        json.dumps([{"id": f"{prefix}-{index}"} for index in range(count)]),
        encoding="utf-8",
    )


def test_frozen_splits_are_deterministic_and_disjoint(tmp_path):
    train = tmp_path / "train.json"
    dev = tmp_path / "dev.json"
    test = tmp_path / "test.json"
    _write_source(train, "train", 8)
    _write_source(dev, "dev", 2)
    _write_source(test, "test", 3)
    counts = {
        "sft_train": 4,
        "sft_validation": 1,
        "grpo_train": 2,
        "grpo_validation": 1,
    }
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = build_split_artifacts(
        train_path=train,
        dev_path=dev,
        test_path=test,
        output_dir=first_dir,
        seed="fixed-seed",
        train_counts=counts,
    )
    second = build_split_artifacts(
        train_path=train,
        dev_path=dev,
        test_path=test,
        output_dir=second_dir,
        seed="fixed-seed",
        train_counts=counts,
    )

    assert first["partitions"] == second["partitions"]
    partitions = {
        name: set(load_partition_ids(first_dir / "manifest.json", name))
        for name in first["partitions"]
    }
    all_ids = set().union(*partitions.values())
    assert len(all_ids) == sum(len(ids) for ids in partitions.values())
    assert len(partitions["sft_train"]) == 4
    assert partitions["checkpoint_selection"] == {"dev-0", "dev-1"}
    assert partitions["frozen_test"] == {"test-0", "test-1", "test-2"}


def test_partition_loader_detects_tampering(tmp_path):
    train = tmp_path / "train.json"
    dev = tmp_path / "dev.json"
    test = tmp_path / "test.json"
    _write_source(train, "train", 4)
    _write_source(dev, "dev", 1)
    _write_source(test, "test", 1)
    output = tmp_path / "splits"
    build_split_artifacts(
        train_path=train,
        dev_path=dev,
        test_path=test,
        output_dir=output,
        train_counts={
            "sft_train": 1,
            "sft_validation": 1,
            "grpo_train": 1,
            "grpo_validation": 1,
        },
    )
    with (output / "sft_train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"task_id":"injected"}\n')

    with pytest.raises(ValueError, match="SHA-256"):
        load_partition_ids(output / "manifest.json", "sft_train")
