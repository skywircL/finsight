import json
from copy import deepcopy

import pytest

from finsight.data import load_finqa
from finsight.training.sft import (
    build_sft_dataset,
    encode_action_only,
    validate_sft_row,
)
from finsight.training.trajectories import build_oracle_trajectory


def _accepted_rows(count: int):
    rows = []
    for sample in load_finqa("data/raw/finqa/train.json", limit=20):
        result = build_oracle_trajectory(sample)
        if result.accepted:
            rows.append(result.sft_row)
        if len(rows) == count:
            return rows
    raise AssertionError("测试数据中没有足够的数值型样本。")


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_validate_sft_row_checks_hash_actions_and_terminal_state():
    row = _accepted_rows(1)[0]
    audit = validate_sft_row(row)

    assert audit.valid
    assert audit.action_names[-1] == "deliver"
    assert audit.estimated_tokens > 0

    corrupted = deepcopy(row)
    corrupted["messages"][2]["content"] = '{"name":"calculate","arguments":{}}'
    bad_audit = validate_sft_row(corrupted)
    assert not bad_audit.valid
    assert "row_sha256_mismatch" in bad_audit.errors
    assert any("tool_name_mismatch" in error for error in bad_audit.errors)

    rejected_observation = deepcopy(row)
    rejected_observation["messages"][3]["content"] = json.dumps(
        {"status": "rejected", "done": False}
    )
    rejected_audit = validate_sft_row(rejected_observation)
    assert not rejected_audit.valid
    assert "message_3_action_not_completed" in rejected_audit.errors


class _CharacterTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(character) + 10 for character in text]


def test_encode_action_only_masks_non_assistant_messages():
    row = _accepted_rows(1)[0]
    encoded = encode_action_only(row, _CharacterTokenizer(), max_length=100_000)

    assert len(encoded["input_ids"]) == len(encoded["labels"])
    assert encoded["labels"][0] == -100
    assert encoded["target_token_count"] > 0
    assert encoded["target_token_count"] < encoded["tokenized_length"]
    assert sum(label != -100 for label in encoded["labels"]) == encoded[
        "target_token_count"
    ]


def test_build_sft_dataset_requires_explicit_oracle_opt_in(tmp_path):
    train_row, validation_row = _accepted_rows(2)
    train_path = tmp_path / "source_train.jsonl"
    validation_path = tmp_path / "source_validation.jsonl"
    _write_jsonl(train_path, [train_row])
    _write_jsonl(validation_path, [validation_row])

    with pytest.raises(ValueError, match="Oracle Gold Replay"):
        build_sft_dataset(
            train_path,
            validation_path,
            output_dir=tmp_path / "rejected",
        )

    manifest = build_sft_dataset(
        train_path,
        validation_path,
        output_dir=tmp_path / "accepted",
        allow_oracle=True,
    )
    assert manifest["task_overlap"] == 0
    assert manifest["oracle_present"]
    assert not manifest["formal_training_data_ready"]
    assert manifest["splits"]["train"]["rows"] == 1
    assert (tmp_path / "accepted" / "manifest.json").exists()
