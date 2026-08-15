from dataclasses import replace

from finsight.data import load_finqa
from finsight.training.trajectories import build_oracle_dataset, build_oracle_trajectory


def test_oracle_trajectory_is_replayed_and_action_only():
    sample = load_finqa("data/raw/finqa/train.json", limit=1)[0]
    result = build_oracle_trajectory(sample)

    assert result.accepted
    assert result.sft_row is not None
    row = result.sft_row
    assert row["metadata"]["loss_policy"] == "assistant_actions_only"
    assert all(
        message["trainable"] == (message["role"] == "assistant")
        for message in row["messages"]
    )
    user_content = row["messages"][1]["content"]
    assert "expected_evidence_ids" not in user_content
    assert '"program"' not in user_content
    acceptance = result.audit["acceptance"]
    assert acceptance["answer_match"]
    assert acceptance["evidence_coverage"] == 1.0
    assert acceptance["termination_reason"] == "delivered"
    context = result.audit["trajectory"]["context"]
    assert context["estimated_tokens_used"] > 0
    assert not context["context_overflow"]


def test_oracle_rejects_non_numeric_finqa_answers():
    samples = load_finqa("data/raw/finqa/train.json")
    boolean_sample = next(sample for sample in samples if isinstance(sample.answer, str))
    result = build_oracle_trajectory(boolean_sample)

    assert not result.accepted
    assert result.sft_row is None
    assert result.audit["acceptance"]["rejection_reasons"] == [
        "unsupported_non_numeric_answer"
    ]


def test_oracle_acceptance_checks_executed_answer_not_program_string():
    sample = load_finqa("data/raw/finqa/train.json", limit=1)[0]
    bad_sample = replace(sample, program="add(1, 1)")
    result = build_oracle_trajectory(bad_sample)

    assert not result.accepted
    assert "not_verified_delivery" in result.audit["acceptance"]["rejection_reasons"]
    assert "answer_mismatch" in result.audit["acceptance"]["rejection_reasons"]


def test_oracle_dataset_reports_fixed_denominator():
    samples = load_finqa("data/raw/finqa/train.json", limit=3)
    rows, audits, summary = build_oracle_dataset(samples)

    assert summary["processed"] == 3
    assert summary["accepted"] + summary["rejected"] == 3
    assert len(rows) == summary["accepted"]
    assert len(audits) == 3
