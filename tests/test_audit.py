import pytest

from finsight.agent import FinSightAgent
from finsight.audit import compare_retrieval_runs, evaluate_result, score_result
from finsight.business_cases import load_business_cases
from finsight.evaluation import evaluate_business_suite


def test_verified_delivery_receives_full_deterministic_reward():
    task = load_business_cases()[0]
    result = FinSightAgent().run(task)
    reward = score_result(task, result)

    assert reward.reward_valid
    assert reward.outcome == "verified_delivery"
    assert reward.reward == 1.0
    assert reward.evidence_coverage == 1.0


def test_missing_input_receives_positive_correct_refusal_reward():
    task = load_business_cases()[3]
    result = FinSightAgent().run(task)
    reward = score_result(task, result)

    assert reward.expected_refusal
    assert reward.correct_refusal
    assert reward.outcome == "correct_refusal"
    assert reward.reward == 0.7
    assert evaluate_result(task, result)["trajectory_panel"]["state_path_valid"]


def test_four_panel_evaluation_is_auditable():
    task = load_business_cases()[0]
    evaluation = evaluate_result(task, FinSightAgent().run(task))

    assert evaluation["schema_version"] == "finsight-evaluation-v1"
    assert set(evaluation) == {
        "schema_version",
        "task_id",
        "outcome_panel",
        "evidence_panel",
        "trajectory_panel",
        "deterministic_panel",
    }
    trajectory = evaluation["trajectory_panel"]
    assert trajectory["event_ids_unique"]
    assert trajectory["state_path_valid"]
    assert trajectory["terminal_consistent"]


def test_business_suite_keeps_fixed_denominator():
    suite = evaluate_business_suite("data/business_eval/cases.json")

    assert suite["fixed_denominator"] == 4
    assert suite["suite_sha256"] == (
        "6661c3926814cbb5f18befa609986d5b090af06d5ddcf9c2701d15e7e7fd81f7"
    )
    assert suite["summary"]["delivered"] == 3
    assert suite["summary"]["correct_refusals"] == 1
    assert suite["summary"]["reward_valid"] == 4


def test_retrieval_comparison_reports_separate_deltas():
    baseline = {
        "dataset": "dev.json",
        "samples": 100,
        "top_k": 5,
        "recall_at_k": 0.80,
        "mrr": 0.70,
        "gold_evidence_coverage": 0.60,
        "table_evidence_recall": 0.70,
        "text_evidence_recall": 0.90,
    }
    candidate = {**baseline, "recall_at_k": 0.82, "table_evidence_recall": 0.75}

    comparison = compare_retrieval_runs(
        baseline,
        candidate,
        baseline_name="E2",
        candidate_name="E3",
    )

    assert comparison["fixed_denominator"] == 100
    assert comparison["metrics"]["recall_at_k"]["delta"] == pytest.approx(0.02)
    assert comparison["metrics"]["table_evidence_recall"]["delta"] == pytest.approx(0.05)
