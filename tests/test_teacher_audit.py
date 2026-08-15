from finsight.data import load_finqa
from finsight.training.collection import collect_teacher_trajectory
from finsight.training.teacher import scripted_oracle_teacher
from finsight.training.teacher_audit import audit_teacher_records


def test_teacher_audit_separates_task_success_from_clean_sft_trace():
    sample = load_finqa("data/raw/finqa/train.json", limit=1)[0]
    clean = collect_teacher_trajectory(sample, scripted_oracle_teacher(sample)).raw
    unclean = {**clean, "accepted": False, "rejection_reasons": ["action_not_completed"]}
    unclean["diagnosis"] = {"primary_error": "action_rejected"}

    report = audit_teacher_records(
        [clean, unclean],
        {sample.sample_id: sample},
    )

    assert report["clean_sft_candidates"] == 1
    assert report["clean_sft_rate"] == 0.5
    assert report["task_successes"] == 2
    assert report["task_success_rate"] == 1.0
    assert report["verdicts"] == {
        "clean_sft_candidate": 1,
        "task_succeeded_but_trace_unclean": 1,
    }
