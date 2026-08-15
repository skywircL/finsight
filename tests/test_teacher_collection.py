import json
from unittest.mock import patch

import pytest

from finsight.data import load_finqa
from finsight.environment import ActionName, AgentAction
from finsight.training.collection import collect_teacher_dataset, collect_teacher_trajectory
from finsight.training.teacher import (
    OpenAICompatibleTeacher,
    ScriptedTeacher,
    TEACHER_SYSTEM_PROMPT,
    TeacherError,
    parse_teacher_action,
    scripted_oracle_teacher,
)


def test_teacher_action_parser_rejects_extra_fields():
    with pytest.raises(TeacherError, match="Schema"):
        parse_teacher_action(
            {"name": "search", "arguments": {"query": "revenue"}, "gold": "table_1"}
        )


def test_teacher_action_parser_rejects_invalid_arguments():
    with pytest.raises(TeacherError, match="缺少参数"):
        parse_teacher_action({"name": "search", "arguments": {}})
    with pytest.raises(TeacherError, match="缺少参数"):
        parse_teacher_action(
            {"name": "emit_program", "arguments": {"expression": "divide(1,2)"}}
        )


def test_scripted_teacher_runs_through_same_environment():
    sample = load_finqa("data/raw/finqa/train.json", limit=1)[0]
    result = collect_teacher_trajectory(sample, scripted_oracle_teacher(sample))

    assert result.accepted
    assert result.raw["diagnosis"]["valid"]
    assert result.raw["evidence_coverage"] == 1.0
    assert all(
        message["trainable"] == (message["role"] == "assistant")
        for message in result.sft_row["messages"]
    )


def test_rejected_teacher_rollout_gets_primary_error():
    sample = load_finqa("data/raw/finqa/train.json", limit=1)[0]
    teacher = ScriptedTeacher([AgentAction(ActionName.DELIVER)])
    result = collect_teacher_trajectory(sample, teacher)

    assert not result.accepted
    assert result.raw["diagnosis"]["primary_error"] == "infrastructure_error"
    assert "action_not_completed" in result.raw["rejection_reasons"]


def test_collection_resumes_without_duplicate_calls(tmp_path):
    samples = load_finqa("data/raw/finqa/train.json", limit=2)
    factory = scripted_oracle_teacher
    first = collect_teacher_dataset(samples, factory, output_dir=tmp_path, limit=1)
    second = collect_teacher_dataset(samples, factory, output_dir=tmp_path, limit=1)

    assert first["processed"] == 1
    assert first["attempted_this_run"] == 1
    assert second["processed"] == 2
    assert second["attempted_this_run"] == 1
    raw = [json.loads(line) for line in (tmp_path / "raw.jsonl").read_text().splitlines()]
    assert len(raw) == 2
    assert len({row["task_id"] for row in raw}) == 2


def test_teacher_can_load_local_secret_file_without_mutating_environment(tmp_path):
    path = tmp_path / ".env.teacher.local"
    path.write_text(
        "export OPENAI_BASE_URL=https://provider.example/v1\n"
        "OPENAI_API_KEY='secret-value'\n"
        "TEACHER_MODEL=teacher-model\n",
        encoding="utf-8",
    )

    teacher = OpenAICompatibleTeacher.from_config_file(path)

    assert teacher.base_url == "https://provider.example/v1"
    assert teacher.api_key == "secret-value"
    assert teacher.model_name == "teacher-model"


def test_teacher_prompt_teaches_exact_calculator_dsl_and_recovery():
    assert "divide(a,b)" in TEACHER_SYSTEM_PROMPT
    assert "没有 percent()" in TEACHER_SYSTEM_PROMPT
    assert "不能只反复 verify" in TEACHER_SYSTEM_PROMPT


def test_openai_teacher_repairs_invalid_action_before_environment():
    responses = iter(
        [
            {"choices": [{"message": {"content": '{"name":"search","arguments":{}}'}}]},
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"name":"search","arguments":{"query":"revenue"}}'
                            )
                        }
                    }
                ]
            },
        ]
    )
    request_bodies = []

    class _Response:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(self.value).encode()

    def fake_urlopen(request, **_kwargs):
        request_bodies.append(json.loads(request.data))
        return _Response(next(responses))

    teacher = OpenAICompatibleTeacher(
        base_url="https://provider.example/v1",
        api_key="secret",
        model_name="teacher",
        max_retries=1,
    )
    with patch("finsight.training.teacher.urllib.request.urlopen", fake_urlopen), patch(
        "finsight.training.teacher.time.sleep"
    ):
        action = teacher.next_action([{"role": "user", "content": "question"}])

    assert action.name == ActionName.SEARCH
    assert action.arguments == {"query": "revenue"}
    assert len(request_bodies) == 2
    assert teacher.schema_repairs == 1
    assert "未通过动作 Schema" in request_bodies[1]["messages"][-1]["content"]


def test_teacher_accepts_full_chat_endpoint_and_optional_provider_parameters():
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"name":"abstain","arguments":'
                                    '{"reason_code":"missing_evidence"}}'
                                )
                            }
                        }
                    ]
                }
            ).encode()

    def fake_urlopen(request, **_kwargs):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        return _Response()

    teacher = OpenAICompatibleTeacher(
        base_url="https://provider.example/v1/chat/completions",
        api_key="secret",
        model_name="provider-model",
        temperature=None,
        response_format=None,
        extra_body={"thinking": {"type": "disabled"}},
    )
    with patch("finsight.training.teacher.urllib.request.urlopen", fake_urlopen):
        action = teacher.next_action([{"role": "user", "content": "question"}])

    assert action.name == ActionName.ABSTAIN
    assert captured["url"] == "https://provider.example/v1/chat/completions"
    assert "temperature" not in captured["body"]
    assert "response_format" not in captured["body"]
    assert captured["body"]["thinking"] == {"type": "disabled"}
