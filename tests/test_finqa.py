from finsight.data.finqa import normalize_finqa_record, serialize_table_row


def test_table_serialization_preserves_headers():
    text = serialize_table_row(["metric", "2023", "2022"], ["revenue", "$ 10", "(8)"])
    assert "revenue" in text
    assert "2023 is $ 10" in text
    assert "2022 is (8)" in text


def test_normalize_record_builds_gold_evidence():
    record = {
        "id": "report-1",
        "pre_text": ["Revenue increased."],
        "post_text": [],
        "table": [["metric", "2023"], ["revenue", "10"]],
        "qa": {
            "question": "What was revenue?",
            "program": "add(10, 0)",
            "gold_inds": {"table_1": "gold"},
            "exe_ans": 10,
        },
    }
    sample = normalize_finqa_record(record)
    assert sample.gold_evidence_ids == ("table_1",)
    assert sample.table[1] == ("revenue", "10")
    assert {item.evidence_id for item in sample.evidences} == {
        "text_0",
        "table_0",
        "table_1",
    }
    table = next(item for item in sample.evidences if item.evidence_id == "table_1")
    assert table.metadata["row_label"] == "revenue"
    assert table.metadata["values"] == ["10"]
    assert table.metadata["header"] == ["metric", "2023"]


def test_headerless_table_preserves_table_zero_gold_evidence():
    record = {
        "id": "report-headerless",
        "pre_text": [],
        "post_text": [],
        "table": [["2007", "$ 10"], ["total", "$ 20"]],
        "qa": {
            "question": "What share was collected in 2007?",
            "program": "divide(10, 20)",
            "gold_inds": {"table_0": "gold", "table_1": "gold"},
            "exe_ans": 0.5,
        },
    }
    sample = normalize_finqa_record(record)

    assert sample.gold_evidence_ids == ("table_0", "table_1")
    assert {item.evidence_id for item in sample.evidences} == {"table_0", "table_1"}
