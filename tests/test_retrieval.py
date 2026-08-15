from finsight.models import Evidence
from finsight.retrieval import BM25Retriever, TableAwareBM25Retriever


def test_bm25_ranks_relevant_table_first():
    evidence = [
        Evidence("revenue", "metric revenue 2023 1250 2022 1100", "table"),
        Evidence("employees", "the company employed 900 people", "text"),
    ]
    result = BM25Retriever(evidence).search("2023 revenue compared with 2022", top_k=2)
    assert result[0].evidence.evidence_id == "revenue"
    assert result[0].score > result[1].score


def test_table_aware_reranker_boosts_matching_row_label():
    evidence = [
        Evidence(
            "table_revenue",
            "metric the revenue of 2023 is 10",
            "table",
            metadata={"row_label": "revenue", "values": ["10"]},
        ),
        Evidence(
            "table_cost",
            "metric the cost of 2023 is revenue adjusted 9",
            "table",
            metadata={"row_label": "cost", "values": ["9"]},
        ),
    ]

    result = TableAwareBM25Retriever(evidence).search("revenue", top_k=2)

    assert result[0].evidence.evidence_id == "table_revenue"
