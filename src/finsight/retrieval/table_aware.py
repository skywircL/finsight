from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from finsight.models import Evidence, RetrievedEvidence
from finsight.retrieval.bm25 import BM25Retriever, tokenize


_NUMBER_TOKEN = re.compile(r"[-+]?\d+(?:\.\d+)?%?")
_QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "during",
    "for",
    "from",
    "how",
    "in",
    "is",
    "much",
    "of",
    "on",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
}


@dataclass(frozen=True)
class TableAwareConfig:
    """Weights selected on the first 2,000 FinQA train tasks, never on test."""

    row_label_boost: float = 0.75
    numeric_value_boost: float = 0.20


class TableAwareBM25Retriever:
    """BM25 plus transparent boosts for row labels and numeric table values.

    The reranker uses only fields visible in the document. It has no access to
    Gold evidence IDs, answers or programs.
    """

    def __init__(
        self,
        evidences: Sequence[Evidence],
        *,
        config: TableAwareConfig = TableAwareConfig(),
    ) -> None:
        self.evidences = tuple(evidences)
        self.config = config
        self._base = BM25Retriever(self.evidences)

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievedEvidence]:
        base_ranked = self._base.search(query, top_k=len(self.evidences))
        query_tokens = [
            token for token in tokenize(query) if token not in _QUESTION_STOPWORDS
        ]
        numeric_query_tokens = {
            token for token in query_tokens if _NUMBER_TOKEN.fullmatch(token)
        }
        rescored: list[tuple[float, int, Evidence]] = []
        for base_order, item in enumerate(base_ranked):
            evidence = item.evidence
            score = item.score
            if evidence.kind == "table":
                row_label = str(evidence.metadata.get("row_label", ""))
                values = evidence.metadata.get("values", [])
                row_tokens = set(tokenize(row_label))
                value_tokens = set(tokenize(" ".join(map(str, values))))
                row_matches = sum(token in row_tokens for token in query_tokens)
                numeric_matches = sum(
                    token in value_tokens for token in numeric_query_tokens
                )
                score += self.config.row_label_boost * row_matches
                score += self.config.numeric_value_boost * numeric_matches
            rescored.append((score, base_order, evidence))

        rescored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievedEvidence(evidence=evidence, score=score, rank=rank)
            for rank, (score, _, evidence) in enumerate(
                rescored[: max(top_k, 0)], start=1
            )
        ]
