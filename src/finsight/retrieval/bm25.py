from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from finsight.models import Evidence, RetrievedEvidence


_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[-+]?\d+(?:\.\d+)?%?", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


class BM25Retriever:
    """Small dependency-free BM25 implementation for per-report evidence retrieval."""

    def __init__(self, evidences: Sequence[Evidence], *, k1: float = 1.5, b: float = 0.75):
        if not evidences:
            raise ValueError("At least one evidence item is required")
        self.evidences = tuple(evidences)
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(item.text) for item in self.evidences]
        self._frequencies = [Counter(tokens) for tokens in self._tokens]
        self._avgdl = sum(map(len, self._tokens)) / len(self._tokens)
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        count = len(self.evidences)
        self._idf = {
            token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievedEvidence]:
        query_tokens = tokenize(query)
        scores: list[tuple[int, float]] = []
        for index, frequencies in enumerate(self._frequencies):
            document_length = len(self._tokens[index])
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * document_length / max(self._avgdl, 1)
                )
                score += self._idf.get(token, 0.0) * frequency * (self.k1 + 1) / denominator
            scores.append((index, score))

        ranked = sorted(scores, key=lambda item: (-item[1], item[0]))[: max(top_k, 0)]
        return [
            RetrievedEvidence(evidence=self.evidences[index], score=score, rank=rank)
            for rank, (index, score) in enumerate(ranked, start=1)
        ]

