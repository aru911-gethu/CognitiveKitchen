"""Shared helpers for retrievers."""
from __future__ import annotations

from ..types import Passage, Scored


def to_scored(passages: list[Passage], order: list[int],
              scores: list[float], k: int) -> list[Scored]:
    out: list[Scored] = []
    for rank, (idx, score) in enumerate(zip(order[:k], scores[:k]), start=1):
        out.append(Scored(passage=passages[idx], score=float(score), rank=rank))
    return out


class MultiQueryMixin:
    """Run several rewritten queries and keep each passage's best score.

    Query transforms such as decomposition return more than one query; this is
    how their results are combined without a separate fusion step.
    """

    def search_multi(self, queries: list[str], k: int) -> list[Scored]:
        if len(queries) == 1:
            return self.search(queries[0], k)
        best: dict[str, Scored] = {}
        for query in queries:
            for hit in self.search(query, k):
                current = best.get(hit.passage.passage_id)
                if current is None or hit.score > current.score:
                    best[hit.passage.passage_id] = hit
        ranked = sorted(best.values(), key=lambda s: -s.score)[:k]
        return [Scored(passage=s.passage, score=s.score, rank=i)
                for i, s in enumerate(ranked, start=1)]