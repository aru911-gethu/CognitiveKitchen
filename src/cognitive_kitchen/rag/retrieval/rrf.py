"""Reciprocal rank fusion of dense and sparse.

Uses only rank position, never the score, so the two retrievers need no
calibration against each other. That is the whole appeal over weighted hybrid.

    RRF(d) = sum over retrievers of  1 / (c + rank(d))
"""
from __future__ import annotations

from ..registry import build, discover, register
from ..types import Passage, Scored
from ._base import MultiQueryMixin


class RRFRetriever(MultiQueryMixin):
    def __init__(self, c: int = 60, sparse: str = "bm25", embedder=None,
                 pool: int = 50) -> None:
        self.name = f"rrf(c={c})"
        self.params = {"c": c, "sparse": sparse, "pool": pool}
        discover("cognitive_kitchen.rag.retrieval")
        self.dense = build("retriever", "dense", embedder=embedder)
        self.sparse = build("retriever", sparse)
        self._by_id: dict[str, Passage] = {}

    def index(self, passages: list[Passage]) -> None:
        self.dense.index(passages)
        self.sparse.index(passages)
        self._by_id = {p.passage_id: p for p in passages}

    def search(self, query: str, k: int) -> list[Scored]:
        pool = max(self.params["pool"], k)
        c = self.params["c"]
        fused: dict[str, float] = {}
        for retriever in (self.dense, self.sparse):
            for hit in retriever.search(query, pool):
                fused[hit.passage.passage_id] = (
                    fused.get(hit.passage.passage_id, 0.0) + 1.0 / (c + hit.rank))
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [Scored(passage=self._by_id[pid], score=score, rank=i)
                for i, (pid, score) in enumerate(ranked, start=1)]


@register("retriever", "rrf")
def make(c: int = 60, sparse: str = "bm25", embedder=None,
         pool: int = 50) -> RRFRetriever:
    return RRFRetriever(c, sparse, embedder, pool)