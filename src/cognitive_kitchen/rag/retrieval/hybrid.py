"""Hybrid dense + sparse, combined on normalised scores.

Dense and BM25 scores live on different scales, so each result list is
min-max normalised before the weighted sum. Without that the raw BM25
magnitudes would swamp cosine similarities.
"""
from __future__ import annotations

from ..registry import build, discover, register
from ..types import Passage, Scored
from ._base import MultiQueryMixin


def _normalise(hits: list[Scored]) -> dict[str, float]:
    if not hits:
        return {}
    scores = [h.score for h in hits]
    low, high = min(scores), max(scores)
    spread = high - low
    return {h.passage.passage_id: (1.0 if spread == 0 else (h.score - low) / spread)
            for h in hits}


class HybridRetriever(MultiQueryMixin):
    def __init__(self, alpha: float = 0.5, sparse: str = "bm25",
                 embedder=None, pool: int = 50) -> None:
        self.name = f"hybrid({sparse},a={alpha})"
        self.params = {"alpha": alpha, "sparse": sparse, "pool": pool}
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
        alpha = self.params["alpha"]
        d = _normalise(self.dense.search(query, pool))
        s = _normalise(self.sparse.search(query, pool))
        merged = {pid: alpha * d.get(pid, 0.0) + (1 - alpha) * s.get(pid, 0.0)
                  for pid in set(d) | set(s)}
        ranked = sorted(merged.items(), key=lambda kv: -kv[1])[:k]
        return [Scored(passage=self._by_id[pid], score=score, rank=i)
                for i, (pid, score) in enumerate(ranked, start=1)]


@register("retriever", "hybrid")
def make(alpha: float = 0.5, sparse: str = "bm25", embedder=None,
         pool: int = 50) -> HybridRetriever:
    return HybridRetriever(alpha, sparse, embedder, pool)