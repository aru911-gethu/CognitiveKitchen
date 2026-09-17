"""Maximal marginal relevance: trade relevance against novelty.

Picks the candidate maximising

    lambda * sim(query, d)  -  (1 - lambda) * max sim(d, already chosen)

so near-duplicate chunks of the same recipe stop crowding out alternatives.
This is the strategy the Diversity metric is meant to reward.

Re-selection works on any candidate list, so `base` is a parameter: MMR over
rrf is as sensible as MMR over dense, and hardwiring it to dense would have made
diversity available only to the weakest source.
"""
from __future__ import annotations

import numpy as np

from ..registry import build, discover, register
from ..types import Passage, Scored
from ._base import MultiQueryMixin


class MMRRetriever(MultiQueryMixin):
    def __init__(self, lambda_: float = 0.6, pool: int = 40, embedder=None,
                 base: str = "dense") -> None:
        self.name = f"mmr({base},l={lambda_})"
        self.params = {"lambda_": lambda_, "pool": pool, "base": base}
        discover("cognitive_kitchen.rag.retrieval")
        kwargs = {"embedder": embedder} if base in (
            "dense", "hybrid", "rrf", "mmr", "cross_encoder") else {}
        self.base = build("retriever", base, **kwargs)
        # Novelty is measured with vectors, so an embedder is needed even when
        # the base is sparse and has none of its own.
        self.embedder = embedder or getattr(self.base, "embedder", None)
        if self.embedder is None:
            discover("cognitive_kitchen.rag.embedding")
            self.embedder = build("embedder", "st")

    def index(self, passages: list[Passage]) -> None:
        self.base.index(passages)

    def search(self, query: str, k: int) -> list[Scored]:
        pool = max(self.params["pool"], k)
        lam = self.params["lambda_"]
        candidates = self.base.search(query, pool)
        if not candidates:
            return []
        vectors = self.embedder.encode([c.passage.text for c in candidates])
        relevance = np.array([c.score for c in candidates], dtype=np.float32)

        chosen: list[int] = []
        remaining = set(range(len(candidates)))
        while len(chosen) < min(k, len(candidates)):
            best_idx, best_value = None, -1e9
            for i in remaining:
                novelty = 0.0 if not chosen else float(
                    max(vectors[i] @ vectors[j] for j in chosen))
                value = lam * float(relevance[i]) - (1 - lam) * novelty
                if value > best_value:
                    best_idx, best_value = i, value
            chosen.append(best_idx)
            remaining.discard(best_idx)

        return [Scored(passage=candidates[i].passage, score=candidates[i].score,
                       rank=rank)
                for rank, i in enumerate(chosen, start=1)]


@register("retriever", "mmr")
def make(lambda_: float = 0.6, pool: int = 40, embedder=None,
         base: str = "dense") -> MMRRetriever:
    return MMRRetriever(lambda_, pool, embedder, base)