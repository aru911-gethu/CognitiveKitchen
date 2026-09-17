"""Cross-encoder reranking of a first-stage shortlist.

A cross-encoder scores the query and passage jointly rather than comparing two
independent vectors, which is more accurate and far slower. It therefore reranks
a pool rather than searching the whole index.
"""
from __future__ import annotations

from ...config import settings
from ..registry import build, discover, register
from ..types import Passage, Scored
from ._base import MultiQueryMixin


class CrossEncoderRetriever(MultiQueryMixin):
    def __init__(self, model_name: str | None = None, pool: int = 20,
                 first_stage: str = "hybrid", embedder=None) -> None:
        self.model_name = model_name or settings.reranker_model
        self.name = f"cross_encoder({self.model_name.split('/')[-1]})"
        self.params = {"pool": pool, "first_stage": first_stage}
        discover("cognitive_kitchen.rag.retrieval")
        kwargs = {"embedder": embedder} if first_stage in ("dense", "hybrid", "rrf", "mmr") else {}
        self.base = build("retriever", first_stage, **kwargs)
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def index(self, passages: list[Passage]) -> None:
        self.base.index(passages)

    def search(self, query: str, k: int) -> list[Scored]:
        candidates = self.base.search(query, max(self.params["pool"], k))
        if not candidates:
            return []
        model = self._ensure()
        scores = model.predict([(query, c.passage.text) for c in candidates],
                               show_progress_bar=False)
        order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
        return [Scored(passage=candidates[i].passage, score=float(scores[i]), rank=rank)
                for rank, i in enumerate(order[:k], start=1)]


@register("retriever", "cross_encoder")
def make(model_name: str | None = None, pool: int = 20,
         first_stage: str = "hybrid", embedder=None) -> CrossEncoderRetriever:
    return CrossEncoderRetriever(model_name, pool, first_stage, embedder)