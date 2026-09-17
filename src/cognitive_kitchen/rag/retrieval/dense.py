"""Dense retrieval with a FAISS inner-product index.

Vectors are L2-normalised by the embedder, so inner product is cosine
similarity and IndexFlatIP gives exact nearest neighbours.
"""
from __future__ import annotations

import faiss
import numpy as np

from ..registry import build, discover, register
from ..types import Passage, Scored
from ._base import MultiQueryMixin, to_scored


class DenseRetriever(MultiQueryMixin):
    def __init__(self, embedder=None) -> None:
        self.name = "dense"
        if embedder is None:
            discover("cognitive_kitchen.rag.embedding")
            embedder = build("embedder", "qwen")
        self.embedder = embedder
        self.passages: list[Passage] = []
        self._index: faiss.Index | None = None
        self._vectors: np.ndarray | None = None

    def index(self, passages: list[Passage]) -> None:
        self.passages = passages
        vectors = self.embedder.encode([p.text for p in passages])
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index = faiss.IndexFlatIP(self._vectors.shape[1])
        self._index.add(self._vectors)

    def search(self, query: str, k: int) -> list[Scored]:
        if self._index is None:
            raise RuntimeError("index() must be called before search()")
        vector = self.embedder.encode([query]).astype(np.float32)
        scores, ids = self._index.search(vector, min(k, len(self.passages)))
        return to_scored(self.passages, ids[0].tolist(), scores[0].tolist(), k)


@register("retriever", "dense")
def make(embedder=None) -> DenseRetriever:
    return DenseRetriever(embedder)