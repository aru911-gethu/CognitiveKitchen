"""BM25 lexical retrieval. Strong when the query names the dish."""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from ..registry import register
from ..types import Passage, Scored
from ._base import MultiQueryMixin, to_scored


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever(MultiQueryMixin):
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.name = "bm25"
        self.params = {"k1": k1, "b": b}
        self.passages: list[Passage] = []
        self._bm25: BM25Okapi | None = None

    def index(self, passages: list[Passage]) -> None:
        self.passages = passages
        self._bm25 = BM25Okapi([tokens(p.text) for p in passages],
                               k1=self.params["k1"], b=self.params["b"])

    def search(self, query: str, k: int) -> list[Scored]:
        if self._bm25 is None:
            raise RuntimeError("index() must be called before search()")
        scores = np.asarray(self._bm25.get_scores(tokens(query)), dtype=np.float32)
        order = np.argsort(-scores).tolist()
        return to_scored(self.passages, order, scores[order].tolist(), k)


@register("retriever", "bm25")
def make(k1: float = 1.5, b: float = 0.75) -> BM25Retriever:
    return BM25Retriever(k1, b)