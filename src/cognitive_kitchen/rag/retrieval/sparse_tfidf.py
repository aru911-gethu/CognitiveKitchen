"""TF-IDF cosine retrieval. Included as the simpler sparse baseline."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..registry import register
from ..types import Passage, Scored
from ._base import MultiQueryMixin, to_scored


class TfidfRetriever(MultiQueryMixin):
    def __init__(self, ngram_max: int = 2, min_df: int = 1) -> None:
        self.name = "tfidf"
        self.params = {"ngram_max": ngram_max, "min_df": min_df}
        self.passages: list[Passage] = []
        self._vec: TfidfVectorizer | None = None
        self._matrix = None

    def index(self, passages: list[Passage]) -> None:
        self.passages = passages
        self._vec = TfidfVectorizer(lowercase=True, sublinear_tf=True,
                                    ngram_range=(1, self.params["ngram_max"]),
                                    min_df=self.params["min_df"])
        self._matrix = self._vec.fit_transform([p.text for p in passages])

    def search(self, query: str, k: int) -> list[Scored]:
        if self._vec is None:
            raise RuntimeError("index() must be called before search()")
        q = self._vec.transform([query])
        scores = np.asarray((self._matrix @ q.T).todense()).ravel()
        order = np.argsort(-scores).tolist()
        return to_scored(self.passages, order, scores[order].tolist(), k)


@register("retriever", "tfidf")
def make(ngram_max: int = 2, min_df: int = 1) -> TfidfRetriever:
    return TfidfRetriever(ngram_max, min_df)