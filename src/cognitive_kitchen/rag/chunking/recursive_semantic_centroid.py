"""Recursive pre-split, then join pieces while they match the running centroid.

Tolerates a single odd piece but closes the chunk on sustained topic drift, so
it yields fewer and longer chunks than the adjacent variant."""
from __future__ import annotations

from ..registry import register
from ..types import Corpus, Passage
from ._base import make_passages
from ._semantic import (default_embedder, groups_to_bounds, join_centroid,
                        recursive_units)


class RecursiveSemanticCentroidChunker:
    def __init__(self, threshold: float = 0.50, unit_size: int = 200,
                 max_chars: int = 1200, embedder=None) -> None:
        self.name = "recursive_semantic_centroid"
        self.params = {"threshold": threshold, "unit_size": unit_size,
                       "max_chars": max_chars}
        self._embedder = embedder

    def split(self, corpus: Corpus) -> list[Passage]:
        units = recursive_units(corpus.text, size=self.params["unit_size"])
        if not units:
            return []
        embedder = self._embedder or default_embedder()
        vectors = embedder.encode([corpus.text[a:b] for a, b in units])
        groups = join_centroid(vectors, self.params["threshold"])
        bounds = groups_to_bounds(units, groups, self.params["max_chars"])
        return make_passages(corpus, bounds, self.name, self.params)


@register("chunker", "recursive_semantic_centroid")
def make(threshold: float = 0.50, unit_size: int = 200,
         max_chars: int = 1200, embedder=None) -> RecursiveSemanticCentroidChunker:
    return RecursiveSemanticCentroidChunker(threshold, unit_size, max_chars, embedder)