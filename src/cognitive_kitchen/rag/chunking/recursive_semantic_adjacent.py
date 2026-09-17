"""Recursive pre-split, then join neighbours while they stay similar.

Recursive supplies boundaries that never fall mid-word; the adjacent rule then
decides which of those pieces belong to the same recipe."""
from __future__ import annotations

from ..registry import register
from ..types import Corpus, Passage
from ._base import make_passages
from ._semantic import (default_embedder, groups_to_bounds, join_adjacent,
                        recursive_units)


class RecursiveSemanticAdjacentChunker:
    def __init__(self, threshold: float = 0.55, unit_size: int = 200,
                 max_chars: int = 1200, embedder=None) -> None:
        self.name = "recursive_semantic_adjacent"
        self.params = {"threshold": threshold, "unit_size": unit_size,
                       "max_chars": max_chars}
        self._embedder = embedder

    def split(self, corpus: Corpus) -> list[Passage]:
        units = recursive_units(corpus.text, size=self.params["unit_size"])
        if not units:
            return []
        embedder = self._embedder or default_embedder()
        vectors = embedder.encode([corpus.text[a:b] for a, b in units])
        groups = join_adjacent(vectors, self.params["threshold"])
        bounds = groups_to_bounds(units, groups, self.params["max_chars"])
        return make_passages(corpus, bounds, self.name, self.params)


@register("chunker", "recursive_semantic_adjacent")
def make(threshold: float = 0.55, unit_size: int = 200,
         max_chars: int = 1200, embedder=None) -> RecursiveSemanticAdjacentChunker:
    return RecursiveSemanticAdjacentChunker(threshold, unit_size, max_chars, embedder)