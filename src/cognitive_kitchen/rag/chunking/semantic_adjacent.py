"""Split where a sentence stops resembling the sentence before it."""
from __future__ import annotations

from ..registry import register
from ..types import Corpus, Passage
from ._base import make_passages
from ._semantic import (default_embedder, groups_to_bounds, join_adjacent,
                        sentence_units)


class SemanticAdjacentChunker:
    def __init__(self, threshold: float = 0.55, unit_size: int = 40,
                 max_chars: int = 1200, embedder=None) -> None:
        self.name = "semantic_adjacent"
        self.params = {"threshold": threshold, "unit_size": unit_size,
                       "max_chars": max_chars}
        self._embedder = embedder

    def split(self, corpus: Corpus) -> list[Passage]:
        units = sentence_units(corpus.text, min_len=self.params["unit_size"])
        if not units:
            return []
        embedder = self._embedder or default_embedder()
        vectors = embedder.encode([corpus.text[a:b] for a, b in units])
        groups = join_adjacent(vectors, self.params["threshold"])
        bounds = groups_to_bounds(units, groups, self.params["max_chars"])
        return make_passages(corpus, bounds, self.name, self.params)


@register("chunker", "semantic_adjacent")
def make(threshold: float = 0.55, unit_size: int = 40,
         max_chars: int = 1200, embedder=None) -> SemanticAdjacentChunker:
    return SemanticAdjacentChunker(threshold, unit_size, max_chars, embedder)