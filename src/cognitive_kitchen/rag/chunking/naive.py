"""Fixed-width chunks with optional overlap.

The honest baseline: it respects nothing about the text, so any smarter chunker
has to beat it to justify itself.
"""
from __future__ import annotations

from ..registry import register
from ..types import Corpus, Passage
from ._base import fixed_bounds, make_passages


class NaiveChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 0) -> None:
        self.name = "naive"
        self.params = {"chunk_size": chunk_size, "overlap": overlap}

    def split(self, corpus: Corpus) -> list[Passage]:
        bounds = fixed_bounds(len(corpus.text), self.params["chunk_size"],
                              self.params["overlap"])
        return make_passages(corpus, bounds, self.name, self.params)


@register("chunker", "naive")
def make(chunk_size: int = 800, overlap: int = 0) -> NaiveChunker:
    return NaiveChunker(chunk_size, overlap)