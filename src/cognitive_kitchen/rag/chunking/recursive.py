"""LangChain RecursiveCharacterTextSplitter, mapped back to corpus offsets.

The splitter returns strings; we re-locate each one so passages keep the
character offsets the metrics depend on.
"""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..registry import register
from ..types import Corpus, Passage
from ._base import make_passages

SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


def locate(text: str, pieces: list[str]) -> list[tuple[int, int]]:
    """Find each piece in order, so offsets stay truthful."""
    bounds: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        if not piece:
            continue
        found = text.find(piece, cursor)
        if found == -1:                      # splitter normalised whitespace
            found = text.find(piece.strip(), cursor)
            if found == -1:
                continue
            piece = piece.strip()
        bounds.append((found, found + len(piece)))
        cursor = found + max(1, len(piece) // 2)
    return bounds


class RecursiveChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 120) -> None:
        self.name = "recursive"
        self.params = {"chunk_size": chunk_size, "overlap": overlap}
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap,
            separators=SEPARATORS, keep_separator=True)

    def split(self, corpus: Corpus) -> list[Passage]:
        pieces = self._splitter.split_text(corpus.text)
        return make_passages(corpus, locate(corpus.text, pieces),
                             self.name, self.params)


@register("chunker", "recursive")
def make(chunk_size: int = 800, overlap: int = 120) -> RecursiveChunker:
    return RecursiveChunker(chunk_size, overlap)