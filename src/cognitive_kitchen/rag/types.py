"""Data contracts passed between pipeline stages.

`Passage.start`/`end` are character offsets into `Corpus.text`. Because
`Corpus.spans` records the same offsets per recipe, chunk quality is arithmetic
rather than a lookup, and needs nothing external to measure.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RecipeSpan:
    """Where one pipeline recipe sits in the rendered corpus."""

    recipe_id: str
    title: str
    start: int          # inclusive
    end: int            # exclusive
    pages: list[int] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Corpus:
    text: str
    spans: tuple[RecipeSpan, ...]
    source: str = ""

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def key(self) -> str:
        """Short hash used to name caches."""
        return self.sha256[:12]

    def span_for(self, recipe_id: str) -> RecipeSpan | None:
        return next((s for s in self.spans if s.recipe_id == recipe_id), None)


@dataclass(frozen=True)
class Passage:
    """A chunk of the corpus."""

    passage_id: str
    text: str
    start: int
    end: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Scored:
    passage: Passage
    score: float
    rank: int


@dataclass
class Answer:
    question: str
    text: str
    contexts: list[Passage] = field(default_factory=list)
    refused: bool = False
    model: str = ""
    elapsed_s: float = 0.0


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Characters shared by two half-open intervals."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of intervals, so coverage cannot exceed 1.0 by double counting."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]