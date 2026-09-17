"""Shared helpers for chunkers."""
from __future__ import annotations

from ..types import Corpus, Passage


def make_passages(corpus: Corpus, bounds: list[tuple[int, int]],
                  strategy: str, params: dict) -> list[Passage]:
    """Turn character bounds into Passages, tagging the recipes each touches.

    `source` travels beside `recipe_ids` because recipe ids restart at r_0001 in
    every ingestion run, so an id alone cannot identify a recipe once more than
    one run exists.
    """
    passages: list[Passage] = []
    for i, (start, end) in enumerate(bounds):
        if end <= start:
            continue
        touched = [s.recipe_id for s in corpus.spans
                   if s.start < end and start < s.end]
        passages.append(Passage(
            passage_id=f"{strategy}-{i:05d}",
            text=corpus.text[start:end],
            start=start,
            end=end,
            meta={"strategy": strategy, "params": params,
                  "recipe_ids": touched, "source": corpus.source},
        ))
    return passages


def fixed_bounds(total: int, size: int, overlap: int) -> list[tuple[int, int]]:
    if size <= 0:
        raise ValueError("size must be positive")
    step = max(1, size - max(0, overlap))
    out: list[tuple[int, int]] = []
    start = 0
    while start < total:
        out.append((start, min(start + size, total)))
        if start + size >= total:
            break
        start += step
    return out