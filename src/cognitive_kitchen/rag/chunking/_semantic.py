"""Shared machinery for the semantic chunkers.

All four work the same way: cut the text into small units, embed them, then
decide where to join. Only the join rule differs.
"""
from __future__ import annotations

import re

import numpy as np

from ..types import Corpus


def sentence_units(text: str, min_len: int = 40) -> list[tuple[int, int]]:
    """Sentence-ish units with true offsets, merging fragments below min_len."""
    bounds: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?:])\s+|\n+", text):
        end = match.start()
        if end > start:
            bounds.append((start, end))
        start = match.end()
    if start < len(text):
        bounds.append((start, len(text)))

    merged: list[tuple[int, int]] = []
    for span in bounds:
        if merged and (span[1] - merged[-1][0]) and (span[1] - span[0]) < min_len:
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    return [(a, b) for a, b in merged if b > a]


def recursive_units(text: str, size: int = 200, overlap: int = 0) -> list[tuple[int, int]]:
    """Pre-split with the recursive splitter, keeping offsets.

    Used by the composite chunkers: recursive gives boundaries that never fall
    mid-word, and the semantic pass then decides which pieces belong together.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from .recursive import SEPARATORS, locate

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=SEPARATORS,
        keep_separator=True)
    return locate(text, splitter.split_text(text))


def join_adjacent(vectors: np.ndarray, threshold: float) -> list[list[int]]:
    """Cut where a unit stops resembling the one before it.

    One odd unit is enough to trigger a break, which suits hard boundaries such
    as the start of a new recipe.
    """
    groups: list[list[int]] = [[0]] if len(vectors) else []
    for i in range(1, len(vectors)):
        if float(vectors[i] @ vectors[i - 1]) < threshold:
            groups.append([i])
        else:
            groups[-1].append(i)
    return groups


def join_centroid(vectors: np.ndarray, threshold: float) -> list[list[int]]:
    """Cut where a unit stops resembling the chunk accumulated so far.

    Compares against a running centroid rather than the previous unit, so a
    single outlier is tolerated but sustained drift ends the chunk.
    """
    groups: list[list[int]] = []
    centroid: np.ndarray | None = None
    count = 0
    for i, vector in enumerate(vectors):
        if centroid is None:
            groups.append([i])
            centroid = vector.astype(np.float64).copy()
            count = 1
            continue
        reference = centroid / max(np.linalg.norm(centroid), 1e-12)
        if float(vector @ reference) < threshold:
            groups.append([i])
            centroid = vector.astype(np.float64).copy()
            count = 1
        else:
            groups[-1].append(i)
            centroid += vector
            count += 1
    return groups


def groups_to_bounds(units: list[tuple[int, int]], groups: list[list[int]],
                     max_chars: int) -> list[tuple[int, int]]:
    """Collapse unit groups into chunk bounds, splitting any that grow too big."""
    bounds: list[tuple[int, int]] = []
    for group in groups:
        start = units[group[0]][0]
        end = units[group[-1]][1]
        while end - start > max_chars:
            cut = start + max_chars
            nearest = min((units[i][1] for i in group if start < units[i][1] <= cut),
                          key=lambda v: abs(v - cut), default=cut)
            bounds.append((start, nearest))
            start = nearest
        if end > start:
            bounds.append((start, end))
    return bounds


def default_embedder():
    from ..registry import build, discover

    discover("cognitive_kitchen.rag.embedding")
    return build("embedder", "st")