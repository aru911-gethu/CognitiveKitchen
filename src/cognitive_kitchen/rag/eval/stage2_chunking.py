"""Stage 2 - chunking quality, four metrics.

    word_purity       shared / words in chunk      is the chunk about one recipe
    word_recall_best  shared / words in recipe     can one chunk answer it
    k_at_90           greedy chunks to reach 90%    how many chunks are needed
    self_sufficiency  1 / rank of its own recipe    can the chunk be found

The first three compare against the golden recipe as reference content; no id
mapping is used, only word overlap. The fourth needs only the pipeline itself.
"""
from __future__ import annotations

import statistics
from typing import Any

import numpy as np

from ..types import Corpus, Passage
from .golden import GoldenRecipe, load_golden
from .tokens import tokenise

COVERAGE_TARGET = 0.90
MAX_K = 25


def _k_for_target(recipe_words: set[str], chunk_words: list[set[str]],
                  target: float) -> tuple[int, float]:
    """Fewest chunks whose union covers `target` of the recipe's words.

    Greedy: repeatedly take the chunk adding the most words not yet covered.
    """
    if not recipe_words:
        return 0, 0.0
    remaining = set(recipe_words)
    used = 0
    while remaining and used < MAX_K:
        best, gain = None, 0
        for words in chunk_words:
            overlap = len(remaining & words)
            if overlap > gain:
                best, gain = words, overlap
        if not gain:
            break
        remaining -= best
        used += 1
        if 1 - len(remaining) / len(recipe_words) >= target:
            break
    return used, 1 - len(remaining) / len(recipe_words)


def _self_sufficiency(corpus: Corpus, passages: list[Passage], embedder,
                      sample: int | None) -> dict[str, Any]:
    """Use each chunk as a query against the recipe cards; score 1/rank."""
    from .. import types

    if not passages or not corpus.spans:
        return {"mean": None, "n": 0}
    cards = [corpus.text[s.start:s.end] for s in corpus.spans]
    ids = np.array([s.recipe_id for s in corpus.spans])
    card_vectors = embedder.encode(cards)

    chosen = passages
    if sample is not None and len(passages) > sample:
        step = len(passages) / sample
        chosen = [passages[int(i * step)] for i in range(sample)]

    vectors = embedder.encode([c.text for c in chosen])
    scores: list[float] = []
    for chunk, vector in zip(chosen, vectors):
        owned: dict[str, int] = {}
        for span in corpus.spans:
            n = types.overlap(chunk.start, chunk.end, span.start, span.end)
            if n:
                owned[span.recipe_id] = owned.get(span.recipe_id, 0) + n
        if not owned:
            continue
        dominant = max(owned, key=owned.get)
        order = np.argsort(-(card_vectors @ vector))
        rank = int(np.where(ids[order] == dominant)[0][0]) + 1
        scores.append(1.0 / rank)
    return {"mean": round(statistics.fmean(scores), 4) if scores else None,
            "n": len(scores)}


def evaluate(corpus: Corpus, passages: list[Passage], embedder=None,
             golden: tuple[GoldenRecipe, ...] | None = None,
             self_suff_sample: int | None = 120) -> dict[str, Any]:
    lengths = [p.length for p in passages] or [0]
    result: dict[str, Any] = {
        "n_chunks": len(passages),
        "mean_chunk_chars": round(statistics.fmean(lengths), 1),
        "word_purity": None,
        "word_recall_best": None,
        "k_at_90": None,
        "k_at_90_worst": None,
        "self_sufficiency": None,
        "recipes_scored": 0,
    }
    if not passages:
        return result

    reference = golden if golden is not None else load_golden()
    chunk_words = [tokenise(p.text) for p in passages]

    purities: list[float] = []
    recalls: list[float] = []
    ks: list[int] = []
    for recipe in reference:
        words = set(recipe.words)
        best_words, best_overlap = None, 0
        for candidate in chunk_words:
            overlap = len(words & candidate)
            if overlap > best_overlap:
                best_words, best_overlap = candidate, overlap
        if best_words is None:
            continue
        purities.append(best_overlap / len(best_words))
        recalls.append(best_overlap / len(words))
        touching = [c for c in chunk_words if words & c]
        k, _ = _k_for_target(words, touching, COVERAGE_TARGET)
        ks.append(k)

    if purities:
        result.update({
            "word_purity": round(statistics.fmean(purities), 4),
            "word_recall_best": round(statistics.fmean(recalls), 4),
            "k_at_90": round(statistics.fmean(ks), 2),
            "k_at_90_worst": max(ks),
            "recipes_scored": len(purities),
        })
    if embedder is not None:
        result["self_sufficiency"] = _self_sufficiency(
            corpus, passages, embedder, self_suff_sample)
    return result