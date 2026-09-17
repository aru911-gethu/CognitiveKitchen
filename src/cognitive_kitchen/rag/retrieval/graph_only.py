"""The graph as a retriever, so the graph route can be judged like any other.

Stage 5 stops at a set of recipes, which is the right shape for set precision
and recall but leaves the route unjudgeable end to end: no answer is generated,
so Faithfulness, AnswerRelevancy and Cookable have nothing to score. That made
the graph a demonstration rather than a candidate.

This adapter closes that. It presents traversal results as Passages, so
stage6_generation.evaluate works unchanged and the graph route lands in the same
table as recursive -> rrf -> stuff_strict, on the same metrics.

Not a competitor in Stage 3: it takes no chunks, ignores the chunker entirely,
and has no ranking of its own. Which is the honest problem here -- a traversal
returns a set, and something has to choose which k of them the generator sees.
That choice is documented in `order` rather than hidden, because it is a real
weakness of answering from a graph alone and not something to paper over.
"""
from __future__ import annotations

import json
from typing import Any

from ..registry import register
from ..types import Passage, Scored


def _render(card: dict[str, Any]) -> str:
    """A recipe card as the text a generator can work from."""
    parts = [card.get("title") or "(untitled)"]
    ingredients = card.get("ingredients") or []
    if ingredients:
        parts.append("Ingredients:")
        parts.extend(f"- {name}" for name in sorted(ingredients))
    text = card.get("text") or ""
    if text:
        parts.append("")
        parts.append(text)
    return "\n".join(parts)


class GraphOnlyRetriever:
    """Traversal in, Passages out. No embeddings, no chunks, no index."""

    def __init__(self, order: str = "fewest_ingredients", model: str = "gpt4o-mini",
                 extractor=None, source: str = "") -> None:
        self.name = f"graph_only({order})"
        self.params = {"order": order, "model": model, "source": source}
        self._extractor = extractor
        self._model = model
        self._source = source
        self.trace: list[dict[str, Any]] = []

    def extractor(self):
        if self._extractor is None:
            from ..graph.constraints import ConstraintExtractor

            self._extractor = ConstraintExtractor(model=self._model)
        return self._extractor

    def index(self, passages: list[Passage]) -> None:
        """Nothing to index. Kept so the evaluators can treat this like a peer.

        The corpus source is taken from the passages when they are offered,
        because the graph spans every ingestion run while a corpus is one.
        """
        sources = {p.meta.get("source") or "" for p in passages or []}
        if len(sources) == 1:
            self._source = sources.pop() or self._source
            self.params["source"] = self._source

    def _ordered(self, keys: list[str], cards: dict[str, dict[str, Any]],
                 constraints) -> list[str]:
        """Choose which of the matching recipes to show first.

        A traversal has no notion of better. `fewest_ingredients` prefers the
        simplest dish that satisfies the condition, which is a defensible answer
        to "what can I make" and is stable across runs. It is a heuristic, not a
        relevance score, and that is the point: the absence of ranking is what
        separates this route from retrieval.
        """
        wanted = set(constraints.include or ())

        def key_for(key: str) -> tuple:
            card = cards.get(key) or {}
            ingredients = set(card.get("ingredients") or ())
            overlap = len(ingredients & wanted)
            return (-overlap, len(ingredients), key)

        if self.params["order"] == "fewest_ingredients":
            return sorted(keys, key=key_for)
        return sorted(keys)

    def search(self, query: str, k: int) -> list[Scored]:
        from ..graph import traverse

        try:
            constraints = self.extractor().extract(query)
        except Exception:
            self.trace.append({"query": query, "error": "extraction failed"})
            return []

        keys = traverse.filter_recipes(
            include=constraints.include, exclude=constraints.exclude,
            exclude_categories=constraints.exclude_categories,
            course=constraints.course)
        if self._source:
            keys = [key for key in keys
                    if key.split(":", 1)[0].startswith(self._source)]

        if not keys:
            self.trace.append({"query": query, "matched": 0,
                               "constraints": constraints.as_dict()})
            return []

        cards = traverse.recipe_cards(keys)
        chosen = self._ordered(keys, cards, constraints)[:k]
        self.trace.append({"query": query, "matched": len(keys),
                           "shown": len(chosen),
                           "constraints": constraints.as_dict()})

        out: list[Scored] = []
        for rank, key in enumerate(chosen, start=1):
            card = cards.get(key) or {}
            recipe_id = card.get("recipe_id") or key.split(":", 1)[-1]
            passage = Passage(
                passage_id=f"graph-{key}",
                text=_render(card),
                start=0, end=len(_render(card)),
                # the same keys the metrics read off a chunk, so constraint and
                # diversity scoring work without a special case
                meta={"strategy": "graph_only", "recipe_ids": [recipe_id],
                      "source": key.split(":", 1)[0], "graph_key": key,
                      "matched_total": len(keys)})
            # no relevance to report, so rank stands in for score
            out.append(Scored(passage=passage, score=1.0 / rank, rank=rank))
        return out


@register("retriever", "graph_only")
def make(order: str = "fewest_ingredients", model: str = "gpt4o-mini",
         extractor=None, source: str = "") -> GraphOnlyRetriever:
    return GraphOnlyRetriever(order, model, extractor, source)