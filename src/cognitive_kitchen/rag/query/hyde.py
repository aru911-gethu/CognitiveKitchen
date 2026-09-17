"""HyDE: search with a hypothetical answer instead of the question.

A question is short and interrogative; a recipe card is long and declarative.
They sit in different parts of the vector space. Writing a fake recipe card and
embedding that puts the query into the same shape as the corpus.

Only the vector is used, never the text, so an inaccurate hypothetical still
helps provided it is recipe-shaped. The generated document is blended with the
original question so a hallucination cannot drag the query away entirely.
"""
from __future__ import annotations

from ..registry import build, discover, register

TEMPLATE = ("Write a short Indian recipe card answering the request below. "
            "Give a title, an ingredient list, and two method steps. "
            "No commentary.\n\nRequest: {q}\n\nRecipe:\n")


class Hyde:
    name = "hyde"

    def __init__(self, max_tokens: int = 120) -> None:
        self.max_tokens = max_tokens
        self._generator = None

    def expand(self, question: str) -> list[str]:
        if self._generator is None:
            discover("cognitive_kitchen.rag.generate")
            self._generator = build("generator", "qwen")
        doc = self._generator.complete(TEMPLATE.format(q=question),
                                       max_tokens=self.max_tokens).strip()
        return [question, doc] if doc else [question]


@register("query", "hyde")
def make(max_tokens: int = 120) -> Hyde:
    return Hyde(max_tokens)