"""Chunkers that use the boundaries ingestion already found.

The other six here are generic text-splitters: they infer where one thing ends
and the next begins, from characters or from embedding similarity. That is the
right tool when structure has to be recovered from prose.

This corpus does not need it. Ingestion already produced, per recipe, a title, a
list of ingredient lines and a list of steps, and `Corpus.spans` records exactly
where each recipe sits. Re-splitting the rendered text by character count throws
that away and then spends effort approximating it.

The cost of throwing it away is visible in answers. Asked "something with
butter", a character-split index returned five chunks that were pure ingredient
lines -- "1 cup sour cream", "4 oz. butter" -- severed from the titles and
methods they belonged to. Every chunk was topically perfect and none of them
named a dish, so the answer was a shopping list.

    recipe            one chunk per recipe. Self-contained by construction.
    recipe_sections   split long recipes at their section headings, but repeat
                      the title into every piece, so a fragment still knows
                      what dish it belongs to.
"""
from __future__ import annotations

import re

from ..registry import register
from ..types import Corpus, Passage
from ._base import make_passages

# Headings the renderer writes, and the ones the source PDF uses.
SECTION = re.compile(r"^(?:ingredients?|method|directions?|steps?|preparation)\s*:?\s*$",
                     re.IGNORECASE | re.MULTILINE)


class RecipeChunker:
    """One chunk per recipe. The boundaries are known, so nothing is inferred."""

    def __init__(self, max_chars: int = 0) -> None:
        self.name = "recipe"
        # 0 means never split. A long recipe stays whole, because a recipe is
        # the unit a cook asks for -- half of one is not a smaller answer, it is
        # a wrong one.
        self.params = {"max_chars": max_chars}

    def split(self, corpus: Corpus) -> list[Passage]:
        limit = self.params["max_chars"]
        bounds: list[tuple[int, int]] = []
        for span in corpus.spans:
            if not limit or span.length <= limit:
                bounds.append((span.start, span.end))
                continue
            # Only oversized recipes get cut, and only on a line break so a
            # sentence is never severed mid-word.
            cursor = span.start
            while cursor < span.end:
                stop = min(cursor + limit, span.end)
                if stop < span.end:
                    newline = corpus.text.rfind("\n", cursor, stop)
                    if newline > cursor:
                        stop = newline
                bounds.append((cursor, stop))
                cursor = stop
        return make_passages(corpus, bounds, self.name, self.params)


@register("chunker", "recipe")
def make_recipe(max_chars: int = 0) -> RecipeChunker:
    return RecipeChunker(max_chars)


class RecipeSectionChunker:
    """Split at section headings, and repeat the title into every piece.

    Two things a cook asks about live in one recipe: what goes in it, and what
    you do with it. Splitting there is a real boundary rather than an arbitrary
    one -- but a bare ingredient list is useless without a dish name, which is
    exactly the failure this exists to prevent. So the title is prefixed onto
    each piece.

    That prefix costs a little duplication and buys two things: every chunk is
    answerable on its own, and the title's words are searchable from the method
    section, so "how do I make lemon rice" can match the steps and not only the
    heading.
    """

    def __init__(self, min_chars: int = 220) -> None:
        self.name = "recipe_sections"
        self.params = {"min_chars": min_chars}

    def split(self, corpus: Corpus) -> list[Passage]:
        passages: list[Passage] = []
        index = 0
        for span in corpus.spans:
            body = corpus.text[span.start:span.end]
            cuts = [m.start() for m in SECTION.finditer(body)]
            # A short recipe is left whole; splitting it produces fragments too
            # small to answer anything.
            if not cuts or span.length < self.params["min_chars"] * 2:
                pieces = [(0, len(body))]
            else:
                edges = sorted({0, *cuts, len(body)})
                pieces = [(a, b) for a, b in zip(edges, edges[1:]) if b - a > 0]

            title = (span.title or "").strip()
            for start, end in pieces:
                text = body[start:end].strip()
                if len(text) < 20:
                    continue
                if title and not text.lower().startswith(title.lower()[:24]):
                    text = f"{title}\n{text}"
                passages.append(Passage(
                    passage_id=f"{self.name}-{index:05d}",
                    text=text,
                    start=span.start + start,
                    end=span.start + end,
                    meta={"strategy": self.name, "params": self.params,
                          "recipe_ids": [span.recipe_id],
                          "source": corpus.source, "title": title}))
                index += 1
        return passages


@register("chunker", "recipe_sections")
def make_recipe_sections(min_chars: int = 220) -> RecipeSectionChunker:
    return RecipeSectionChunker(min_chars)