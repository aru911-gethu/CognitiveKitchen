"""Generation strategies: four ways to turn retrieved chunks into an answer.

All four are grounded, and that is deliberate -- the earlier idea of a permissive
variant was dropped because a cooking assistant that invents quantities is not a
strategy, it is a defect. What varies is how the context is arranged and how many
model calls it takes:

    stuff_strict   every chunk, in rank order, one call            the baseline
    reordered      best chunks first AND last, one call            position effect
    structured     forced sections, one call                       shape
    map_reduce     one call per chunk, then a combining call       cost vs recall

The interesting comparison is not who wins on faithfulness -- they are all
grounded -- but what each costs and whether the shape survives.
"""
from __future__ import annotations

import time
from typing import Any

from ..registry import register
from ..types import Answer, Passage

REFUSAL = "NOT_IN_CONTEXT"


def _join(passages: list[Passage]) -> str:
    return "\n\n---\n\n".join(p.text for p in passages)


class _Base:
    name = "base"
    params: dict[str, Any] = {}

    def __init__(self, generator=None, model: str = "qwen") -> None:
        self._generator = generator
        self._model = model

    def generator(self):
        if self._generator is None:
            from ..registry import build, discover

            discover("cognitive_kitchen.rag.generate")
            self._generator = build("generator", self._model)
        return self._generator

    def _answer(self, question: str, contexts: list[Passage],
                text: str, started: float) -> Answer:
        return Answer(question=question, text=text, contexts=list(contexts),
                      refused=REFUSAL in text,
                      model=f"{self.name}:{getattr(self.generator(), 'name', '')}",
                      elapsed_s=round(time.perf_counter() - started, 2))


# ---------------------------------------------------------------------------
class StuffStrict(_Base):
    """Everything in rank order, one call. The control the others are read against."""

    name = "stuff_strict"

    def generate(self, question: str, contexts: list[Passage]) -> Answer:
        started = time.perf_counter()
        if not contexts:
            return self._answer(question, contexts, REFUSAL, started)
        reply = self.generator().answer(question, contexts)
        return self._answer(question, contexts, reply.text, started)


@register("strategy", "stuff_strict")
def make_stuff(generator=None, model: str = "qwen") -> StuffStrict:
    return StuffStrict(generator, model)


# ---------------------------------------------------------------------------
class Reordered(_Base):
    """Strongest chunks at both ends, weakest buried in the middle.

    Long contexts are read unevenly: attention favours the beginning and the end,
    so a chunk placed in the middle of five is the one most likely to be missed.
    This reorders rank 1,2,3,4,5 as 1,3,5,4,2 -- same chunks, same count, same
    cost, only the seating plan changes. If the metric moves, position was
    costing accuracy and no amount of better retrieval would have shown it.
    """

    name = "reordered"

    def generate(self, question: str, contexts: list[Passage]) -> Answer:
        started = time.perf_counter()
        if not contexts:
            return self._answer(question, contexts, REFUSAL, started)
        front, back = [], []
        for index, passage in enumerate(contexts):
            (front if index % 2 == 0 else back).append(passage)
        arranged = front + list(reversed(back))
        reply = self.generator().answer(question, arranged)
        return self._answer(question, arranged, reply.text, started)


@register("strategy", "reordered")
def make_reordered(generator=None, model: str = "qwen") -> Reordered:
    return Reordered(generator, model)


# ---------------------------------------------------------------------------
STRUCTURED_PROMPT = """Recipes:
{context}

Question: {question}

Answer using only the recipes above, in exactly this shape:

DISH: <name, or {refusal} if the recipes do not answer the question>
INGREDIENTS:
- <ingredient with quantity where given>
STEPS:
1. <step>
NOTES: <anything the recipes say about timing or servings, or "none">

Do not add an ingredient that is not listed above."""


class Structured(_Base):
    """Forced sections. Tests whether shape helps or crowds out the answer.

    A cook wants a list they can follow, not a paragraph. Fixing the shape also
    makes the answer machine-checkable: the ingredient lines can be pulled out
    and compared against the context, which is what NoInventedIngredients does.
    """

    name = "structured"

    def generate(self, question: str, contexts: list[Passage]) -> Answer:
        started = time.perf_counter()
        if not contexts:
            return self._answer(question, contexts, REFUSAL, started)
        prompt = STRUCTURED_PROMPT.format(context=_join(contexts),
                                          question=question, refusal=REFUSAL)
        text = self.generator().complete(prompt, max_tokens=320)
        return self._answer(question, contexts, text, started)


@register("strategy", "structured")
def make_structured(generator=None, model: str = "qwen") -> Structured:
    return Structured(generator, model)


# ---------------------------------------------------------------------------
MAP_PROMPT = """Recipe:
{chunk}

Question: {question}

If this recipe helps answer the question, summarise only the relevant part in at
most three lines, keeping quantities exactly as written. If it does not help,
reply {refusal} and nothing else."""

REDUCE_PROMPT = """Notes gathered from separate recipes:
{notes}

Question: {question}

Write one answer from these notes. Use only what they say. If none of them
answer the question, reply {refusal}."""


class MapReduce(_Base):
    """Read each chunk alone, then combine the notes.

    One call per chunk plus one to combine, so five chunks cost six calls. The
    trade is that no chunk can be crowded out by its neighbours -- each gets the
    model's full attention -- against paying several times the latency. On CPU
    that difference is minutes, not milliseconds, which is why it is measured
    rather than assumed.
    """

    name = "map_reduce"

    def __init__(self, generator=None, model: str = "qwen",
                 map_tokens: int = 110) -> None:
        super().__init__(generator, model)
        self.params = {"map_tokens": map_tokens}

    def generate(self, question: str, contexts: list[Passage]) -> Answer:
        started = time.perf_counter()
        if not contexts:
            return self._answer(question, contexts, REFUSAL, started)

        notes: list[str] = []
        for passage in contexts:
            note = self.generator().complete(
                MAP_PROMPT.format(chunk=passage.text, question=question,
                                  refusal=REFUSAL),
                max_tokens=self.params["map_tokens"])
            if REFUSAL not in note and note.strip():
                notes.append(note.strip())

        if not notes:
            return self._answer(question, contexts, REFUSAL, started)

        text = self.generator().complete(
            REDUCE_PROMPT.format(notes="\n\n".join(f"- {n}" for n in notes),
                                 question=question, refusal=REFUSAL),
            max_tokens=260)
        return self._answer(question, contexts, text, started)


@register("strategy", "map_reduce")
def make_map_reduce(generator=None, model: str = "qwen",
                    map_tokens: int = 110) -> MapReduce:
    return MapReduce(generator, model, map_tokens)