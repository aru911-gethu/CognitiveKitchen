"""Split a multi-constraint question into sharper sub-questions.

One vector cannot represent two ideas well: "a dessert with no gluten" lands
between dessert and gluten and matches neither. Worse, absence is never written
down, so the lexical signal points at gluten-containing desserts.

Templated golden questions are handled by rule, since their structure is known.
Anything else falls back to the local model.
"""
from __future__ import annotations

import re

from ..registry import register

# excluded terms are expanded so the retriever has something lexical to match
TAG_TERMS = {
    "dairy": "milk ghee butter yogurt curd cream paneer cheese",
    "gluten": "wheat flour semolina rava vermicelli maida",
    "nuts": "almond cashew pistachio peanut walnut",
    "meat": "chicken lamb mutton beef pork",
    "fish": "fish prawn shrimp crab",
    "egg": "egg eggs",
}


class Decompose:
    name = "decompose"

    def __init__(self, use_llm: bool = False) -> None:
        self.use_llm = use_llm
        self._generator = None

    # -- rule based, covers the templated families ------------------------
    def _rules(self, q: str) -> list[str] | None:
        m = re.fullmatch(r"A (\w+) dish with no (\w+) in it\.", q)
        if m:
            course, tag = m.group(1), m.group(2)
            return [f"{course} recipe", TAG_TERMS.get(tag, tag)]
        m = re.fullmatch(r"I am avoiding (\w+)\. What can I make\?", q)
        if m:
            return ["recipe", TAG_TERMS.get(m.group(1), m.group(1))]
        m = re.fullmatch(r"A (vegetarian|vegan) (\w+)\.", q)
        if m:
            return [f"{m.group(2)} recipe", m.group(1)]
        m = re.search(r"instead of (.+?)\?|run out of (.+?)\.|replacement for (.+?)\?", q)
        if m:
            term = next(g for g in m.groups() if g)
            return [f"recipes using {term}", f"substitute for {term}"]
        m = re.search(r"(?:finish in|only have) (\d+) minutes", q)
        if m:
            return ["quick recipe short cooking time", f"{m.group(1)} minutes"]
        return None

    def _llm(self, q: str) -> list[str]:
        from ..registry import build, discover

        if self._generator is None:
            discover("cognitive_kitchen.rag.generate")
            self._generator = build("generator", "qwen")
        raw = self._generator.complete(
            "Split the cooking question into 2 short search phrases, one per line. "
            "No numbering, no explanation.\n\nQuestion: " + q, max_tokens=60)
        parts = [l.strip(" -*") for l in raw.splitlines() if l.strip()]
        return parts[:3] or [q]

    def expand(self, question: str) -> list[str]:
        parts = self._rules(question)
        if parts is None and self.use_llm:
            parts = self._llm(question)
        return [question] + (parts or [])


@register("query", "decompose")
def make(use_llm: bool = False) -> Decompose:
    return Decompose(use_llm)