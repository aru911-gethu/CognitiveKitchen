"""Turn a natural-language question into graph conditions.

An LLM does this, not a keyword list. "no lexicon matching" was the whole point:
"I'm avoiding dairy", "nothing with milk in it" and "lactose free" are the same
condition and share no words, so any pattern table would need endless upkeep.

Answers are cached to disk by question, because evaluation replays the same
questions repeatedly and there is no reason to pay twice. Cost and latency go
into a Ledger so the Stage 3 table can show honestly what the graph route costs
against retrievers that are free.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...config import settings
from ..telemetry import Ledger, price_of

CATEGORIES = ("dairy", "gluten", "nut", "meat", "fish", "egg")

PROMPT = """Read the cooking question and return only JSON, no prose.

{{"include": [...], "exclude": [...], "exclude_categories": [...],
 "substitute_for": "...", "course": "...", "dish": "..." }}

  include             ingredients the recipe MUST contain, as plain names
  exclude             specific ingredients it must NOT contain
  exclude_categories  any of: dairy, gluten, nut, meat, fish, egg
                      use this when the question rules out a whole family:
                      "no dairy", "lactose free"  -> ["dairy"]
                      "vegan"                     -> ["dairy","egg","meat","fish"]
                      "gluten free"               -> ["gluten"]
                      "nut allergy"               -> ["nut"]

                      "no meat", "avoiding meat"   -> ["meat","fish"]
                      "vegetarian"                 -> ["meat","fish"]
                      Someone avoiding meat does not want prawns, so flesh of
                      any kind is excluded in both cases. Use ["fish"] alone
                      only when the question rules out fish while allowing
                      meat, which is rare.
  substitute_for      the ingredient the cook has run out of and wants to
                      replace, "" if none
  course              one of bread, rice, main, side, condiment, snack,
                      dessert, drink, soup, salad, breakfast, spice-mix.
                      "" when the question names no course. This field is
                      INDEPENDENT of include: filling it never excuses leaving
                      include empty.
                      "a bread dish with no dairy" -> course bread, include []
                      "what can I cook with cumin" -> course "",    include [cumin]
                      "a rice dish with paneer"    -> course rice,  include [paneer]
  dish                the specific dish named, "" if none

Rules:
- "What can I cook with X", "I have some X left over", "Which recipes use X"
  all put X in include. These are the commonest questions asked; do not leave
  include empty for them.
- Asking for a REPLACEMENT is not the same as ruling something out.
  "What can I use instead of ghee?", "I have run out of milk. What now?",
  "Is there a replacement for cardamom?" all mean substitute_for, and
  exclude MUST stay empty. The cook still wants the recipe that uses it;
  they want to know what to put in its place.
  Contrast: "a dish with no ghee" IS an exclusion.
- Only fill exclude/exclude_categories when the question genuinely rules
  something out. A plain lookup like "how do I make dosa" has all lists empty.
- Ingredients the question merely mentions as context are not include terms.
- Never put a category name in include or exclude; categories go in
  exclude_categories only.

Question: {question}"""


@dataclass
class Constraints:
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    exclude_categories: list[str] = field(default_factory=list)
    substitute_for: str = ""
    course: str = ""
    dish: str = ""
    source: str = "llm"

    @property
    def restricting(self) -> bool:
        """True when the graph can do something a retriever cannot.

        Course alone does not count: a retriever finds "a bread dish" perfectly
        well by similarity. Only absence needs a traversal.
        """
        return bool(self.exclude or self.exclude_categories)

    @property
    def narrows(self) -> bool:
        """True when the graph can narrow the candidate set at all."""
        return bool(self.exclude or self.exclude_categories
                    or self.include or self.course)

    @property
    def empty(self) -> bool:
        return not (self.include or self.exclude or self.exclude_categories
                    or self.substitute_for)

    def as_dict(self) -> dict[str, Any]:
        return {"include": self.include, "exclude": self.exclude,
                "exclude_categories": self.exclude_categories,
                "substitute_for": self.substitute_for,
                "course": self.course, "dish": self.dish,
                "source": self.source}


def cache_path() -> Path:
    return settings.data_dir / "eval" / "constraints.json"


def _load() -> dict[str, Any]:
    path = cache_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save(cache: dict[str, Any]) -> None:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8")


def _key(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()[:16]


def _parse(reply: str) -> Constraints:
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return Constraints(source="unparsed")
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Constraints(source="unparsed")

    def strings(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(v).strip().lower() for v in value if str(v).strip()]

    families = [c for c in strings(obj.get("exclude_categories")) if c in CATEGORIES]
    # a category that slipped into exclude belongs in exclude_categories
    excluded = []
    for item in strings(obj.get("exclude")):
        if item in CATEGORIES:
            if item not in families:
                families.append(item)
        else:
            excluded.append(item)
    substitute_for = str(obj.get("substitute_for") or "").strip().lower()
    if substitute_for:
        # a replacement request is not an exclusion: the cook still wants the
        # recipe that uses the ingredient, so filtering it away destroys the
        # only useful answer
        excluded = [e for e in excluded if e != substitute_for]
    from .course import COURSES

    course = str(obj.get("course") or "").strip().lower()
    return Constraints(
        include=[i for i in strings(obj.get("include")) if i not in CATEGORIES],
        exclude=excluded,
        exclude_categories=families,
        substitute_for=substitute_for,
        course=course if course in COURSES else "",
        dish=str(obj.get("dish") or "").strip(),
    )


class ConstraintExtractor:
    def __init__(self, generator: Any = None, model: str = "gpt4o-mini",
                 use_cache: bool = True) -> None:
        self.model = model
        self.use_cache = use_cache
        self._generator = generator
        self.ledger = Ledger()
        self.calls = 0
        self.cache_hits = 0

    def _gen(self):
        if self._generator is None:
            from ..registry import build, discover

            discover("cognitive_kitchen.rag.generate")
            self._generator = build("generator", self.model)
        return self._generator

    @property
    def cost_usd(self) -> float:
        """Read the generator's own ledger: it is the thing that spent tokens."""
        generator = self._generator
        if generator is None:
            return 0.0
        if hasattr(generator, "cost_usd"):
            return generator.cost_usd
        model = getattr(generator, "model_name", "")
        return round(sum(price_of(model, s.tokens_in, s.tokens_out)
                         for s in getattr(generator, "ledger", self.ledger)
                         .spans.values()), 6)

    @property
    def seconds(self) -> float:
        span = self.ledger.spans.get("constraint.extract")
        return round(span.seconds, 2) if span else 0.0

    def extract(self, question: str) -> Constraints:
        cache = _load() if self.use_cache else {}
        key = _key(question)
        if key in cache:
            self.cache_hits += 1
            entry = dict(cache[key])
            entry.pop("question", None)
            entry["source"] = entry.get("source", "cache")
            return Constraints(**entry)

        started = time.perf_counter()
        try:
            reply = self._gen().complete(PROMPT.format(question=question),
                                         max_tokens=200)
            result = _parse(reply)
        except Exception:
            result = Constraints(source="failed")
        self.calls += 1
        self.ledger.record("constraint.extract", time.perf_counter() - started)

        if self.use_cache and result.source not in ("failed",):
            cache[key] = {"question": question, **result.as_dict()}
            _save(cache)
        return result