"""One-time course classification for recipe nodes.

Stage 5 needs this. Asked "a bread dish with no dairy", the graph answered the
dairy half and ignored the bread half, so it returned 86 recipes where 1 was
wanted -- precision 0.01. Absence it models; course it did not.

Course cannot come from the ingested data, which has no such field, and it
cannot come from the golden dataset, which is the truth set and must stay on the
scoring side. So it is inferred once from the title and ingredients, cached to
disk, and written onto the Recipe nodes. The same shape as the ingredient
vocabulary: pay once, read forever.

Titles alone are not enough -- "Kootu" and "Adai" say nothing to a keyword rule
-- which is why this is a model call and not a lookup table.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from ...config import settings

COURSES = ("bread", "rice", "main", "side", "condiment", "snack", "dessert",
           "drink", "soup", "salad", "breakfast", "spice-mix", "other")

BATCH = 25

PROMPT = """Classify each Indian recipe into exactly one course.

Allowed: bread, rice, main, side, condiment, snack, dessert, drink, soup,
salad, breakfast, spice-mix, other

Guidance:
  bread       roti, naan, paratha, puri, chapatti, dosa, adai, pesarattu
  rice        pulao, biryani, plain or flavoured rice, bath, upma made of rice
  main        curries and gravies eaten with rice or bread, dal as a main
  side        dry vegetable dishes, poriyal, subzi, kootu
  condiment   chutney, thokku, pickle, raita, sauce
  snack       pakora, samosa, vada, bhujia, sundal
  dessert     halwa, burfi, kheer, payasam, laddu, jamun, sweets
  drink       lassi, thandai, chai, juice, milk drinks
  soup        shorba, rasam, clear soups
  salad       raw vegetable salads, kosambari
  breakfast   idli, upma, poha eaten at breakfast
  spice-mix   masala or podi powders that are an ingredient, not a dish
  other       glossary entries, method fragments, anything that is not a dish

Reply with one JSON object per line and nothing else:
{{"n": <number>, "course": "<course>"}}

Recipes:
{lines}"""


def cache_path() -> Path:
    return settings.data_dir / "eval" / "recipe_course.json"


def load() -> dict[str, str]:
    path = cache_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save(courses: dict[str, str]) -> Path:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(courses, indent=2, ensure_ascii=False,
                               sort_keys=True), encoding="utf-8")
    return path


def _parse(reply: str, batch: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in re.finditer(r"\{[^{}]*\}", reply):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        index, course = obj.get("n"), obj.get("course")
        if not isinstance(index, int) or not 1 <= index <= len(batch):
            continue
        if course in COURSES:
            out[batch[index - 1]["key"]] = course
    return out


def classify(recipes: Iterable[dict[str, Any]], generator=None,
             progress=None) -> dict[str, str]:
    """Classify any recipe not already cached. Returns the whole cache."""
    cache = load()
    todo = [r for r in recipes if r["key"] not in cache]
    if not todo:
        return cache

    if generator is None:
        from ..registry import build, discover

        discover("cognitive_kitchen.rag.generate")
        generator = build("generator", "gpt4o-mini")

    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    for number, batch in enumerate(batches, start=1):
        lines = []
        for index, recipe in enumerate(batch, start=1):
            ingredients = ", ".join(sorted(recipe.get("ingredients") or ())[:10])
            lines.append(f'{index}. "{recipe["title"][:70]}"  [{ingredients}]')
        try:
            reply = generator.complete(PROMPT.format(lines="\n".join(lines)),
                                       max_tokens=20 * len(batch) + 60)
            parsed = _parse(reply, batch)
        except Exception:
            parsed = {}
        for recipe in batch:
            cache[recipe["key"]] = parsed.get(recipe["key"], "other")
        save(cache)
        if progress:
            progress(number, len(batches), len(parsed), len(batch))
    return cache