"""Golden dataset access.

The only module that opens golden_dataset.json. It is the answer key, so it is
read at scoring time and never by the pipeline that produces answers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ...config import settings
from .tokens import tokenise


@dataclass(frozen=True)
class GoldenRecipe:
    recipe_id: str
    title: str
    pages: tuple[int, ...]
    words: frozenset[str]                 # title + ingredients + method
    ingredient_names: tuple[str, ...]
    atoms: tuple[str, ...] = field(default_factory=tuple)


def golden_path() -> Path:
    return settings.data_dir / "golden_dataset.json"


@lru_cache(maxsize=1)
def load_golden() -> tuple[GoldenRecipe, ...]:
    raw = json.loads(golden_path().read_text(encoding="utf-8"))
    atoms_by_id = {s["recipe_id"]: tuple(s.get("atoms", []))
                   for s in raw.get("corpus", {}).get("spans", [])}
    out: list[GoldenRecipe] = []
    for r in raw["recipes"]:
        blob = [r["title"]]
        blob += [(i.get("display") or i.get("name") or "") for i in r["ingredients"]]
        blob += [(s.get("text") or "") for s in r["method"]]
        words = tokenise(" ".join(blob))
        if len(words) < 5:
            continue
        out.append(GoldenRecipe(
            recipe_id=r["id"],
            title=r["title"],
            pages=tuple(r["source_pages"]),
            words=frozenset(words),
            ingredient_names=tuple(
                i["name"].lower().strip() for i in r["ingredients"]
                if i.get("name") and len(i["name"]) > 3),
            atoms=atoms_by_id.get(r["id"], ()),
        ))
    return tuple(out)


@lru_cache(maxsize=1)
def load_questions() -> tuple[dict, ...]:
    raw = json.loads(golden_path().read_text(encoding="utf-8"))
    return tuple(raw["evaluation"]["queries"])