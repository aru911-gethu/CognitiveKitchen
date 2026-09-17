"""The ingredient vocabulary, and the lookups its three consumers need.

    graph          node identity -- one ginger node, not sixteen
    hallucination  "ginger" must not read as invented when the context said
                   "1 inch fresh ginger, chopped"
    chat pantry    the user types "ginger"

Reads data/eval/ingredient_map.json, built once by vocab.build. Nothing else in
the pipeline touches this: retrievers handle wording variation natively, so
forcing canonical names on them would only throw away signal.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

from ...config import settings
from .normalise import normalise


# -- disk -----------------------------------------------------------------
def map_path() -> Path:
    return settings.data_dir / "eval" / "ingredient_map.json"


def load_map() -> dict[str, dict[str, Any]]:
    path = map_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_map(vocabulary: dict[str, dict[str, Any]]) -> Path:
    path = map_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocabulary, indent=2, ensure_ascii=False,
                               sort_keys=True), encoding="utf-8")
    return path


@functools.lru_cache(maxsize=1)
def _map() -> dict[str, dict[str, Any]]:
    return load_map()


def reload() -> None:
    _map.cache_clear()


# -- lookups --------------------------------------------------------------
def resolve(raw: str) -> list[str]:
    """A raw ingredient line -> zero, one or more canonical ingredient names.

    Zero means the line is not an ingredient. More than one means the PDF
    wrapped two ingredients onto a single line.
    """
    key = normalise(raw)
    if key is None:
        return []
    entry = _map().get(key)
    if entry is None:
        return [key]                       # unseen: the rules are the fallback
    if entry.get("split"):
        return [s for s in entry["split"] if s]
    canonical = entry.get("canonical")
    return [canonical] if canonical else []


def canonical(raw: str) -> str | None:
    names = resolve(raw)
    return names[0] if names else None


def category(name: str) -> str:
    """Allergen family: dairy, gluten, nut, meat, fish, egg, or none."""
    entry = _map().get(name)
    if entry:
        return entry.get("category") or "none"
    for value in _map().values():
        if value.get("canonical") == name:
            return value.get("category") or "none"
    return "none"


def all_canonical() -> set[str]:
    out: set[str] = set()
    for entry in _map().values():
        if entry.get("split"):
            out.update(entry["split"])
        elif entry.get("canonical"):
            out.add(entry["canonical"])
    return out


def same_ingredient(a: str, b: str) -> bool:
    """Do two wordings mean the same ingredient?

    Deliberately false for coriander seeds vs coriander leaves: the seed is a
    spice, the leaf is a herb, and recipes use both.
    """
    ra, rb = set(resolve(a)), set(resolve(b))
    if ra and rb and (ra & rb):
        return True
    na, nb = normalise(a) or a.lower(), normalise(b) or b.lower()
    return na == nb or na in nb or nb in na
