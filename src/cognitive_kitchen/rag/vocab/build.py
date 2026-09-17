"""Build data/eval/ingredient_map.json from the ingested recipes.

    uv run python -m cognitive_kitchen.rag.vocab.build

Two tiers:

    normalise.py   rules strip quantities, units and preparation words
    curated.py     the hand-decided vocabulary for this corpus

A name that appears in neither keeps its normalised form as its own canonical
entry. That default is safe: redundant at worst, never wrong.
"""
from __future__ import annotations

import json
from typing import Any

from ...config import settings
from . import curated
from .normalise import normalise
from .ingredients import map_path, save_map


def corpus_names() -> dict[str, int]:
    """Distinct tier-1 names in the ingested recipes, with frequencies."""
    freq: dict[str, int] = {}
    for path in sorted((settings.data_dir / "ingested").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for recipe in doc.get("recipes", []):
            for line in recipe.get("ingredient_lines") or []:
                name = normalise(line)
                if name:
                    freq[name] = freq.get(name, 0) + 1
    return freq


def build() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    freq = corpus_names()
    hand = curated.build()

    vocabulary: dict[str, dict[str, Any]] = {}
    unmapped: list[str] = []
    for name in sorted(freq):
        if name in hand:
            vocabulary[name] = hand[name]
        else:
            unmapped.append(name)
            vocabulary[name] = {"canonical": name, "category": "none", "split": None}

    # curated canonicals that no raw line spells exactly still need entries
    for name, entry in hand.items():
        vocabulary.setdefault(name, entry)

    canonical: set[str] = set()
    for entry in vocabulary.values():
        if entry.get("split"):
            canonical.update(entry["split"])
        elif entry.get("canonical"):
            canonical.add(entry["canonical"])

    categories: dict[str, int] = {}
    for entry in vocabulary.values():
        categories[entry["category"]] = categories.get(entry["category"], 0) + 1

    stats = {
        "corpus_names": len(freq),
        "decided_by_hand": len(freq) - len(unmapped),
        "unmapped": unmapped,
        "entries": len(vocabulary),
        "canonical": len(canonical),
        "rejected": sum(1 for v in vocabulary.values()
                        if not v["canonical"] and not v["split"]),
        "splits": sum(1 for v in vocabulary.values() if v["split"]),
        "orphan_split_targets": sorted(
            {t for v in vocabulary.values() for t in (v.get("split") or [])
             if t not in canonical}),
        "categories": dict(sorted(categories.items())),
    }
    return vocabulary, stats


def main() -> None:
    vocabulary, stats = build()
    save_map(vocabulary)
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {map_path()}")


if __name__ == "__main__":
    main()
