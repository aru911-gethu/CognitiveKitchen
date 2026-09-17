"""Turn ingested recipes into graph nodes and edges.

Pure Python: no driver, no Cypher. Keeping the shaping separate from the writing
means the node model can be inspected and tested without a database, and the
Cypher in build.py stays boring.

The model, and why each part exists:

    Recipe                     the thing we retrieve. Keyed by "<source>:<id>"
                               because recipe_id restarts at r_0001 in every
                               ingestion run, so the raw id is not unique.
                               is_recipe is false for section headers and for
                               continuation pages whose ingredient block landed
                               on a different page -- they keep their steps, but
                               exclusion queries must not count them as answers.
    Ingredient                 158 canonical names from the vocabulary, so
                               "ginger" is one node and not sixteen
    Category                   dairy, gluten, nut, meat, fish, egg -- this is
                               what makes "without dairy" a traversal instead
                               of a similarity guess
    Step                       ordered, so "what do I do after frying" works
    Action, Equipment          extracted from step text; the showcase for
                               "which recipes need a pressure cooker"

    (Recipe)-[:USES]->(Ingredient)          carries the raw line it came from
    (Ingredient)-[:IN_CATEGORY]->(Category)
    (Recipe)-[:HAS_STEP]->(Step)
    (Step)-[:NEXT]->(Step)
    (Step)-[:PERFORMS]->(Action)
    (Step)-[:NEEDS]->(Equipment)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from ...config import settings
from ..vocab import ingredients as vocab

# Cooking verbs worth a node. Extraction, not constraint answering -- the
# constraint questions are answered by traversal over USES and IN_CATEGORY.
ACTIONS = {
    "fry": ("fry", "frying", "fries", "fried"),
    "deep fry": ("deep fry", "deep-fry", "deep frying"),
    "saute": ("saute", "sauté", "sauteed", "sautéed", "sauting"),
    "boil": ("boil", "boiling", "boiled"),
    "simmer": ("simmer", "simmering", "simmered"),
    "steam": ("steam", "steaming", "steamed"),
    "roast": ("roast", "roasting", "roasted"),
    "grind": ("grind", "grinding", "ground", "grinder"),
    "blend": ("blend", "blending", "blended"),
    "knead": ("knead", "kneading", "kneaded"),
    "soak": ("soak", "soaking", "soaked"),
    "marinate": ("marinate", "marinating", "marinated", "marinade"),
    "temper": ("temper", "tempering", "tadka", "seasoning"),
    "garnish": ("garnish", "garnishing", "garnished"),
    "bake": ("bake", "baking", "baked"),
    "grill": ("grill", "grilling", "grilled"),
    "pressure cook": ("pressure cook", "pressure-cook", "pressure cooking"),
    "stir": ("stir", "stirring", "stirred"),
    "mash": ("mash", "mashing", "mashed"),
    "strain": ("strain", "straining", "strained", "drain"),
    "cool": ("cool", "cooling", "cooled"),
    "serve": ("serve", "serving", "served"),
}

EQUIPMENT = {
    "pressure cooker": ("pressure cooker", "pressure pan", "cooker"),
    "kadai": ("kadai", "kadhai", "wok"),
    "tawa": ("tawa", "tava", "griddle"),
    "pan": ("pan", "frying pan", "skillet", "saucepan"),
    "oven": ("oven",),
    "blender": ("blender", "mixie", "mixer", "grinder", "food processor"),
    "idli steamer": ("idli steamer", "steamer", "idli plate"),
    "microwave": ("microwave",),
    "mortar": ("mortar", "pestle"),
    "colander": ("colander", "strainer", "sieve"),
}


# A page needs at least this many ingredients to count as a real recipe.
# Below it: "SOUPS", "Chutneys", or a page whose ingredient list wrapped away.
MIN_INGREDIENTS = 2


@dataclass
class GraphData:
    recipes: list[dict[str, Any]] = field(default_factory=list)
    ingredients: list[dict[str, Any]] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    uses: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    performs: list[dict[str, Any]] = field(default_factory=list)
    needs: list[dict[str, Any]] = field(default_factory=list)
    next_steps: list[dict[str, Any]] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {"Recipe": len(self.recipes),
                "Recipe(real)": sum(1 for r in self.recipes if r["is_recipe"]),
                "Ingredient": len(self.ingredients),
                "Category": len(self.categories), "Step": len(self.steps),
                "USES": len(self.uses), "NEXT": len(self.next_steps),
                "PERFORMS": len(self.performs), "NEEDS": len(self.needs)}


def _found(text: str, table: dict[str, tuple[str, ...]]) -> list[str]:
    low = text.lower()
    hits = []
    for canonical, surface in table.items():
        if any(re.search(rf"\b{re.escape(s)}\b", low) for s in surface):
            hits.append(canonical)
    # "deep fry" implies "fry"; keep the specific one only
    if "deep fry" in hits and "fry" in hits:
        hits.remove("fry")
    if "pressure cook" in hits and "cool" in hits and "pressure cooker" not in low:
        pass
    return hits


def load_recipes() -> list[dict[str, Any]]:
    """Every recipe from every ingestion run, newest file last."""
    out: list[dict[str, Any]] = []
    for path in sorted((settings.data_dir / "ingested").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for recipe in doc.get("recipes", []):
            recipe = dict(recipe)
            recipe["_source_file"] = path.name
            out.append(recipe)
    return out


def shape(recipes: Iterable[dict[str, Any]], *,
          skip_empty: bool = True) -> GraphData:
    """Build the node and edge lists. No database involved."""
    data = GraphData()
    seen_ingredients: dict[str, str] = {}
    seen_categories: set[str] = set()

    for recipe in recipes:
        lines = recipe.get("ingredient_lines") or []
        steps = recipe.get("step_lines") or []
        # a page with no ingredients and no steps is front matter, not a recipe
        if skip_empty and not lines and not steps:
            continue

        source = recipe["_source_file"].rsplit(".", 1)[0]
        rid = f"{source}:{recipe['recipe_id']}"
        resolved: list[tuple[str, str]] = []
        for position, raw in enumerate(lines):
            for name in vocab.resolve(raw):
                resolved.append((name, raw))
                if name not in seen_ingredients:
                    category = vocab.category(name)
                    seen_ingredients[name] = category
                    if category != "none":
                        seen_categories.add(category)

        n_ingredients = len({n for n, _ in resolved})
        data.recipes.append({
            "key": rid,
            "recipe_id": recipe["recipe_id"],
            "source_file": recipe["_source_file"],
            "is_recipe": n_ingredients >= MIN_INGREDIENTS,
            "title": recipe.get("title") or "",
            "source_type": recipe.get("source_type") or "",
            "origin": recipe.get("origin") or "",
            "url": recipe.get("url"),
            "pages": recipe.get("pages") or [],
            "n_ingredients": n_ingredients,
            "n_steps": len(steps),
            "text": recipe.get("raw_text") or "",
        })

        for name, raw in dict.fromkeys(resolved):
            data.uses.append({"key": rid, "name": name, "raw": raw})

        for index, text in enumerate(steps):
            step_id = f"{rid}#s{index:02d}"
            data.steps.append({"step_id": step_id, "key": rid,
                               "index": index, "text": text})
            for action in _found(text, ACTIONS):
                data.performs.append({"step_id": step_id, "name": action})
            for item in _found(text, EQUIPMENT):
                data.needs.append({"step_id": step_id, "name": item})

    # consecutive steps within one recipe, in one pass
    by_recipe: dict[str, list[dict[str, Any]]] = {}
    for s in data.steps:
        by_recipe.setdefault(s["key"], []).append(s)
    for chain in by_recipe.values():
        chain.sort(key=lambda s: s["index"])
        for a, b in zip(chain, chain[1:]):
            data.next_steps.append({"a": a["step_id"], "b": b["step_id"]})

    data.ingredients = [{"name": n, "category": c}
                        for n, c in sorted(seen_ingredients.items())]
    data.categories = sorted(seen_categories)
    return data