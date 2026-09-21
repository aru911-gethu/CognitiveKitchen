"""Traversals that answer the questions similarity cannot.

The whole reason this graph exists. A dense retriever scores "rice without
dairy" by how much the text looks like the question, so it happily returns the
most dairy-laden rice dish in the book -- absence has no embedding. Here it is a
pattern that either matches or does not:

    MATCH (r:Recipe) WHERE NOT (r)-[:USES]->(:Ingredient {category:'dairy'})

Every function returns recipe keys, plain strings, so the caller decides how to
rank them. Only nodes with is_recipe = true are eligible: section headers and
continuation pages have no ingredients, which would make them trivially match
every exclusion.
"""
from __future__ import annotations

from typing import Any, Iterable

from . import client
from ..vocab import ingredients as vocab

REAL = "r.is_recipe = true"


def _names(raw: Iterable[str]) -> list[str]:
    """Map free text onto canonical ingredient names."""
    out: list[str] = []
    for item in raw:
        for name in vocab.resolve(item):
            if name not in out:
                out.append(name)
    return out


# -- presence -------------------------------------------------------------
def using_all(names: Iterable[str]) -> list[str]:
    """Recipes that use every one of these ingredients."""
    wanted = _names(names)
    if not wanted:
        return []
    rows = client.run(f"""
        MATCH (r:Recipe)-[:USES]->(i:Ingredient)
        WHERE {REAL} AND i.name IN $wanted
        WITH r, count(DISTINCT i.name) AS hits
        WHERE hits = $n
        RETURN r.key AS key ORDER BY key""", wanted=wanted, n=len(wanted))
    return [row["key"] for row in rows]


def using_any(names: Iterable[str]) -> list[str]:
    wanted = _names(names)
    if not wanted:
        return []
    rows = client.run(f"""
        MATCH (r:Recipe)-[:USES]->(i:Ingredient)
        WHERE {REAL} AND i.name IN $wanted
        RETURN DISTINCT r.key AS key ORDER BY key""", wanted=wanted)
    return [row["key"] for row in rows]


# -- absence: the families that scored 0.000 in stage 3 -------------------
def without_ingredients(names: Iterable[str]) -> list[str]:
    """Recipes that use none of these ingredients."""
    unwanted = _names(names)
    if not unwanted:
        return all_recipes()
    rows = client.run(f"""
        MATCH (r:Recipe) WHERE {REAL}
          AND NOT EXISTS {{
            MATCH (r)-[:USES]->(i:Ingredient) WHERE i.name IN $unwanted
          }}
        RETURN r.key AS key ORDER BY key""", unwanted=unwanted)
    return [row["key"] for row in rows]


def without_categories(categories: Iterable[str]) -> list[str]:
    """Recipes free of a whole allergen family: dairy, gluten, nut, ..."""
    families = [c for c in categories if c and c != "none"]
    if not families:
        return all_recipes()
    rows = client.run(f"""
        MATCH (r:Recipe) WHERE {REAL}
          AND NOT EXISTS {{
            MATCH (r)-[:USES]->(:Ingredient)-[:IN_CATEGORY]->(c:Category)
            WHERE c.name IN $families
          }}
        RETURN r.key AS key ORDER BY key""", families=families)
    return [row["key"] for row in rows]


def all_recipes() -> list[str]:
    rows = client.run(
        f"MATCH (r:Recipe) WHERE {REAL} RETURN r.key AS key ORDER BY key")
    return [row["key"] for row in rows]


# -- the combined filter the retriever uses -------------------------------
def of_course(course: str) -> list[str]:
    rows = client.run(f"""
        MATCH (r:Recipe) WHERE {REAL} AND r.course = $course
        RETURN r.key AS key ORDER BY key""", course=course)
    return [row["key"] for row in rows]


def filter_recipes(include: Iterable[str] = (), exclude: Iterable[str] = (),
                   exclude_categories: Iterable[str] = (),
                   course: str = "") -> list[str]:
    """Membership, from every kind of condition at once.

    `course` matters more than it looks. Without it "a bread dish with no dairy"
    returned all 86 dairy-free recipes, so precision was 0.01 on a question the
    graph appeared to answer correctly.
    """
    keys: set[str] | None = None

    if course:
        keys = set(of_course(course))

    wanted = _names(include)
    if include and not wanted:
        # The question named an ingredient and the vocabulary could not resolve
        # it. Falling through to "everything" would answer "what can I cook with
        # freshly?" with all 177 recipes, which reads as confidence rather than
        # failure. An empty set is the honest reply.
        return []
    if wanted:
        found = set(using_any(wanted))
        keys = found if keys is None else (keys & found)

    unwanted = _names(exclude)
    families = [c for c in exclude_categories if c and c != "none"]
    if unwanted or families:
        allowed = set(without_ingredients(unwanted)) if unwanted else None
        if families:
            by_family = set(without_categories(families))
            allowed = by_family if allowed is None else (allowed & by_family)
        keys = allowed if keys is None else (keys & allowed)

    return sorted(keys) if keys is not None else all_recipes()


# -- pantry: what can I actually cook tonight ----------------------------
def pantry_gap(have: Iterable[str], max_missing: int = 3,
               limit: int = 20) -> list[dict[str, Any]]:
    """Recipes ranked by how few ingredients you are missing."""
    owned = _names(have)
    rows = client.run(f"""
        MATCH (r:Recipe)-[:USES]->(i:Ingredient)
        WHERE {REAL}
        WITH r, collect(DISTINCT i.name) AS needed
        WITH r, needed,
             [n IN needed WHERE NOT n IN $owned] AS missing
        WHERE size(missing) <= $max_missing
        RETURN r.key AS key, r.title AS title, needed, missing,
               size(missing) AS n_missing, size(needed) AS n_needed
        ORDER BY n_missing ASC, n_needed DESC, key
        LIMIT $limit""", owned=owned, max_missing=max_missing, limit=limit)
    return rows


# -- substitution candidates ---------------------------------------------
def substitutes(name: str, limit: int = 8,
                min_shared_context: int = 3) -> list[dict[str, Any]]:
    """Ingredients that fill the same role, by distributional similarity.

    Plain co-occurrence is the wrong measure and gives the wrong answer: the
    ingredients most often found next to ghee are water, salt and turmeric,
    because those are in almost every recipe. Appearing WITH something is the
    opposite of substituting FOR it.

    So this uses the distributional idea instead -- two ingredients are alike if
    they keep the same company, even when they never meet:

        context(x)  = every other ingredient sharing a recipe with x
        similarity  = jaccard(context(a), context(b))
        penalty     = how often a and b appear together, which argues against
                      substitution, since a recipe using both treats them as
                      different things

        score = similarity * (1 - together / recipes(a))

    Same-category candidates are kept even when the text never suggests it, so
    a dairy swap stays inside dairy.
    """
    resolved = vocab.resolve(name)
    if not resolved:
        return []
    target = resolved[0]
    return client.run("""
        MATCH (t:Ingredient {name: $target})<-[:USES]-(r:Recipe)
        WHERE r.is_recipe = true
        WITH t, collect(DISTINCT r) AS t_recipes
        WITH t, t_recipes, size(t_recipes) AS t_n
        UNWIND t_recipes AS r
        MATCH (r)-[:USES]->(ctx:Ingredient) WHERE ctx <> t
        WITH t, t_n, collect(DISTINCT ctx.name) AS t_context

        MATCH (c:Ingredient) WHERE c <> t
        MATCH (c)<-[:USES]-(cr:Recipe) WHERE cr.is_recipe = true
        WITH t, t_n, t_context, c, collect(DISTINCT cr) AS c_recipes
        WITH t, t_n, t_context, c, c_recipes, size(c_recipes) AS c_n
        UNWIND c_recipes AS cr
        MATCH (cr)-[:USES]->(ctx2:Ingredient) WHERE ctx2 <> c
        WITH t, t_n, t_context, c, c_n,
             collect(DISTINCT ctx2.name) AS c_context

        WITH t, t_n, c, c_n,
             size([x IN t_context WHERE x IN c_context]) AS shared_ctx,
             size(t_context) + size(c_context)
               - size([x IN t_context WHERE x IN c_context]) AS union_ctx
        WHERE shared_ctx >= $min_shared

        OPTIONAL MATCH (both:Recipe)-[:USES]->(t)
        WHERE both.is_recipe = true AND (both)-[:USES]->(c)
        WITH t, c, c_n, t_n, shared_ctx, union_ctx,
             count(DISTINCT both) AS together
        WITH c, shared_ctx, together, t.category AS t_cat,
             toFloat(shared_ctx) / union_ctx AS similarity,
             1.0 - (toFloat(together) / t_n) AS apart
        WHERE (t_cat <> "none" AND c.category = t_cat) OR (t_cat = "none" AND c.category = "none")
        RETURN c.name AS name, c.category AS category,
               shared_ctx AS shared_context, together,
               round(similarity * apart * 1000) / 1000 AS score,
               c.category = t_cat AS same_category
        ORDER BY score DESC
        LIMIT $limit""", target=target, limit=limit,
        min_shared=min_shared_context)


# -- steps, actions, equipment -------------------------------------------
def steps_for(key: str) -> list[dict[str, Any]]:
    return client.run("""
        MATCH (r:Recipe {key: $key})-[:HAS_STEP]->(s:Step)
        OPTIONAL MATCH (s)-[:PERFORMS]->(a:Action)
        OPTIONAL MATCH (s)-[:NEEDS]->(e:Equipment)
        RETURN s.index AS index, s.text AS text,
               collect(DISTINCT a.name) AS actions,
               collect(DISTINCT e.name) AS equipment
        ORDER BY index""", key=key)


def needing_equipment(name: str) -> list[str]:
    rows = client.run(f"""
        MATCH (r:Recipe)-[:HAS_STEP]->(:Step)-[:NEEDS]->(e:Equipment {{name: $name}})
        WHERE {REAL}
        RETURN DISTINCT r.key AS key ORDER BY key""", name=name)
    return [row["key"] for row in rows]


def recipe_cards(keys: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Titles, text and ingredient lists for a set of keys, in one round trip."""
    rows = client.run("""
        MATCH (r:Recipe) WHERE r.key IN $keys
        OPTIONAL MATCH (r)-[:USES]->(i:Ingredient)
        RETURN r.key AS key, r.title AS title, r.text AS text,
               r.pages AS pages, r.recipe_id AS recipe_id,
               collect(DISTINCT i.name) AS ingredients""", keys=list(keys))
    return {row["key"]: row for row in rows}


def stats() -> dict[str, Any]:
    rows = client.run("""
        MATCH (r:Recipe) WHERE r.is_recipe = true
        RETURN count(r) AS real_recipes""")
    families = client.run("""
        MATCH (c:Category)<-[:IN_CATEGORY]-(i:Ingredient)<-[:USES]-(r:Recipe)
        WHERE r.is_recipe = true
        RETURN c.name AS category, count(DISTINCT r) AS recipes,
               count(DISTINCT i) AS ingredients ORDER BY recipes DESC""")
    return {"real_recipes": rows[0]["real_recipes"],
            "by_category": families}