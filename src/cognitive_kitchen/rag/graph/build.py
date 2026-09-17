"""Write the shaped graph into Neo4j.

    uv run ck-graph            rebuild from data/ingested/
    uv run ck-graph --dry-run  shape only, report counts, touch nothing

Everything is UNWIND-batched: Aura free tier does not enjoy 2,000 single-row
statements, and batching turns a slow load into a few seconds.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any

from . import client
from .course import classify as classify_courses
from .schema import GraphData, load_recipes, shape

BATCH = 500

CONSTRAINTS = (
    "CREATE CONSTRAINT recipe_key IF NOT EXISTS "
    "FOR (r:Recipe) REQUIRE r.key IS UNIQUE",
    "CREATE CONSTRAINT ingredient_name IF NOT EXISTS "
    "FOR (i:Ingredient) REQUIRE i.name IS UNIQUE",
    "CREATE CONSTRAINT category_name IF NOT EXISTS "
    "FOR (c:Category) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT step_id IF NOT EXISTS "
    "FOR (s:Step) REQUIRE s.step_id IS UNIQUE",
    "CREATE CONSTRAINT action_name IF NOT EXISTS "
    "FOR (a:Action) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT equipment_name IF NOT EXISTS "
    "FOR (e:Equipment) REQUIRE e.name IS UNIQUE",
)


def _batched(rows: list[dict[str, Any]], size: int = BATCH):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def wipe() -> int:
    """Delete everything. The old graph was built from the golden dataset --
    score-side data on the build side -- so it is not worth preserving."""
    removed = 0
    while True:
        rows = client.run(
            "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS n")
        n = rows[0]["n"] if rows else 0
        removed += n
        if n == 0:
            return removed


def drop_constraints() -> list[str]:
    """Remove every existing constraint and index.

    DETACH DELETE clears nodes but leaves the schema behind, so a constraint
    from an earlier graph shape survives the wipe and then rejects the new one.
    The previous attempt had recipe_id UNIQUE; this build wants recipe_id to be
    an ordinary repeatable property, because ids restart per ingestion run.
    """
    dropped: list[str] = []
    for row in client.run("SHOW CONSTRAINTS YIELD name RETURN name"):
        client.run(f"DROP CONSTRAINT {row['name']} IF EXISTS")
        dropped.append(row["name"])
    for row in client.run("SHOW INDEXES YIELD name, type RETURN name, type"):
        if row.get("type") != "LOOKUP":
            try:
                client.run(f"DROP INDEX {row['name']} IF EXISTS")
                dropped.append(row["name"])
            except Exception:
                pass
    return dropped


def ensure_constraints() -> None:
    for statement in CONSTRAINTS:
        try:
            client.run(statement)
        except Exception:
            pass          # already exists, or an older server spells it differently


def load(data: GraphData, progress=None) -> dict[str, int]:
    written: dict[str, int] = {}

    def step(label: str, cypher: str, rows: list[dict[str, Any]]) -> None:
        total = 0
        for chunk in _batched(rows):
            counters = client.write(cypher, rows=chunk)
            total += counters["nodes"] + counters["relationships"]
        written[label] = len(rows)
        if progress:
            progress(label, len(rows))

    step("Category", """
        UNWIND $rows AS row
        MERGE (:Category {name: row.name})""",
        [{"name": c} for c in data.categories])

    step("Ingredient", """
        UNWIND $rows AS row
        MERGE (i:Ingredient {name: row.name})
        SET i.category = row.category
        WITH i, row WHERE row.category <> 'none'
        MERGE (c:Category {name: row.category})
        MERGE (i)-[:IN_CATEGORY]->(c)""", data.ingredients)

    step("Recipe", """
        UNWIND $rows AS row
        MERGE (r:Recipe {key: row.key})
        SET r.course = row.course,
            r.recipe_id = row.recipe_id, r.source_file = row.source_file,
            r.is_recipe = row.is_recipe, r.title = row.title,
            r.source_type = row.source_type, r.origin = row.origin,
            r.url = row.url, r.pages = row.pages,
            r.n_ingredients = row.n_ingredients, r.n_steps = row.n_steps,
            r.text = row.text""", data.recipes)

    step("USES", """
        UNWIND $rows AS row
        MATCH (r:Recipe {key: row.key})
        MATCH (i:Ingredient {name: row.name})
        MERGE (r)-[u:USES]->(i)
        SET u.raw = row.raw""", data.uses)

    step("Step", """
        UNWIND $rows AS row
        MATCH (r:Recipe {key: row.key})
        MERGE (s:Step {step_id: row.step_id})
        SET s.index = row.index, s.text = row.text
        MERGE (r)-[:HAS_STEP]->(s)""", data.steps)

    step("NEXT", """
        UNWIND $rows AS row
        MATCH (a:Step {step_id: row.a})
        MATCH (b:Step {step_id: row.b})
        MERGE (a)-[:NEXT]->(b)""", data.next_steps)

    step("Action", """
        UNWIND $rows AS row
        MERGE (a:Action {name: row.name})
        WITH a, row
        MATCH (s:Step {step_id: row.step_id})
        MERGE (s)-[:PERFORMS]->(a)""", data.performs)

    step("Equipment", """
        UNWIND $rows AS row
        MERGE (e:Equipment {name: row.name})
        WITH e, row
        MATCH (s:Step {step_id: row.step_id})
        MERGE (s)-[:NEEDS]->(e)""", data.needs)

    return written


def rebuild(dry_run: bool = False, progress=None) -> dict[str, Any]:
    started = time.perf_counter()
    recipes = load_recipes()
    data = shape(recipes)
    report: dict[str, Any] = {
        "ingested_recipes": len(recipes),
        "recipes_kept": len(data.recipes),
        "recipes_skipped_as_front_matter": len(recipes) - len(data.recipes),
        "shaped": data.counts(),
    }
    if dry_run:
        report["dry_run"] = True
        report["elapsed_s"] = round(time.perf_counter() - started, 2)
        return report

    # course is inferred once and cached; the graph cannot answer "a bread dish
    # with no dairy" without it
    courses = classify_courses(
        [{"key": r["key"], "title": r["title"],
          "ingredients": [u["name"] for u in data.uses if u["key"] == r["key"]]}
         for r in data.recipes],
        progress=progress and (lambda i, n, p_, s: progress(f"course {i}/{n}", p_)))
    for recipe in data.recipes:
        recipe["course"] = courses.get(recipe["key"], "other")
    counts: dict[str, int] = {}
    for recipe in data.recipes:
        counts[recipe["course"]] = counts.get(recipe["course"], 0) + 1
    report["courses"] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    report["dropped_schema"] = drop_constraints()
    report["deleted"] = wipe()
    ensure_constraints()
    report["written"] = load(data, progress=progress)
    report["health"] = client.health()
    report["elapsed_s"] = round(time.perf_counter() - started, 2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                       help="shape and report, but do not touch the database")
    args = parser.parse_args()
    report = rebuild(dry_run=args.dry_run,
                     progress=lambda label, n: print(f"  {label:12s} {n}"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()