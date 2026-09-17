"""Stage 3 - retrieval quality, and whether the shortlist is safe to use.

    Hit@k                 1.0 if any correct recipe appears in the shortlist
    Recall@k              proportion of the k slots spent correctly
    MAP                   precision at each hit, averaged
    Diversity@k           distinct recipes in the shortlist / k
    ConstraintRespected   of the recipes handed downstream, how many actually
                          satisfy the restriction the question asked for

The last one is not a quality measure, it is a safety one, and the four above it
cannot see it. Asked "I am avoiding nuts", a dense retriever returns Nut Milk
and a cross-encoder returns Nut Milk, Cashew Chutney and Almond Milk: the better
the ranker, the more confidently wrong, because "avoiding nuts" sits closest in
embedding space to recipes about nuts. Negation has no direction there.

It matters at this stage rather than at generation because a generator handed a
nut recipe can only hallucinate compliance or refuse. Measuring here says
whether retrieval or generation is at fault; measuring only at the end cannot.

Chunk hits are collapsed to recipe level first, keeping the best rank, so a
chunker that fragments a recipe cannot inflate its score by filling the
shortlist with pieces of the same dish.
"""
from __future__ import annotations

import statistics
import time
from typing import Any

from ..telemetry import Ledger, traced
from ..types import Corpus, Scored
from .atoms import Fingerprint, build_fingerprints, matches
from .golden import load_golden, load_questions

RESTRICTING = {"exclusion", "constraint"}


def recipe_ingredients(source: str = "") -> dict[str, set[str]]:
    """recipe_id -> canonical ingredients, read from the ingested runs.

    Deliberately not read from the graph: this metric must still be computable
    when Neo4j is unreachable, and it should judge every retriever by the same
    yardstick rather than by the graph's own view of the world.
    """
    import json

    from ...config import settings
    from ..vocab import ingredients as vocab

    out: dict[str, set[str]] = {}
    for path in sorted((settings.data_dir / "ingested").glob("*.json")):
        if source and not path.name.startswith(source):
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        for recipe in doc.get("recipes", []):
            names: set[str] = set()
            for line in recipe.get("ingredient_lines") or []:
                names.update(vocab.resolve(line))
            out[recipe["recipe_id"]] = names
    return out


def violations(hits: list[Scored], constraints,
               ings: dict[str, set[str]]) -> list[tuple[str, list[str]]]:
    """Recipes in the shortlist that break the restriction, and what broke it."""
    from ..vocab import ingredients as vocab

    banned_families = set(constraints.exclude_categories or ())
    banned_names = set()
    for item in constraints.exclude or ():
        banned_names.update(vocab.resolve(item))

    found: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for hit in hits:
        for rid in hit.passage.meta.get("recipe_ids") or []:
            if rid in seen:
                continue
            seen.add(rid)
            present = ings.get(rid, set())
            offending = [n for n in present
                         if n in banned_names
                         or vocab.category(n) in banned_families]
            if offending:
                found.append((rid, sorted(offending)))
    return found


def constraint_respected(hits: list[Scored], constraints,
                         ings: dict[str, set[str]]) -> float | None:
    """Fraction of distinct recipes in the shortlist that satisfy the question.

    None when the question restricts nothing, so unconstrained queries do not
    dilute the score with free passes.
    """
    if constraints is None or not constraints.restricting:
        return None
    recipes = {rid for hit in hits
               for rid in (hit.passage.meta.get("recipe_ids") or [])}
    if not recipes:
        return 1.0        # nothing returned cannot violate anything
    bad = {rid for rid, _ in violations(hits, constraints, ings)}
    return (len(recipes) - len(bad)) / len(recipes)


def collapse_to_recipes(hits: list[Scored],
                        fingerprints: dict[str, Fingerprint]) -> list[str]:
    """Ordered recipe ids, best rank first, duplicates removed."""
    seen: list[str] = []
    for hit in hits:
        for rid, fp in fingerprints.items():
            if fp.usable and matches(hit.passage.text, fp) and rid not in seen:
                seen.append(rid)
    return seen


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Binary: did any correct recipe reach the shortlist."""
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Proportional: how well the k slots were used.

    The denominator is capped at k, because with 21 correct answers and 5 slots
    filling all 5 correctly is already a perfect result.
    """
    denominator = min(len(relevant), k)
    if not denominator:
        return 0.0
    return len(set(ranked[:k]) & relevant) / denominator


def mean_average_precision(ranked: list[str], relevant: set[str], k: int) -> float:
    hits = 0
    total = 0.0
    for position, rid in enumerate(ranked[:k], start=1):
        if rid in relevant:
            hits += 1
            total += hits / position
    denominator = min(len(relevant), k)
    return total / denominator if denominator else 0.0


def diversity_at_k(hits: list[Scored], k: int) -> float:
    """Distinct dishes in the shortlist, over the most it could hold.

    Counted by each chunk's PRIMARY recipe -- the first span it touches -- not by
    every recipe it overlaps. Summing all overlaps let a chunk straddling two
    recipes contribute two to the numerator and one to the denominator, so the
    score could exceed 1.0 and diversity read as better than it was.
    """
    if not hits:
        return 0.0
    window = hits[:k]
    primary = {ids[0] for hit in window
               if (ids := hit.passage.meta.get("recipe_ids") or [])}
    return len(primary) / len(window)


@traced("stage3.evaluate")
def evaluate(retriever, corpus: Corpus, query_transform=None, k: int = 5,
             limit: int | None = 20, families: set[str] | None = None,
             extractor=None, check_constraints: bool = True,
             progress=None) -> dict[str, Any]:
    golden = load_golden()
    fingerprints = build_fingerprints(golden, corpus)
    known = {r.recipe_id for r in golden}

    questions = [q for q in load_questions()
                 if (families is None or q["family"] in families)
                 and set(q["relevant_ids"]) & known]
    if limit is not None and len(questions) > limit:
        step = len(questions) / limit
        questions = [questions[int(i * step)] for i in range(limit)]

    ings: dict[str, set[str]] = {}
    if check_constraints:
        try:
            ings = recipe_ingredients(corpus.source)
            if extractor is None:
                from ..graph.constraints import ConstraintExtractor

                extractor = ConstraintExtractor()
        except Exception:
            extractor = None

    per_family: dict[str, list[tuple[float, float, float]]] = {}
    respected: list[float] = []
    breaches: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    ledger = Ledger()

    for question in questions:
        with ledger.timed("query_transform"):
            queries = ([question["question"]] if query_transform is None
                       else query_transform.expand(question["question"]))
        with ledger.timed("retrieve"):
            hits = (retriever.search_multi(queries, k)
                    if hasattr(retriever, "search_multi") and len(queries) > 1
                    else retriever.search(queries[0], k))
        ranked = collapse_to_recipes(hits, fingerprints)
        relevant = set(question["relevant_ids"]) & known

        h = hit_at_k(ranked, relevant, k)
        r = recall_at_k(ranked, relevant, k)
        m = mean_average_precision(ranked, relevant, k)
        d = diversity_at_k(hits, k)
        per_family.setdefault(question["family"], []).append((h, r, m, d))

        cr = None
        if extractor is not None:
            try:
                with ledger.timed("constraint_check"):
                    constraints = extractor.extract(question["question"])
                cr = constraint_respected(hits, constraints, ings)
                if cr is not None:
                    respected.append(cr)
                    if cr < 1.0:
                        for rid, offending in violations(hits, constraints, ings):
                            breaches.append({
                                "query_id": question["query_id"],
                                "question": question["question"][:60],
                                "recipe_id": rid, "offending": offending,
                                "banned": constraints.exclude_categories
                                          + constraints.exclude})
            except Exception:
                cr = None

        rows.append({"query_id": question["query_id"], "family": question["family"],
                     "question": question["question"][:70],
                     "hit": h, "recall": round(r, 3), "map": round(m, 4),
                     "diversity": round(d, 3),
                     "constraint_respected": None if cr is None else round(cr, 3),
                     "n_queries": len(queries),
                     "n_relevant": len(relevant), "top_recipes": ranked[:3]})
        if progress is not None:
            progress(len(rows), len(questions), question["question"],
                     family=question["family"], hit=h,
                     recall=round(r, 2),
                     constraint=None if cr is None else round(cr, 2))

    flat = [v for values in per_family.values() for v in values]
    return {
        "retriever": getattr(retriever, "name", type(retriever).__name__),
        "query_transform": getattr(query_transform, "name", "passthrough"),
        "k": k,
        "questions": len(questions),
        "hit_at_k": round(statistics.fmean(v[0] for v in flat), 4) if flat else None,
        "recall_at_k": round(statistics.fmean(v[1] for v in flat), 4) if flat else None,
        "map": round(statistics.fmean(v[2] for v in flat), 4) if flat else None,
        "diversity_at_k": round(statistics.fmean(v[3] for v in flat), 4) if flat else None,
        "constraint_respected": (round(statistics.fmean(respected), 4)
                                 if respected else None),
        "constrained_questions": len(respected),
        "breaches": breaches,
        "by_family": {name: {
            "n": len(values),
            "hit": round(statistics.fmean(v[0] for v in values), 3),
            "recall": round(statistics.fmean(v[1] for v in values), 3),
            "map": round(statistics.fmean(v[2] for v in values), 3),
            "diversity": round(statistics.fmean(v[3] for v in values), 3),
        } for name, values in sorted(per_family.items())},
        "elapsed_s": round(time.perf_counter() - started, 2),
        "telemetry": ledger.as_dict(),
        "rows": rows,
    }