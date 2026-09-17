"""Stage 5 - the graph on its own, with no retriever underneath it.

Scored differently from Stage 3 on purpose. Retrieval returns a ranked
shortlist, so MAP and Hit@k mean something. A traversal returns a SET: every
recipe satisfying the condition, in no particular order, complete. Ranking
metrics have nothing to measure there, and using them would reward the graph for
an ordering it never claimed to produce.

So:

    set_precision         of the returned recipes the truth set has an opinion
                          about, how many belong
    set_recall            of the recipes that belong, how many came back
    set_f1                the two together

Precision is scored only over the recipes the golden dataset covers, and that
qualifier is essential rather than pedantic. Golden holds 50 of the corpus's 184
recipes. Asked "what can I cook with cumin" the graph returns 168 recipes that
really do contain cumin, and golden names 23; scoring the other 145 as false
positives would put a correct exhaustive answer at precision 0.14 and cap a
perfect one near 0.27. That measures the truth set's coverage, not the graph.
    constraint_extracted  did the LLM read the question correctly
    constraint_respected  does every recipe returned actually satisfy it

The last two separate the two ways this can fail. A wrong answer is either a
misread question or a bad traversal, and the fix differs: reword the prompt, or
fix the graph.
"""
from __future__ import annotations

import statistics
import time
from typing import Any

from ..graph import traverse
from ..telemetry import Ledger, traced
from ..types import Corpus
from ..vocab import ingredients as vocab
from .atoms import build_fingerprints, matches
from .golden import load_golden, load_questions

# Questions the graph is built to answer. Everything else needs similarity.
ANSWERABLE = {"exclusion", "by_ingredient"}


def golden_to_pipeline(corpus: Corpus) -> dict[str, list[str]]:
    """Map golden recipe ids onto pipeline recipe ids by content, not by id.

    The two id spaces do not line up: the pipeline is one recipe per page, and a
    golden recipe can span two pages. Matching on fingerprints is the only join
    that survives that.
    """
    golden = load_golden()
    fingerprints = build_fingerprints(golden, corpus)
    out: dict[str, list[str]] = {}
    for gid, fingerprint in fingerprints.items():
        if not fingerprint.usable:
            continue
        out[gid] = [span.recipe_id for span in corpus.spans
                    if matches(corpus.text[span.start:span.end], fingerprint)]
    return out


def set_scores(returned: set[str], expected: set[str],
               judged: set[str]) -> tuple[float, float, float]:
    """Precision over the judged subset, recall over everything expected.

    `judged` is every pipeline recipe the golden dataset covers. Returned
    recipes outside it are neither right nor wrong -- unjudged -- so they are
    left out of precision instead of being counted against it.
    """
    if not expected:
        return (1.0, 1.0, 1.0) if not returned else (0.0, 1.0, 0.0)
    if not returned:
        return 0.0, 0.0, 0.0
    hits = len(returned & expected)
    scoreable = returned & judged
    precision = hits / len(scoreable) if scoreable else 0.0
    recall = hits / len(expected)
    f1 = (0.0 if precision + recall == 0
          else 2 * precision * recall / (precision + recall))
    return precision, recall, f1


def _respected(returned: set[str], constraints,
               ings: dict[str, set[str]]) -> float | None:
    """Fraction of returned recipes that genuinely satisfy the restriction."""
    if not constraints.restricting:
        return None
    if not returned:
        return 1.0
    banned_families = set(constraints.exclude_categories or ())
    banned_names: set[str] = set()
    for item in constraints.exclude or ():
        banned_names.update(vocab.resolve(item))
    clean = 0
    for rid in returned:
        present = ings.get(rid, set())
        if any(n in banned_names or vocab.category(n) in banned_families
               for n in present):
            continue
        clean += 1
    return clean / len(returned)


def _extracted(constraints, question: dict[str, Any]) -> float:
    """Did the extractor find a restriction where the question implies one?

    Coarse by design: the question's family already says whether a restriction
    exists, so this checks agreement rather than trying to grade the contents.
    """
    implies = question["family"] in ("exclusion", "constraint")
    found = constraints.restricting
    return 1.0 if implies == found else 0.0


@traced("stage5.evaluate")
def evaluate(corpus: Corpus, limit: int | None = 20,
             families: set[str] | None = None, extractor=None,
             progress=None) -> dict[str, Any]:
    from ..eval.stage3_ranking import recipe_ingredients
    from ..graph.constraints import ConstraintExtractor

    extractor = extractor or ConstraintExtractor()
    ings = recipe_ingredients(corpus.source)
    mapping = golden_to_pipeline(corpus)
    known = set(mapping)
    judged = {rid for ids in mapping.values() for rid in ids}

    wanted = families or ANSWERABLE
    questions = [q for q in load_questions()
                 if q["family"] in wanted and set(q["relevant_ids"]) & known]
    if limit is not None and len(questions) > limit:
        step = len(questions) / limit
        questions = [questions[int(i * step)] for i in range(limit)]

    ledger = Ledger()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for question in questions:
        with ledger.timed("extract"):
            constraints = extractor.extract(question["question"])

        with ledger.timed("traverse"):
            keys = traverse.filter_recipes(
                include=constraints.include,
                exclude=constraints.exclude,
                exclude_categories=constraints.exclude_categories,
                course=constraints.course)
        prefix = corpus.source
        if prefix:
            keys = [k for k in keys if k.split(":", 1)[0].startswith(prefix)]
        returned = {k.split(":", 1)[-1] for k in keys}

        expected: set[str] = set()
        for gid in question["relevant_ids"]:
            expected.update(mapping.get(gid, []))

        precision, recall, f1 = set_scores(returned, expected, judged)
        rows.append({
            "query_id": question["query_id"], "family": question["family"],
            "question": question["question"][:64],
            "n_returned": len(returned), "n_expected": len(expected),
            "n_judged_returned": len(returned & judged),
            "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3),
            "extracted": _extracted(constraints, question),
            "respected": _respected(returned, constraints, ings),
            "constraints": constraints.as_dict(),
        })
        if progress is not None:
            progress(len(rows), len(questions), question["question"],
                     returned=len(returned), expected=len(expected),
                     precision=round(precision, 2), recall=round(recall, 2))

    def mean(key: str) -> float | None:
        values = [r[key] for r in rows if r[key] is not None]
        return round(statistics.fmean(values), 4) if values else None

    per_family: dict[str, dict[str, Any]] = {}
    for row in rows:
        per_family.setdefault(row["family"], []).append(row)

    return {
        "stage": "5-graph-standalone",
        "questions": len(rows),
        "corpus_recipes": len(corpus.spans),
        "judged_recipes": len(judged),
        "set_precision": mean("precision"),
        "set_recall": mean("recall"),
        "set_f1": mean("f1"),
        "constraint_extracted": mean("extracted"),
        "constraint_respected": mean("respected"),
        "by_family": {name: {
            "n": len(group),
            "precision": round(statistics.fmean(r["precision"] for r in group), 3),
            "recall": round(statistics.fmean(r["recall"] for r in group), 3),
            "f1": round(statistics.fmean(r["f1"] for r in group), 3),
        } for name, group in sorted(per_family.items())},
        "elapsed_s": round(time.perf_counter() - started, 2),
        "cost_usd": getattr(extractor, "cost_usd", 0.0),
        "telemetry": ledger.as_dict(),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# End to end, on the same metrics Stage 6 uses.
# ---------------------------------------------------------------------------
def evaluate_generation(corpus: Corpus, strategy_name: str = "stuff_strict",
                        generator=None, k: int = 5, limit: int = 5,
                        judge=None, progress=None,
                        families: set[str] | None = None) -> dict[str, Any]:
    """Answer from the graph alone, then score exactly as Stage 6 does.

    Set precision and recall say whether the traversal found the right recipes.
    They say nothing about whether an answer built from them is faithful,
    relevant or followable, which is the only question that decides whether this
    route could serve a user. Reusing stage6.evaluate rather than writing a
    parallel scorer is what makes the two routes comparable: same metrics, same
    judge, same thresholds, one table.
    """
    from ..registry import build, discover
    from .stage6_generation import evaluate as evaluate_generation_stage

    discover("cognitive_kitchen.rag.retrieval", "cognitive_kitchen.rag.generate",
             "cognitive_kitchen.rag.strategies")

    retriever = build("retriever", "graph_only", source=corpus.source)
    if generator is None:
        generator = build("generator", "gpt4o-mini")
    strategy = build("strategy", strategy_name, generator=generator)

    result = evaluate_generation_stage(
        strategy, retriever, corpus, query_transform=None, k=k, limit=limit,
        judge=judge, families=families, progress=progress)
    result["route"] = f"graph_only -> {strategy_name}"
    result["stage"] = "5-graph-generation"
    result["trace"] = retriever.trace
    return result


# ---------------------------------------------------------------------------
# Showcase: the questions no retriever can answer at all.
# ---------------------------------------------------------------------------
def showcase(pantry: list[str] | None = None) -> dict[str, Any]:
    pantry = pantry or ["onion", "tomato", "oil", "salt", "turmeric",
                        "chilli powder"]
    stats = traverse.stats()
    return {
        "pantry": {
            "have": pantry,
            "note": ("'missing' is written in no document. It is "
                     "(ingredients needed) minus (ingredients you have)."),
            "results": traverse.pantry_gap(pantry, max_missing=2, limit=10),
        },
        "substitutes": {
            name: traverse.substitutes(name, limit=5)
            for name in ("ghee", "yogurt", "cashew")
        },
        "counts": {
            "real_recipes": stats["real_recipes"],
            "by_allergen": stats["by_category"],
            "dairy_free": len(traverse.without_categories(["dairy"])),
            "gluten_free": len(traverse.without_categories(["gluten"])),
            "nut_free": len(traverse.without_categories(["nut"])),
        },
        "equipment": {
            name: len(traverse.needing_equipment(name))
            for name in ("pressure cooker", "blender", "oven", "tawa")
        },
    }