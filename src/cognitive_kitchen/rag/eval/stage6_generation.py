"""Stage 6 - generation quality.

Two metrics computed here, two delegated to a judge.

    NoInventedIngredients   computed. Every ingredient the answer names must
                            appear in the context it was given.
    HonestAbstention        computed. When the context cannot answer the
                            question, the answer must say so rather than guess.
    ConstraintRespected     computed. When the question ruled something out, the
                            answer must not recommend it.
    Faithfulness            judged. Is every claim supported by the context?
    AnswerRelevancy         judged. Did it answer THE question, or a neighbour?
    Cookable                judged. Could someone actually follow this?

NoInventedIngredients leans on the ingredient vocabulary, and that is the whole
reason the vocabulary exists. A naive string check flags "ginger" as invented
when the context said "1 inch fresh ginger, chopped"; vocab.same_ingredient
knows they are the same thing, and equally that coriander seeds and coriander
leaves are not.

HonestAbstention is a floor rather than a ranking. All four strategies are
grounded, so all four should refuse correctly; a score below 1.0 is a defect to
fix, not a strategy to reject.
"""
from __future__ import annotations

import re
import statistics
import time
from typing import Any

from ..telemetry import Ledger, traced
from ..types import Answer, Corpus, Passage
from ..vocab import ingredients as vocab
from .golden import load_questions

# Lines that look like an ingredient listing rather than prose.
BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*(.+)$", re.MULTILINE)


def answer_ingredients(text: str) -> list[str]:
    """Ingredient names the answer appears to claim.

    Only bulleted or numbered lines are read, and only up to a STEPS heading if
    there is one. Prose is skipped on purpose: "fry until the onions soften" is
    a method mentioning an ingredient, not a claim that the recipe contains it,
    and counting it would manufacture violations.
    """
    head = re.split(r"\bSTEPS?\b\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    found: list[str] = []
    for line in BULLET.findall(head):
        for name in vocab.resolve(line):
            if name and name not in found:
                found.append(name)
    return found


def context_ingredients(contexts: list[Passage]) -> set[str]:
    names: set[str] = set()
    for passage in contexts:
        for line in passage.text.splitlines():
            line = line.strip(" -*\u2022")
            if line:
                names.update(vocab.resolve(line))
    return names


def no_invented_ingredients(answer: Answer) -> tuple[float, list[str]]:
    """1.0 when every ingredient the answer names is present in its context."""
    claimed = answer_ingredients(answer.text)
    if not claimed:
        return 1.0, []          # nothing claimed cannot be invented
    available = context_ingredients(answer.contexts)
    invented = [name for name in claimed
                if not any(vocab.same_ingredient(name, have) for have in available)]
    return (len(claimed) - len(invented)) / len(claimed), invented


def honest_abstention(answer: Answer, answerable: bool | None) -> float | None:
    """Did it refuse exactly when it should have?

    `answerable` must mean "the retrieved context actually contains the answer",
    not "something was retrieved". Those are different, and conflating them
    inverts the metric: asked "what can I cook with freshly?" -- a golden query
    whose subject is not an ingredient -- the retriever still returns five
    chunks, and a model that correctly says it cannot answer was being scored
    zero for it.

    None when the question's answerability cannot be established, so it is left
    out rather than guessed at.
    """
    if answerable is None:
        return None
    if answerable:
        return 0.0 if answer.refused else 1.0
    return 1.0 if answer.refused else 0.0


def constraint_respected(answer: Answer, constraints,
                         ings: dict[str, set[str]]) -> float | None:
    """Does the answer avoid recommending what the question ruled out?"""
    if constraints is None or not constraints.restricting:
        return None
    banned_families = set(constraints.exclude_categories or ())
    banned_names: set[str] = set()
    for item in constraints.exclude or ():
        banned_names.update(vocab.resolve(item))

    claimed = answer_ingredients(answer.text)
    if not claimed:
        return 1.0
    offending = [n for n in claimed
                 if n in banned_names or vocab.category(n) in banned_families]
    return (len(claimed) - len(offending)) / len(claimed)


@traced("stage6.evaluate")
def evaluate(strategy, retriever, corpus: Corpus, query_transform=None,
             k: int = 5, limit: int = 5, judge=None,
             families: set[str] | None = None, progress=None) -> dict[str, Any]:
    """Run a strategy over a small sample and score the answers."""
    from .stage3_ranking import recipe_ingredients

    ings = recipe_ingredients(corpus.source)

    # Which pipeline recipes each golden question considers correct. This is the
    # only honest way to know whether retrieved context can answer a question.
    from .atoms import build_fingerprints, matches
    from .golden import load_golden

    golden = load_golden()
    fingerprints = build_fingerprints(golden, corpus)
    gold_to_pipe: dict[str, list[str]] = {}
    for gid, fingerprint in fingerprints.items():
        if fingerprint.usable:
            gold_to_pipe[gid] = [s.recipe_id for s in corpus.spans
                                 if matches(corpus.text[s.start:s.end], fingerprint)]

    extractor = None
    try:
        from ..graph.constraints import ConstraintExtractor

        extractor = ConstraintExtractor()
    except Exception:
        pass

    from .golden import questions_for_eval

    # Nothing here reads relevant_ids, so this stage works unchanged on a plain
    # question list. Only the family filter has to let "user" through.
    pool = questions_for_eval()
    labelled = any(q.get("relevant_ids") for q in pool)
    questions = [q for q in pool
                 if families is None or not labelled or q["family"] in families]
    if limit and len(questions) > limit:
        step = len(questions) / limit
        questions = [questions[int(i * step)] for i in range(limit)]

    ledger = Ledger()
    rows: list[dict[str, Any]] = []
    contexts_per_row: list[list[str]] = []
    started = time.perf_counter()

    for question in questions:
        text = question["question"]
        with ledger.timed("retrieve"):
            queries = ([text] if query_transform is None
                       else query_transform.expand(text))
            hits = (retriever.search_multi(queries, k)
                    if hasattr(retriever, "search_multi") and len(queries) > 1
                    else retriever.search(queries[0], k))
        contexts = [hit.passage for hit in hits]

        if progress is not None:
            progress(len(rows) + 1, len(questions), text,
                     stage="retrieved", contexts=len(contexts))

        with ledger.timed("generate"):
            answer = strategy.generate(text, contexts)

        # answerable when a retrieved chunk belongs to a recipe golden calls correct
        expected: set[str] = set()
        for gid in question.get("relevant_ids") or []:
            expected.update(gold_to_pipe.get(gid, []))
        retrieved = {rid for p in contexts
                     for rid in (p.meta.get("recipe_ids") or [])}
        answerable = None if not expected else bool(expected & retrieved)

        invented_score, invented = no_invented_ingredients(answer)
        constraints = None
        if extractor is not None:
            try:
                constraints = extractor.extract(text)
            except Exception:
                constraints = None

        contexts_per_row.append([p.text for p in contexts])
        rows.append({
            "query_id": question["query_id"], "family": question["family"],
            "question": text[:60],
            "answer": answer.text[:400],
            "refused": answer.refused,
            "n_contexts": len(contexts),
            "no_invented": round(invented_score, 3),
            "invented": invented,
            "answerable": answerable,
            "abstention": honest_abstention(answer, answerable),
            "constraint_respected": constraint_respected(answer, constraints, ings),
            "elapsed_s": answer.elapsed_s,
        })
        if progress is not None:
            progress(len(rows), len(questions), text, stage="answered",
                     seconds=answer.elapsed_s, refused=answer.refused,
                     invented=round(invented_score, 2))

    def mean(key: str) -> float | None:
        values = [r[key] for r in rows if r[key] is not None]
        return round(statistics.fmean(values), 4) if values else None

    result = {
        "stage": "6-generation",
        "strategy": getattr(strategy, "name", type(strategy).__name__),
        "questions": len(rows),
        "no_invented_ingredients": mean("no_invented"),
        "honest_abstention": mean("abstention"),
        "constraint_respected": mean("constraint_respected"),
        "refusal_rate": round(statistics.fmean(
            1.0 if r["refused"] else 0.0 for r in rows), 3) if rows else None,
        "mean_answer_seconds": round(statistics.fmean(
            r["elapsed_s"] for r in rows), 1) if rows else None,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "telemetry": ledger.as_dict(),
        "rows": rows,
    }

    if judge is not None:
        if progress is not None:
            progress(len(rows), len(rows), "judging faithfulness and cookability",
                     stage="judge")
        # the judge needs the retrieved text, which the rows do not carry
        contexts_by_query = {r["query_id"]: c for r, c in
                             zip(rows, contexts_per_row)}
        result.update(judge.score(rows, contexts_by_query, progress=progress))
    return result