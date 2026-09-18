"""Golden dataset access.

The only module that opens golden_dataset.json. It is the answer key, so it is
read at scoring time and never by the pipeline that produces answers.

Absence is a supported state, not an error. Requiring a hand-labelled truth set
before anything can be measured is the single biggest barrier to using this on
your own documents, and it is a barrier that only some of the metrics actually
need. The split is between questions and labels, not between golden and nothing:

    no questions        nothing is measurable, there is nothing to ask
    questions only      purity, self-sufficiency, latency, cost, diversity,
                        faithfulness, relevancy, cookability, abstention,
                        and constraint compliance
    questions + labels  adds recall, hit@k, MAP, k@90, set recall

Recall is the reason labels cannot be replaced by an LLM judge. To know a
retriever missed something you have to know what existed; a judge only ever sees
what you showed it. Everything else on that list is computable from the answer
alone, including the constraint metric this project exists to demonstrate.

So `data/questions.txt` -- one question per line -- is enough to score most of
the pipeline on a corpus nobody has labelled.
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


def questions_path() -> Path:
    """Plain question list, used when there is no labelled truth set."""
    return settings.data_dir / "questions.txt"


def golden_available() -> bool:
    return golden_path().exists()


def has_labels() -> bool:
    """True when relevance judgements exist, so recall and MAP are meaningful."""
    return bool(load_questions())


@lru_cache(maxsize=1)
def load_golden() -> tuple[GoldenRecipe, ...]:
    if not golden_path().exists():
        return ()
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
    """Labelled questions. Empty when there is no golden dataset."""
    if not golden_path().exists():
        return ()
    raw = json.loads(golden_path().read_text(encoding="utf-8"))
    return tuple(raw["evaluation"]["queries"])


@lru_cache(maxsize=1)
def load_user_questions() -> tuple[dict, ...]:
    """data/questions.txt, one per line, shaped like a golden query.

    `relevant_ids` is empty by construction: nobody has said which recipes are
    correct. Evaluators read that as "do not score the label-dependent metrics"
    rather than as "no recipe is relevant", which would report a false zero.
    """
    path = questions_path()
    if not path.exists():
        return ()
    out = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        out.append({"query_id": f"u_{index:04d}", "question": text,
                    "family": "user", "relevant_ids": []})
    return tuple(out)


def questions_for_eval() -> tuple[dict, ...]:
    """Whatever we have to ask: labelled if possible, otherwise the plain list."""
    return load_questions() or load_user_questions()