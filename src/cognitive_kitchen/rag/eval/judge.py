"""Judged generation metrics, all three from DeepEval.

The computed metrics in stage6 are a floor: every strategy here is grounded, so
NoInventedIngredients and HonestAbstention come out at 1.000 for all of them and
rank nothing. These three discriminate.

    Faithfulness      FaithfulnessMetric. Splits the answer into claims and
                      checks each against the retrieved context. Catches what a
                      vocabulary check cannot -- a quantity or a step stated
                      confidently and absent from the source.

    AnswerRelevancy   AnswerRelevancyMetric. Did it answer THE question, or a
                      neighbouring one? A perfectly faithful recipe for the
                      wrong dish scores well on faithfulness and is useless.

    Cookable          GEval with explicit steps and a rubric. Could a cook
                      follow this and get the dish?

Cookable was a hand-written prompt asking for a score in JSON. GEval is the
right home for it: it reasons through named steps before scoring rather than
producing a number in one shot, the rubric is stated rather than implied, and
the parsing is the library's problem. Nothing about judging "is this followable"
was specific enough to justify hand-rolling it.

All three run on gpt-4o-mini and cache to disk by (metric, question, answer), so
re-scoring an unchanged run costs nothing.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from ...config import settings
from ..telemetry import Ledger

COOKABLE_CRITERIA = (
    "Judge whether a competent home cook could follow the answer and end up "
    "with the dish. Judge usability only, not truthfulness -- another metric "
    "covers whether the answer matches its source."
)

COOKABLE_STEPS = [
    "Check whether the answer names ingredients, and whether quantities are "
    "given for the ones that need them. A recipe that says 'add spices' without "
    "saying which is not followable.",
    "Check whether there are steps, and whether their order makes sense: "
    "tempering before the tempered ingredients exist is a fault.",
    "Check for a missing stage a cook would notice -- nothing is soaked, "
    "nothing is cooked, or the dish is never assembled.",
    "Ignore whether the dish is a good answer to the question, and ignore "
    "whether it is faithful to any source. Score only followability.",
]

# GEval wants Rubric objects with a score band each, not a prose scale. Passing
# a string fails inside the metric with "'str' object has no attribute
# score_range", which is not an obvious message for the mistake.
COOKABLE_RUBRIC = [
    (0, 2, "No usable instruction: no ingredients, or no steps, or it declines "
           "to answer."),
    (3, 5, "Names the dish and some ingredients but a cook could not follow "
           "it: no quantities, or no method."),
    (6, 8, "Usable but thin: one quantity missing, or one obvious step absent."),
    (9, 10, "Ingredients with quantities, steps in a sensible order, nothing "
            "obviously missing."),
]


def cache_path() -> Path:
    return settings.data_dir / "eval" / "judged.json"


def _load() -> dict[str, Any]:
    path = cache_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save(cache: dict[str, Any]) -> None:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False,
                               sort_keys=True), encoding="utf-8")


def _key(metric: str, question: str, answer: str) -> str:
    blob = f"{metric}\x00{question}\x00{answer}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:20]


class Judge:
    """Scores the rows produced by stage6_generation.evaluate()."""

    def __init__(self, model: str | None = None, use_cache: bool = True,
                 threshold: float = 0.7) -> None:
        self.model = model or settings.judge_model
        self.use_cache = use_cache
        self.threshold = threshold
        self.ledger = Ledger()
        self.calls = 0
        self.cache_hits = 0
        self.failures: list[str] = []
        self._metrics: dict[str, Any] | None = None

    # -- metric construction, one at a time ------------------------------
    def metric(self, name: str):
        """Build lazily and independently.

        Building all three into one dict meant a single bad argument took out
        every metric with the same misleading error, and the failure looked like
        an API outage rather than a typed parameter. Each is built alone now, so
        one broken metric costs only itself.
        """
        if self._metrics is None:
            self._metrics = {}
        if name in self._metrics:
            return self._metrics[name]

        from deepeval.metrics import (AnswerRelevancyMetric, FaithfulnessMetric,
                                      GEval)
        from deepeval.test_case import LLMTestCaseParams

        if name == "faithfulness":
            built = FaithfulnessMetric(threshold=self.threshold,
                                       model=self.model, include_reason=False)
        elif name == "answer_relevancy":
            built = AnswerRelevancyMetric(threshold=self.threshold,
                                          model=self.model, include_reason=False)
        elif name == "cookable":
            from deepeval.metrics.g_eval import Rubric

            built = GEval(
                name="Cookable",
                criteria=COOKABLE_CRITERIA,
                evaluation_steps=COOKABLE_STEPS,
                rubric=[Rubric(score_range=(low, high), expected_outcome=text)
                        for low, high, text in COOKABLE_RUBRIC],
                evaluation_params=[LLMTestCaseParams.INPUT,
                                   LLMTestCaseParams.ACTUAL_OUTPUT],
                model=self.model, threshold=self.threshold)
        else:
            raise KeyError(name)

        self._metrics[name] = built
        return built

    @property
    def cost_usd(self) -> float:
        """DeepEval bills through its own client, so tokens are not visible here.

        Reported from a measured rate instead of guessed at: roughly $0.0002 for
        the three metrics on one answer at this answer length.
        """
        return round(self.calls * 0.00007, 6)

    def _score(self, name: str, question: str, answer: str,
               contexts: list[str]) -> float | None:
        if not settings.openai_api_key or not answer.strip():
            return None
        try:
            from deepeval.test_case import LLMTestCase

            case = LLMTestCase(input=question, actual_output=answer,
                               retrieval_context=contexts or [""])
            metric = self.metric(name)
            metric.measure(case)
            return None if metric.score is None else float(metric.score)
        except Exception as exc:
            self.failures.append(f"{name}: {type(exc).__name__}: {exc}")
            return None

    # -- entry point -----------------------------------------------------
    def score(self, rows: list[dict[str, Any]],
              contexts_by_query: dict[str, list[str]] | None = None,
              progress=None) -> dict[str, Any]:
        cache = _load() if self.use_cache else {}
        collected: dict[str, list[float]] = {"faithfulness": [],
                                             "answer_relevancy": [],
                                             "cookable": []}
        started = time.perf_counter()

        for position, row in enumerate(rows, start=1):
            question = row.get("question", "")
            answer = row.get("answer", "")
            contexts = (contexts_by_query or {}).get(row.get("query_id", ""), [])
            refused = bool(row.get("refused"))

            for name in collected:
                # A refusal is not an uncookable recipe, it is the absence of
                # one, and it cannot be irrelevant either. Abstention is
                # measured separately and is where a refusal belongs.
                if refused and name in ("cookable", "answer_relevancy"):
                    row[name] = None
                    continue

                key = _key(name, question, answer)
                if self.use_cache and key in cache:
                    self.cache_hits += 1
                    score = cache[key]
                else:
                    score = self._score(name, question, answer, contexts)
                    self.calls += 1
                    if self.use_cache and score is not None:
                        cache[key] = score
                        _save(cache)
                if score is not None:
                    collected[name].append(float(score))
                    row[name] = round(float(score), 3)

            if progress is not None:
                progress(position, len(rows), question, stage="judged")

        self.ledger.record("judge", time.perf_counter() - started)
        result: dict[str, Any] = {
            "judged_questions": len(rows),
            "judge_model": self.model,
            "judge_calls": self.calls,
            "judge_cache_hits": self.cache_hits,
            "judge_cost_usd": self.cost_usd,
            "judge_seconds": round(time.perf_counter() - started, 1),
        }
        for name, values in collected.items():
            result[name] = round(statistics.fmean(values), 4) if values else None
        if self.failures:
            result["judge_failures"] = self.failures[:5]
        return result