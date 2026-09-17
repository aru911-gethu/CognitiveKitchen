"""Graph as a retrieval pre-filter: the graph decides membership, a retriever
decides order.

This is the eighth entry in the Stage 3 dropdown, scored with the same four
metrics as the other seven so the comparison is honest. It exists because two
query families sat at 0.000 for every one of them:

    constraint   "a rice dish I can make without dairy"
    exclusion    "something with no nuts"

No amount of reranking fixes that. Absence is not a direction in embedding
space, so a dense retriever asked for "without dairy" returns the most
dairy-heavy rice dish in the book -- the words match beautifully. The graph
answers it as a pattern that either holds or does not, then the base retriever
ranks whatever survived.

When a question carries no restriction the graph has nothing to add, so this
delegates straight to the base retriever and costs nothing extra. That keeps the
comparison fair rather than flattering.
"""
from __future__ import annotations

import time

from ..registry import build, discover, register
from ..types import Passage, Scored
from ._base import MultiQueryMixin


class GraphHybridRetriever(MultiQueryMixin):
    def __init__(self, base: str = "rrf", embedder=None, pool: int = 0,
                 extractor=None, model: str = "gpt4o-mini") -> None:
        self.name = f"graph_hybrid({base})"
        self.params = {"base": base, "pool": pool, "model": model}
        discover("cognitive_kitchen.rag.retrieval")
        kwargs = {"embedder": embedder} if base in (
            "dense", "hybrid", "rrf", "mmr", "cross_encoder") else {}
        self.base = build("retriever", base, **kwargs)
        self._passages: list[Passage] = []
        self._source_prefix = ""
        self._extractor = extractor
        self._model = model
        # what happened, for the metrics table
        self.trace: list[dict] = []
        self.graph_seconds = 0.0

    # -- lazy, so importing this costs no database and no API key ---------
    def extractor(self):
        if self._extractor is None:
            from ..graph.constraints import ConstraintExtractor

            self._extractor = ConstraintExtractor(model=self._model)
        return self._extractor

    def index(self, passages: list[Passage]) -> None:
        self.base.index(passages)
        self._passages = list(passages)
        sources = {p.meta.get("source") or "" for p in passages}
        self._source_prefix = sources.pop() if len(sources) == 1 else ""
        self.params["corpus_source"] = self._source_prefix or "(mixed)"

    def _allowed_recipe_ids(self, constraints) -> set[str]:
        """Bare recipe ids the graph permits, restricted to the indexed corpus.

        The graph spans every ingestion run and ids restart at r_0001 in each
        one, so dropping the "<source>:" prefix merges the PDF's r_0002 with the
        website's r_0002 -- the same collision the graph guards against
        internally, reintroduced at the gate. It let a glossary page inherit a
        web recipe's eligibility and take the top three slots ahead of the real
        answers. Filter to this corpus first, then drop the prefix.
        """
        from ..graph import traverse

        keys = traverse.filter_recipes(
            include=constraints.include,
            exclude=constraints.exclude,
            exclude_categories=constraints.exclude_categories,
            course=constraints.course)
        if self._source_prefix:
            keys = [k for k in keys
                    if k.split(":", 1)[0].startswith(self._source_prefix)]
        return {k.split(":", 1)[-1] for k in keys}

    def search(self, query: str, k: int) -> list[Scored]:
        started = time.perf_counter()
        try:
            constraints = self.extractor().extract(query)
        except Exception:
            constraints = None

        if constraints is None or not constraints.restricting:
            # nothing to filter on: behave exactly like the base retriever
            self.trace.append({"query": query, "filtered": False,
                               "reason": "no restriction in question"})
            return self.base.search(query, k)

        allowed = self._allowed_recipe_ids(constraints)
        self.graph_seconds += time.perf_counter() - started

        if not allowed:
            self.trace.append({"query": query, "filtered": True, "allowed": 0,
                               "constraints": constraints.as_dict(),
                               "reason": "no recipe satisfies the constraint"})
            return []

        # Rank the whole corpus, then keep the survivors. Taking the base
        # retriever's top-N first and intersecting afterwards is the same
        # mistake in a different place: a correct answer ranked 200th by
        # similarity never reaches the filter, so the graph gets blamed for a
        # miss it did not cause. The corpus is a few hundred chunks, so asking
        # for all of them costs almost nothing and makes this a genuine
        # pre-filter rather than a post-filter wearing its name.
        pool = max(self.params["pool"], len(self._passages))
        candidates = self.base.search(query, pool)
        kept = [hit for hit in candidates
                if allowed & set(hit.passage.meta.get("recipe_ids") or [])]

        self.trace.append({
            "query": query, "filtered": True, "allowed": len(allowed),
            "constraints": constraints.as_dict(),
            "pool": len(candidates), "survived": len(kept)})

        return [Scored(passage=hit.passage, score=hit.score, rank=i)
                for i, hit in enumerate(kept[:k], start=1)]


@register("retriever", "graph_hybrid")
def make(base: str = "rrf", embedder=None, pool: int = 0,
         extractor=None, model: str = "gpt4o-mini") -> GraphHybridRetriever:
    return GraphHybridRetriever(base, embedder, pool, extractor, model)