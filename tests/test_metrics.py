"""Regression tests for the logic that has already been wrong once.

Every case here corresponds to a bug that shipped and was found by inspecting
output rather than by a test. That is the wrong order, so they are pinned.

No network, no database, no model. These run in under a second so there is no
excuse for skipping them.
"""
from __future__ import annotations

import pytest

from cognitive_kitchen.rag.types import Passage


# ---------------------------------------------------------------- vocabulary
class TestVocabulary:
    """Ingredient identity. Wrong answers here corrupt the graph's node set."""

    def test_form_that_changes_identity_is_kept_apart(self):
        from cognitive_kitchen.rag.vocab import ingredients as vocab

        # gpt-4o-mini collapsed all three onto "coriander". The seed is a spice
        # and the leaf is a herb; recipes use both and they are not swappable.
        assert not vocab.same_ingredient("coriander seeds", "coriander leaves")
        assert not vocab.same_ingredient("cumin seeds", "cumin powder")

    def test_form_that_only_changes_shape_is_merged(self):
        from cognitive_kitchen.rag.vocab import ingredients as vocab

        assert vocab.same_ingredient("ginger", "1 inch fresh ginger, chopped")
        assert vocab.same_ingredient("onion", "1 cup chopped onions")
        assert vocab.same_ingredient("turmeric", "1/2 tsp turmeric powder")
        assert vocab.same_ingredient("ghee", "2 tbsp ghee, melted")

    def test_ghee_and_butter_are_different_ingredients(self):
        from cognitive_kitchen.rag.vocab import ingredients as vocab

        assert not vocab.same_ingredient("ghee", "butter")

    def test_clarified_butter_is_ghee(self):
        from cognitive_kitchen.rag.vocab import ingredients as vocab

        assert vocab.same_ingredient("ghee", "clarified butter")

    @pytest.mark.parametrize("name,expected", [
        ("ghee", "dairy"), ("paneer", "dairy"), ("wheat flour", "gluten"),
        ("semolina", "gluten"), ("cashew", "nut"), ("chicken", "meat"),
        ("prawn", "fish"), ("egg", "egg"),
        # these were all filed wrongly by the model at least once
        ("besan", "none"), ("gram flour", "none"), ("rice flour", "none"),
        ("coconut milk", "none"), ("sunflower seeds", "none"),
    ])
    def test_allergen_family(self, name, expected):
        from cognitive_kitchen.rag.vocab import ingredients as vocab

        assert vocab.category(name) == expected, (
            f"{name} filed as {vocab.category(name)}, not {expected}. A wrong "
            f"allergen is a safety bug, not a ranking problem.")

    def test_gram_is_a_food_word_not_only_a_unit(self):
        from cognitive_kitchen.rag.vocab.normalise import normalise

        # "grams?" is in the unit list, and digits are stripped before units, so
        # a bare "gram flour" lost its first word and became wheat flour.
        assert normalise("2 tbsp gram flour") == "gram flour"
        assert normalise("100 grams paneer") == "paneer"

    def test_cloves_stays_a_unit_of_garlic(self):
        from cognitive_kitchen.rag.vocab.normalise import normalise

        # Guarding "cloves" for the spice fragmented a clean "garlic" into six
        # variants. In this corpus the unit is the common case and wins.
        assert normalise("2 cloves garlic") == "garlic"
        assert normalise("4 cloves of garlic") == "garlic"

    def test_rules_never_strip_an_identity_bearing_word(self):
        from cognitive_kitchen.rag.vocab.normalise import IDENTITY_BEARING, PREP

        overlap = {w for w in IDENTITY_BEARING if w in PREP.split("|")}
        assert not overlap, f"tier 1 would undo itself on {sorted(overlap)}"


# ------------------------------------------------------------------- metrics
class TestStage3Metrics:
    def _passage(self, pid: str, recipes: list[str]) -> Passage:
        return Passage(passage_id=pid, text="x", start=0, end=1,
                       meta={"recipe_ids": recipes, "source": "pdf"})

    def test_diversity_cannot_exceed_one(self):
        from cognitive_kitchen.rag.eval.stage3_ranking import diversity_at_k
        from cognitive_kitchen.rag.types import Scored

        # A chunk spanning two recipes used to add two to the numerator and one
        # to the denominator, so diversity reported 1.02.
        hits = [Scored(passage=self._passage("a", ["r1", "r2"]), score=1.0, rank=1),
                Scored(passage=self._passage("b", ["r3", "r4"]), score=0.9, rank=2)]
        assert diversity_at_k(hits, 2) <= 1.0

    def test_diversity_counts_distinct_dishes(self):
        from cognitive_kitchen.rag.eval.stage3_ranking import diversity_at_k
        from cognitive_kitchen.rag.types import Scored

        same = [Scored(passage=self._passage(str(i), ["r1"]), score=1.0, rank=i)
                for i in range(1, 5)]
        assert diversity_at_k(same, 4) == pytest.approx(0.25)

    def test_recall_denominator_is_capped_at_k(self):
        from cognitive_kitchen.rag.eval.stage3_ranking import recall_at_k

        # 21 correct answers and 5 slots: 5 right is a perfect result.
        assert recall_at_k([f"r{i}" for i in range(5)],
                           {f"r{i}" for i in range(21)}, 5) == 1.0

    def test_map_rewards_ranking_correct_answers_early(self):
        from cognitive_kitchen.rag.eval.stage3_ranking import mean_average_precision

        relevant = {"r1", "r2"}
        early = mean_average_precision(["r1", "r2", "x", "y", "z"], relevant, 5)
        late = mean_average_precision(["x", "y", "z", "r1", "r2"], relevant, 5)
        assert early > late


class TestStage5Metrics:
    def test_precision_ignores_recipes_the_truth_set_does_not_judge(self):
        from cognitive_kitchen.rag.eval.stage5_graph import set_scores

        # Golden covers 52 of 184 recipes. Counting the other 132 as false
        # positives measured the truth set's coverage, not the graph: a correct
        # exhaustive answer scored 0.14.
        returned = {f"r{i}" for i in range(100)}
        expected = {"r1", "r2", "r3"}
        judged = {"r1", "r2", "r3", "r4"}
        precision, recall, _ = set_scores(returned, expected, judged)
        assert precision == pytest.approx(0.75)     # 3 of the 4 it may judge
        assert recall == 1.0

    def test_returning_nothing_scores_zero_not_one(self):
        from cognitive_kitchen.rag.eval.stage5_graph import set_scores

        assert set_scores(set(), {"r1"}, {"r1"}) == (0.0, 0.0, 0.0)


class TestStage6Metrics:
    def _answer(self, text: str, refused: bool = False, contexts=()):
        from cognitive_kitchen.rag.types import Answer

        return Answer(question="q", text=text, contexts=list(contexts),
                      refused=refused)

    def test_abstention_is_undefined_when_answerability_is_unknown(self):
        from cognitive_kitchen.rag.eval.stage6_generation import honest_abstention

        assert honest_abstention(self._answer("x"), None) is None

    def test_refusing_an_unanswerable_question_is_correct(self):
        from cognitive_kitchen.rag.eval.stage6_generation import honest_abstention

        # This used bool(contexts) as the proxy for answerable. Retrieval always
        # returns something, so a model that correctly declined scored zero.
        assert honest_abstention(self._answer("NOT_IN_CONTEXT", refused=True),
                                 answerable=False) == 1.0
        assert honest_abstention(self._answer("here you go"),
                                 answerable=False) == 0.0

    def test_answering_an_answerable_question_is_correct(self):
        from cognitive_kitchen.rag.eval.stage6_generation import honest_abstention

        assert honest_abstention(self._answer("here you go"),
                                 answerable=True) == 1.0
        assert honest_abstention(self._answer("NOT_IN_CONTEXT", refused=True),
                                 answerable=True) == 0.0

    def test_prose_mentions_are_not_read_as_ingredient_claims(self):
        from cognitive_kitchen.rag.eval.stage6_generation import answer_ingredients

        # "fry until the onions soften" is a method, not a claim that the recipe
        # contains onions. Counting prose would manufacture violations.
        assert answer_ingredients("Fry until the onions soften.") == []

    def test_bulleted_ingredients_are_read(self):
        from cognitive_kitchen.rag.eval.stage6_generation import answer_ingredients

        found = answer_ingredients("INGREDIENTS:\n- 2 tbsp ghee\n- 1 onion\n"
                                   "STEPS:\n1. add cashew nuts")
        assert "ghee" in found and "onion" in found
        # anything after STEPS is method, not a claim
        assert "cashew" not in found

    def test_nothing_claimed_cannot_be_invented(self):
        from cognitive_kitchen.rag.eval.stage6_generation import (
            no_invented_ingredients)

        score, invented = no_invented_ingredients(self._answer("NOT_IN_CONTEXT",
                                                              refused=True))
        assert score == 1.0 and invented == []


# --------------------------------------------------------------- composition
class TestPipelineComposition:
    @pytest.mark.parametrize("kwargs,expected", [
        ({}, ("rrf", {"sparse": "bm25"})),
        ({"source": "bm25"}, ("bm25", {})),
        ({"use_graph": True}, ("graph_hybrid", {"base": "rrf"})),
        ({"use_rerank": True}, ("cross_encoder", {"first_stage": "rrf"})),
        ({"source": "dense", "use_mmr": True}, ("mmr", {"base": "dense"})),
    ])
    def test_controls_compose_into_one_retriever(self, kwargs, expected):
        from cognitive_kitchen.rag.pipeline import Pipeline

        assert Pipeline(**kwargs).retriever_spec() == expected

    def test_graph_wraps_the_reranker_not_the_other_way_round(self):
        from cognitive_kitchen.rag.pipeline import Pipeline

        # A gate applied after reranking is still a gate; a reranker applied
        # after a gate can only reorder what survived. Graph must be outermost.
        name, params = Pipeline(use_rerank=True, use_graph=True).retriever_spec()
        assert name == "graph_hybrid"
        assert params["base"] == "cross_encoder"

    def test_mmr_is_not_hardwired_to_dense(self):
        from cognitive_kitchen.rag.pipeline import Pipeline

        # Diversity used to be available only on top of the weakest source, so
        # "does diversity help my best retriever" was unaskable.
        _, params = Pipeline(source="rrf", use_mmr=True).retriever_spec()
        assert params["base"] == "rrf"


class TestGraphKeys:
    def test_recipe_ids_are_qualified_by_source(self):
        from cognitive_kitchen.rag.graph.schema import load_recipes, shape

        # recipe_id restarts at r_0001 in every ingestion run, so the PDF's
        # r_0002 and the website's r_0002 are different recipes. Merging them let
        # a glossary page inherit a web recipe's eligibility.
        data = shape(load_recipes())
        keys = [r["key"] for r in data.recipes]
        assert len(keys) == len(set(keys)), "graph keys collide across sources"
        assert all(":" in key for key in keys)

    def test_pages_without_ingredients_are_flagged_not_dropped(self):
        from cognitive_kitchen.rag.graph.schema import load_recipes, shape

        # Section headers and continuation pages hold real step text, so they
        # stay -- but an exclusion query must not count them as answers, since a
        # page with no ingredients trivially satisfies every restriction.
        data = shape(load_recipes())
        for recipe in data.recipes:
            if recipe["n_ingredients"] < 2:
                assert recipe["is_recipe"] is False