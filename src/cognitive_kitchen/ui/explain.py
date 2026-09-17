"""What each option does, what each metric means, and which model is doing it.

Kept as data in one place rather than scattered through the page, so a stage can
render "what am I choosing between" and "how was this number produced" without
the reader having to open the source to find out.
"""
from __future__ import annotations

from ..config import settings

# --------------------------------------------------------------- chunkers
CHUNKERS: dict[str, tuple[str, str]] = {
    "naive": (
        "Fixed windows",
        "Cuts every `size` characters with `overlap` carried over. Ignores "
        "sentences and recipe boundaries entirely, so a cut can land mid-word. "
        "The control: any smarter chunker has to beat this."),
    "recursive": (
        "Recursive character split",
        "Tries paragraph breaks first, then line breaks, then sentence ends, "
        "then spaces, backing off only when a piece is still too big. Respects "
        "the shape the writer put in, which is why it wins purity here."),
    "semantic_adjacent": (
        "Split where neighbours stop matching",
        "Embeds small units and cuts wherever similarity to the next unit dips. "
        "Finds topic changes, but a recipe's ingredient list and its method read "
        "as different topics, so it tends to split dishes in half."),
    "semantic_centroid": (
        "Split on drift from a running average",
        "Keeps a centroid of the current group and cuts when a unit drifts too "
        "far. More stable than neighbour comparison; still no notion of where a "
        "recipe ends."),
    "recursive_semantic_adjacent": (
        "Recursive first, then join on neighbours",
        "Pre-splits structurally, then merges adjacent pieces that look alike. "
        "Higher purity than plain semantic, but fragments more than recursive."),
    "recursive_semantic_centroid": (
        "Recursive first, then join on centroid",
        "Same idea, merging by drift from the group average instead of by "
        "neighbour similarity."),
    "recipe": (
        "One chunk per recipe — uses the boundaries ingestion already found",
        "The six above all INFER where one thing ends and the next begins. "
        "Ingestion already recorded it: title, ingredient lines, step lines and "
        "a page span per recipe. This uses that directly, so nothing is "
        "approximated. Every chunk is a whole dish and carries its own name."),
    "recipe_sections": (
        "Split at section headings, with the title repeated into each piece",
        "Cuts between ingredients and method — a real boundary rather than an "
        "arbitrary one — then prefixes the dish name onto every piece. The "
        "prefix is the point: a bare ingredient list is useless without knowing "
        "which dish it belongs to. Higher purity than one-chunk-per-recipe, "
        "lower recall."),
}

# ------------------------------------------------------------- retrievers
SOURCES: dict[str, tuple[str, str]] = {
    "dense": ("Vectors only",
              "Embeds the question and every chunk, ranks by cosine similarity "
              "over a FAISS index. Finds paraphrases a keyword search misses; "
              "cannot represent 'without'."),
    "bm25": ("Keywords, BM25",
             "Term frequency weighted against how rare each term is, with a "
             "length penalty. Exact on names and quantities, blind to synonyms."),
    "tfidf": ("Keywords, TF-IDF cosine",
              "Simpler than BM25, no length normalisation. Cheap baseline."),
    "hybrid": ("Fusion by score",
               "Min-max normalises the dense and sparse score lists, then takes "
               "a weighted sum. Needs the two scales calibrated against each "
               "other, which is what alpha is for."),
    "rrf": ("Fusion by rank",
            "Sums 1/(c+rank) across the dense and sparse lists. Uses only "
            "position, never the score, so no calibration is needed. That is "
            "the whole appeal over weighted hybrid."),
}

MODIFIERS: dict[str, tuple[str, str]] = {
    "mmr": ("Diversity — Maximal Marginal Relevance",
            "Re-selects from the candidate list, each pick maximising "
            "`lambda*relevance - (1-lambda)*similarity to what is already "
            "chosen`. Stops five slices of one dish crowding out alternatives. "
            "Works on any source."),
    "rerank": ("Cross-encoder rerank",
               "Re-scores the shortlist by putting question and chunk through "
               "the model TOGETHER, so it can weigh them against each other "
               "instead of comparing two separately-made vectors. Much more "
               "accurate, and much slower: one model pass per candidate."),
    "graph": ("Graph pre-filter",
              "Reads the question with an LLM, traverses Neo4j for recipes that "
              "satisfy it, and lets only those through. Decides MEMBERSHIP; the "
              "source still decides order. The only thing here that can answer "
              "'without dairy', because absence is a pattern, not a direction "
              "in embedding space."),
}

# --------------------------------------------------------------- queries
TRANSFORMS: dict[str, tuple[str, str]] = {
    "passthrough": ("Unchanged",
                    "The question goes to the retriever as typed. The control."),
    "decompose": ("Split into parts",
                  "Breaks a compound question into separate retrievals and "
                  "merges the results. 'A rice dish with no dairy' becomes a "
                  "query about rice and a query about dairy, so neither half "
                  "gets diluted by the other."),
    "hyde": ("Hypothetical document embedding",
             "Writes a fake recipe that WOULD answer the question, then "
             "searches using that instead of the question. Answers look like "
             "answers, not like questions. Needs a generation call per query, "
             "which on CPU costs minutes for no measured gain here."),
}

# ------------------------------------------------------------ strategies
STRATEGIES: dict[str, tuple[str, str]] = {
    "stuff_strict": ("Everything, in rank order",
                     "All chunks in one prompt, one call, refuse if the context "
                     "does not answer. The baseline the others are read against."),
    "reordered": ("Best chunks at both ends",
                  "Same chunks, same count, same cost — rank 1,2,3,4,5 becomes "
                  "1,3,5,4,2. Long contexts are read unevenly, so the middle is "
                  "where things get missed. If the score moves, position was "
                  "costing accuracy and no better retrieval would have shown it."),
    "structured": ("Forced sections",
                   "Demands DISH / INGREDIENTS / STEPS / NOTES. A cook wants a "
                   "list, and fixed shape also makes the answer machine-"
                   "checkable, which is how NoInventedIngredients reads it."),
    "map_reduce": ("One call per chunk, then combine",
                   "Each chunk gets the model's full attention with no "
                   "neighbours competing, then the notes are merged. Five chunks "
                   "cost six calls, so the question is whether the isolation is "
                   "worth several times the latency."),
}

# ----------------------------------------------------------------- metrics
METRICS: dict[str, dict[str, tuple[str, str]]] = {
    "stage2": {
        "word_purity": ("words(chunk ∩ recipe) / words(chunk)",
                        "How much of the chunk belongs to one recipe. Low means "
                        "the chunk is polluted with a neighbouring dish."),
        "word_recall_best": ("words(chunk ∩ recipe) / words(recipe)",
                             "How much of a recipe fits in its single best "
                             "chunk. Low means the dish is split up."),
        "k_at_90": ("greedy union of chunks until 90% of the recipe's words",
                    "How many chunks it takes to rebuild the recipe. This is "
                    "what k should be set from, rather than guessing 5."),
        "self_sufficiency": ("1 / rank of its own recipe",
                             "Use the chunk as a search query against all "
                             "recipe cards. If it cannot find its own dish, no "
                             "retriever will find it either."),
    },
    "stage3": {
        "hit@k": ("1 if any correct recipe is in the top k, else 0",
                  "Did it find anything at all. Blind to position and to how "
                  "many were missed."),
        "recall@k": ("correct in top k / min(all correct, k)",
                     "How well the k slots were spent. Capped at k because with "
                     "21 correct answers and 5 slots, 5 right is perfect."),
        "MAP": ("mean of precision at each hit",
                "Rewards putting correct answers early. If hit@k is fine and "
                "MAP is poor, you found them and ranked them badly — that is "
                "what reranking fixes."),
        "diversity": ("distinct primary recipes in top k / k",
                      "Five slices of one dish scores 0.2. Counted by each "
                      "chunk's first recipe so it cannot exceed 1.0."),
        "constraint": ("recipes satisfying the restriction / recipes returned",
                       "SAFETY, not quality — the four above cannot see it. "
                       "Only scored on questions that rule something out. "
                       "Asked 'avoiding nuts', dense returns Nut Milk and the "
                       "cross-encoder returns Nut Milk, Cashew Chutney AND "
                       "Almond Milk: the better the ranker, the more "
                       "confidently wrong."),
    },
    "stage5": {
        "set_precision": ("correct / returned, counted only over judged recipes",
                          "Golden covers 52 of 184 recipes. Recipes it has no "
                          "opinion about are unjudged, not wrong, so they are "
                          "excluded — otherwise a correct exhaustive answer to "
                          "'what has cumin' scores 0.14."),
        "set_recall": ("correct / all correct",
                       "A traversal should never miss a valid answer. This is "
                       "the number that matters for a pre-filter."),
        "constraint_extracted": ("did the LLM read the restriction correctly",
                                 "Separates a misread question from a bad "
                                 "traversal — the fixes differ."),
        "constraint_respected": ("returned recipes that genuinely satisfy it",
                                 "Provable, not approximate: zero overlap "
                                 "between the 92 dairy-free recipes and the 41 "
                                 "that use ghee."),
    },
    "stage6": {
        "NoInvented": ("ingredients in the answer that appear in its context",
                       "Uses the ingredient vocabulary, not string matching, so "
                       "'ginger' is not flagged when the context said '1 inch "
                       "fresh ginger, chopped' — and coriander seeds still do "
                       "not count as coriander leaves."),
        "Abstention": ("refused exactly when the context could not answer",
                       "Answerability comes from the golden dataset, not from "
                       "'did anything get retrieved'. Those differ, and "
                       "conflating them punishes a model for correctly "
                       "declining an unanswerable question."),
        "Relevancy": ("DeepEval AnswerRelevancyMetric",
                      "Did it answer THE question or a neighbouring one. A "
                      "perfectly faithful recipe for the wrong dish scores well "
                      "on faithfulness and is still useless. Blank for "
                      "refusals."),
        "Relevancy": ("DeepEval AnswerRelevancyMetric",
                      "Did it answer THE question or a neighbouring one. A "
                      "perfectly faithful recipe for the wrong dish scores well "
                      "on faithfulness and is still useless. Blank for a "
                      "refusal."),
        "Faithful": ("DeepEval FaithfulnessMetric",
                     "Splits the answer into claims and checks each against the "
                     "retrieved context. Catches what a vocabulary check "
                     "cannot: a quantity or a step stated confidently and not "
                     "present in the source."),
        "Cookable": ("DeepEval GEval, custom criteria",
                     "Could a competent home cook follow this and get the dish. "
                     "An answer can be perfectly faithful and still useless: "
                     "right ingredients, no quantities, steps out of order. "
                     "Blank for refusals — a refusal is not an uncookable "
                     "recipe."),
    },
}


def models_in_play(stage: str) -> list[tuple[str, str]]:
    """Which models a stage actually runs, read from .env."""
    embedding = settings.embedding_model
    if stage == "stage2":
        return [("Embeddings", f"{embedding}  — only for the semantic chunkers "
                               f"and for self-sufficiency")]
    if stage == "stage3":
        return [("Embeddings", embedding),
                ("Reranker", f"{settings.reranker_model}  — only with rerank on"),
                ("Constraint reader", f"{settings.judge_model}  — only with "
                                      f"graph on, cached by question")]
    if stage == "stage4":
        return [("Embeddings", embedding),
                ("Rewriter", f"{settings.generation_model}  — decompose and "
                             f"hyde only")]
    if stage == "stage5":
        return [("Constraint reader", f"{settings.judge_model}"),
                ("Course labels", f"{settings.judge_model}  — one-off, cached "
                                  f"to data/eval/recipe_course.json"),
                ("Graph", "Neo4j Aura · no embeddings involved")]
    if stage == "stage6":
        return [("Generator", "your choice below"),
                ("Judge", f"{settings.judge_model}  — Faithfulness and Cookable")]
    return []