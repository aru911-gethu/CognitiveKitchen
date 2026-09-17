# Known issues

Open items, with enough diagnosis to pick each one up cold. Numbers are measured
on the 184-recipe PDF corpus unless stated.

---

## 1. The `constraint` query family scores 0.000

**Status:** deferred, revisit later.

Every retriever gets zero, including the graph. Not a ranking problem.

### Why

All 14 questions in the family ask about **numbers**, not about ingredients:

    q_0151 - q_0160   "Something I can finish in 20 / 30 / 45 / 60 / 90 minutes or less"
    q_0161 - q_0164   "A recipe with 5 / 6 / 8 / 10 ingredients or fewer"

Nothing in the pipeline compares numbers.

- **Similarity cannot.** "20 minutes" appears in the text of a recipe that takes
  ninety. Embedding distance has no notion of less-than, exactly as it has no
  notion of absence, which is the same reason `exclusion` needed the graph.
- **The graph cannot, yet.** `Recipe` nodes carry `n_ingredients` -- set in
  `graph/schema.py` -- but no query ever reads it. Duration is not stored at all.
- **The extractor does not look.** `graph/constraints.py` returns include,
  exclude, exclude_categories, course and substitute_for. There is no field for a
  numeric bound, so the question's actual content is discarded before the
  traversal runs.

### What the data supports

The ingredient half is nearly free. The time half is not:

    recipes with a usable time_hint     4 of 184   (2%)
    golden with total_time_minutes      1 of 50

`time_hint` is not the route. Golden's rationale says 34 recipes have a
"recoverable duration", and golden carries `timed_steps_minutes_sum` -- so it
derived time by summing durations mentioned in the METHOD TEXT, not from a
field:

    "simmer for 20 minutes"  ->  20
    "soak overnight"         ->  ?      needs a decision
    "cook until done"        ->  nothing

That is reproducible on the build side, since `step_lines` holds the same
sentences. Golden reached 68% coverage this way, so roughly 125 of 184 recipes
looks achievable.

### Sketch of the fix

1. `graph/duration.py` -- parse durations out of `step_lines`. Sum them into
   `total_minutes`, and record `duration_confidence` (how many steps had a
   number) so a recipe with one timed step out of twelve is not treated as
   precisely known.
2. `graph/schema.py` -- write `total_minutes` onto Recipe nodes.
   `n_ingredients` is already there.
3. `graph/constraints.py` -- add `max_minutes` and `max_ingredients` to the
   extracted Constraints, and to the prompt.
4. `graph/traverse.py` -- `filter_recipes` gains range predicates:
   `WHERE r.total_minutes <= $max_minutes`.
5. Re-measure. Recall on the time questions will be capped by parse coverage, so
   report coverage beside the score or the number will read as a retrieval
   failure when it is a source-data limit.

Roughly 80 lines plus the parser. The ingredient-count questions (q_0161-q_0164)
would work immediately after steps 3 and 4, since the property already exists --
worth doing first, as it splits the family cleanly and proves the mechanism
before investing in duration parsing.

### Why it is deferred

`exclusion` was worth the graph because a wrong answer is a safety problem: tell
someone avoiding nuts to make Nut Milk and it matters. A wrong answer here means
a dish takes forty minutes instead of twenty. Real, but not urgent, and the fix
is bounded by how many recipes state their timings at all.

---

## 2. Stage 6 numbers come from `gpt-4o-mini`, not the local model

**Status:** open.

The strategy ranking was measured with the API model, because `map_reduce` makes
one call per chunk and the local model runs at roughly a minute an answer on CPU
-- around thirty minutes for one comparison.

    strategy      NoInvent  Abstain  Faithful  Relevancy  Cookable  s/answer
    reordered        1.000    1.000     0.760      ...       0.525       1.6
    stuff_strict     0.980    1.000     0.720      ...       0.600       2.0
    structured       0.980    0.800     0.749      ...       0.480       2.3
    map_reduce       1.000    0.800     0.550      ...       0.420       5.2

`reordered` beating `stuff_strict` on faithfulness with identical chunks is the
lost-in-the-middle effect, and that effect is normally *stronger* on smaller
models -- so the gap may widen on Qwen, or may vanish into noise at n=5. It is
unvalidated either way. One slow run settles it.

Note these predate the move of `Cookable` from a hand-written prompt to GEval, so
that column will shift regardless.

---

## 3. `hyde` has no trustworthy number

**Status:** open, low priority.

Roughly 400s per query on CPU, since it needs a generation call before it can
retrieve. Measured once at hit@5 0.600 against passthrough's 0.600 -- no gain for
several minutes a query. Registered and selectable, excluded from the Stage 4
defaults. Worth revisiting on the GPU machine, where the cost disappears.

---

## 4. Golden query artifacts

**Status:** accepted, not fixable from this side.

Three questions ask about words that are not ingredients:

    "I have some black left over. Any ideas?"     black
    "I have some pods left over. Any ideas?"      pods
    "What can I cook with freshly?"               freshly    (q_0053)

The subject was taken from an ingredient line without checking it names an
ingredient -- "black" from black mustard seeds, "pods" from cardamom pods. These
are unanswerable by construction and drag `by_ingredient` down for every
strategy equally, so comparisons stay valid. The graph now returns nothing rather
than all 177 recipes, which is the honest reply.

Two recipes have unusable content fingerprints, `ck_023` and `ck_048`, so 48 of
50 golden recipes can be matched to pipeline recipes.

---

## 5. A dish name is not a graph pattern

**Status:** by design, worth knowing.

The graph-only route refuses `by_dish` questions:

    "Give me the recipe for Samosa."       matched 10, course=snack   -> refused
    "Do you have anything called Aviyal?"  matched 168, no constraint -> refused

There is no node for "samosa". The traversal narrows to `course=snack` and stops,
so the generator sees five snacks that are not samosa and correctly declines.
Inverted from the retrieval route, which scores 0.80 on `by_dish` and zero on
`exclusion`. This is why both routes exist rather than one replacing the other.

Related: 86 recipes match "avoiding dairy" and 5 reach the generator. A traversal
has no relevance ranking, so which 5 is a heuristic -- fewest ingredients first,
in `retrieval/graph_only.py`. Documented rather than hidden, because it is a real
weakness of answering from structure alone.