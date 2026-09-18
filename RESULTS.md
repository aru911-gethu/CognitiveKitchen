# Results

Every measured number in this project, in one file, so a re-run edits one place.

## Provenance

```
corpus           1 cookbook PDF, 222 pages, 184 recipes
                 + 11 recipes crawled from chitrasfoodbook.com
benchmark        data/golden_dataset.json - 50 recipes rebuilt from the PDF
                 and verified against it, 229 queries across 7 families
hardware         12 CPU cores, 15.6 GB RAM, no GPU
embeddings       sentence-transformers/all-MiniLM-L6-v2
generation       Qwen/Qwen2.5-1.5B-Instruct  (CPU, float32)
reranker         cross-encoder/ms-marco-MiniLM-L-6-v2
judge            gpt-4o-mini
```

Models come from `.env`, which overrides the defaults in `config.py`. The numbers
below were produced with the models named above, not with the `config.py`
defaults. The Lab's model selector can re-measure any stage on a different model.

**Which numbers move when the generation model changes:** Stage 6 and the
head-to-head only. Stage 2 is set arithmetic over chunk boundaries. Stage 3
ranking depends on the embedder and retriever, and `constraint_respected` depends
on the graph and the constraint extractor. Stage 5 is graph traversal. None of
those involve the generator.

## Stage 2 - Chunking

8 chunkers scored. `recipe` uses the boundaries ingestion already found, so a
chunk is one whole recipe and never straddles two.

```
chunker     recall   k@90   self-sufficiency
recipe       0.991   1.04              1.000
recursive    0.745   1.96              0.932
```

`recursive` splits on character count, so a chunk can hold the end of one recipe
and the start of another. That produced an answer to "something with butter" made
of butter quantities from five unrelated dishes. Structure the parser already
knew beat every similarity-based splitter tried.

## Stage 3 - Retrieval

9 retrievers, exposed as four independent controls: source, diversity, rerank,
graph pre-filter.

```
configuration          hit@5   constraint respected   secs
dense                  0.500                  33.8%    0.3
bm25                   0.350                  58.8%    0.1
fusion:rrf             0.600                  38.3%    0.1
rrf + rerank           0.700                  40.0%   25.8
rrf + graph            0.700                 100.0%    1.1
rrf + graph + rerank   0.800                 100.0%   62.5
```

The four standard metrics - hit@k, recall@k, MAP, diversity - all rank
`rrf + rerank` as strong. It satisfies the user's stated constraint 40% of the
time. `constraint_respected` is the only metric that sees this.

`rrf + graph` matches that ranking quality, respects the constraint every time,
and runs 23x faster. Compliance is a safety axis, not a quality one.

**The `constraint` family scores 0.000 for every retriever.** All 14 of its
questions ask about numbers - minutes, ingredient counts - and nothing in the
pipeline compares numbers. See KNOWN_ISSUES.md.

## Stage 4 - Query transform

`passthrough`, `decompose`, `hyde`. `decompose` scores best. `hyde` costs roughly
400s per query on CPU for no measurable gain - a negative result worth keeping,
because the technique is widely recommended and the cost is invisible until you
are the one without a GPU.

## Stage 5 - Pure graph

No chunks, no embeddings, no ranking, so it is scored on set precision and recall
rather than MAP.

```
set_precision          0.689
set_recall             0.617
exclusion family P/R   0.801 / 0.800
constraint_extracted   1.000
constraint_respected   1.000
```

Precision is measured only over the 52 recipes the golden dataset has an opinion
about, out of 184. Counting the ones golden is silent on as errors would measure
the truth set's coverage, not the graph.

## Stage 6 - Generation

4 strategies, two computed metrics and two judged.

```
strategy      NoInvent  Abstain  Faithful  Cookable  s/answer
reordered        1.000    1.000     0.760     0.525       1.6
stuff_strict     0.980    1.000     0.720     0.600       2.0
structured       0.980    0.800     0.749     0.480       2.3
map_reduce       1.000    0.800     0.550     0.420       5.2
```

`NoInvent` and `Abstain` are a floor: all four are grounded, so anything below
1.0 is a defect rather than a reason to prefer a strategy. `Faithful` (DeepEval)
and `Cookable` (G-Eval) are the two that discriminate.

`reordered` beats `stuff_strict` on faithfulness using the *same chunks* - only
the order changes. That is the lost-in-the-middle effect, and no improvement to
retrieval would have surfaced it.

## The two routes, head to head

Same questions, same corpus, both scored on Stage 6 metrics.

```
route         faithful  relevancy  abstain   secs
retrieval        0.587      0.889    1.000    3.1
graph only       0.694      1.000    0.833    1.4
```

The graph is more faithful, perfectly relevant, and twice as fast. It also
abstains less often, which is the cost: when a question falls outside what the
graph models, retrieval knows to refuse and the graph sometimes answers anyway.
Neither route dominates, which is why both ship.

## Graph shape

```
Recipe 195 (177 real)   Ingredient 156   Category 6   Step 1358
USES 1658   NEXT 1164   PERFORMS 869   NEEDS 242
```

`ck-vocab` collapses 1,776 raw ingredient lines into 158 canonical ingredients,
so the graph holds one `ginger` node instead of sixteen.

## What runs without a labelled truth set

Verified by hiding `golden_dataset.json` and supplying `data/questions.txt` with
three plain questions:

```
hit_at_k               None    needs relevance labels
recall_at_k            None    needs relevance labels
map                    None    needs relevance labels
diversity_at_k         1.0     computed
constraint_respected   0.9     computed
```

Recall is the reason labels cannot be replaced by an LLM judge: to know a
retriever missed something you have to know what existed, and a judge only sees
what you showed it. The constraint metric - the one thing no comparable tool
measures - needs questions, not an answer key.

## Corrections

The golden dataset was wrong twice. Queries `q_0119` and `q_0129` listed fish and
prawn recipes (`ck_041`, `ck_042`) as valid answers to "no meat". Both corrected,
and recorded in the `provenance.corrections_to_previous_build` block rather than
quietly fixed.