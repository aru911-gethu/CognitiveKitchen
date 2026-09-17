# Cognitive Kitchen

[![tests](https://github.com/aru911-gethu/CognitiveKitchen/actions/workflows/tests.yml/badge.svg)](https://github.com/aru911-gethu/CognitiveKitchen/actions/workflows/tests.yml)

**My best-performing retriever was my least safe one, and none of my metrics
could see it.**

Asked *"I'm avoiding nuts, what can I make?"*, the highest-scoring retriever in
this lab returned Nut Milk, Cashew Nut Chutney and Almond Honey Milk. That is not
a bug in the ranker. Similarity search has no direction for *without* --
"avoiding nuts" sits closest in embedding space to the recipes that are mostly
nuts. The better the ranker, the more confidently wrong.

    configuration          hit@5   constraint respected   secs
    dense                  0.500                  33.8%    0.3
    bm25                   0.350                  58.8%    0.1
    fusion:rrf             0.600                  38.3%    0.1
    rrf + rerank           0.700                  40.0%   25.8
    rrf + graph            0.700                 100.0%    1.1
    rrf + graph + rerank   0.800                 100.0%   62.5

Four standard retrieval metrics -- hit@k, recall@k, MAP, diversity -- all rank
`rrf + rerank` as a strong configuration. It satisfies the user's stated
constraint 40% of the time. A fifth metric, `constraint_respected`, is the only
one that can see this, and it exists only because I went looking for it.

`rrf + graph` matches that ranking quality, respects the constraint every time,
and runs 23x faster. Compliance is a safety axis, not a quality one, and no
amount of better ranking finds it.

## One system, two layers

A domain-agnostic evaluation core, and a cooking assistant built on top of it.
The second one is how I found out where the first one does *not* generalise yet:
eight files in the lab still import cooking by name, and
[KNOWN_ISSUES.md](KNOWN_ISSUES.md) names them.

Six stages. Each one scores every option against a hand-verified benchmark, you
lock a winner, and the next stage is measured using that choice rather than a
default.

<!-- Screenshot: save the Lab's Stage 2 comparison table to
     docs/img/lab-stage2.png, then uncomment the line below.
![The RAG Lab: eight chunkers scored on the same corpus](docs/img/lab-stage2.png)
-->

Built on one cookbook -- 222 pages, 184 recipes -- deliberately. I can verify
this corpus myself. I know besan is gluten-free, so I caught the model claiming
it wasn't.

## The stages

**Stage 2 - Chunking.** 8 chunkers, scored on chunk purity, best-case recall,
the chunks needed to reach 90% recall, and self-sufficiency.

    chunker     recall   k@90   self-sufficiency
    recipe       0.991   1.04              1.000
    recursive    0.745   1.96              0.932

`recipe` uses the recipe boundaries ingestion already found, so a chunk is a
whole recipe and never straddles two. `recursive` splits on character count, so
a single chunk can hold the end of one recipe and the start of another. When
that happened the assistant answered "something with butter" by listing butter
quantities from five unrelated recipes. Structure the parser already knew beat
every similarity-based splitter I tried.

**Stage 3 - Retrieval.** 9 retrievers, exposed as four independent controls
rather than one flat list:

    source     dense | bm25 | tfidf | hybrid | rrf     pick one
    diversity  MMR                                     optional
    rerank     cross-encoder                           optional
    graph      pre-filter by constraint                optional

Scored on hit@k, recall@k, MAP and diversity, plus `constraint_respected`. See
the table at the top: that fifth metric is the whole finding.

**Stage 4 - Query transform.** `passthrough`, `decompose`, `hyde`. `decompose`
scores best. `hyde` costs roughly 400s a query on CPU for no measurable gain --
a negative result worth keeping, because the technique is widely recommended and
the cost is invisible until you are the one without a GPU.

**Stage 5 - Pure graph.** No chunks, no embeddings, no ranking, so it is scored
on set precision and recall instead of MAP.

    set_precision 0.689 - set_recall 0.617 - exclusion family P/R 0.801 / 0.800
    constraint_extracted 1.000 - constraint_respected 1.000

Precision is measured only over the 52 recipes the golden dataset has an opinion
about, out of 184. Asked "what can I cook with cumin" the graph returns every
recipe that really contains cumin; counting the ones golden is silent on as
errors would measure the truth set's coverage, not the graph.

This stage answers what no retriever can, because the answer is in no document:

    "what can I cook tonight?"   Masoor Dhal, missing 1 of 5: masoor dal
    "instead of ghee?"           yogurt, butter, milk, cream
    "how many are dairy-free?"   92 of 177

"Missing" appears in no recipe. It is a traversal, not a similarity guess.

**Stage 6 - Generation.** 4 strategies, two computed metrics and two judged.

    strategy      NoInvent  Abstain  Faithful  Cookable  s/answer
    reordered        1.000    1.000     0.760     0.525       1.6
    stuff_strict     0.980    1.000     0.720     0.600       2.0
    structured       0.980    0.800     0.749     0.480       2.3
    map_reduce       1.000    0.800     0.550     0.420       5.2

`NoInvent` and `Abstain` are a floor: all four are grounded, so anything below
1.0 is a defect rather than a reason to prefer a strategy. `Faithful`
(DeepEval) and `Cookable` (G-Eval) are the two that discriminate.

`reordered` beats `stuff_strict` on faithfulness using the *same chunks* -- only
the order changes. That is the lost-in-the-middle effect, and no improvement to
retrieval would ever have surfaced it.

## The two routes, head to head

Same questions, same corpus, both scored on Stage 6 metrics:

    route         faithful  relevancy  abstain   secs
    retrieval        0.587      0.889    1.000    3.1
    graph only       0.694      1.000    0.833    1.4

The graph is more faithful, perfectly relevant, and twice as fast. It also
abstains less often, which is the cost: when a question falls outside what the
graph models, retrieval knows to refuse and the graph sometimes answers anyway.
Neither route dominates, which is why both ship.

## Run it

Two terminals, from the project root.

    uv run ck-api      # terminal 1, http://127.0.0.1:8010
    uv run ck-ui       # terminal 2, http://127.0.0.1:8501

API docs at http://127.0.0.1:8010/docs

## Build steps, in order

Ingestion first, then two one-off builds. Both read `data/ingested/`, so they
come after you have ingested something and before any retrieval or chat.

    uv run ck-vocab    ingredient vocabulary  -> data/eval/ingredient_map.json
    uv run ck-graph    knowledge graph        -> Neo4j

**`ck-vocab`** collapses 1,776 raw ingredient lines into 158 canonical
ingredients, so the graph holds one `ginger` node instead of sixteen. It is
deterministic and needs no network: the decisions live in `rag/vocab/curated.py`,
hand-written for this corpus rather than generated, because a model that files
besan as gluten, or treats coriander seeds and coriander leaves as one
ingredient, produces a vocabulary that is wrong in the ways a cook notices.
Re-run it after ingesting a new book; names it has never seen keep their
rule-normalised form, which is safe rather than silently wrong.

**`ck-graph`** rebuilds from every run in `data/ingested/`. It wipes first, so it
is repeatable. `--dry-run` shapes and reports without touching the database.

    Recipe 195 (177 real)   Ingredient 156   Category 6
    Step 1358   USES 1658   NEXT 1164   PERFORMS 869   NEEDS 242

## The golden dataset

`data/golden_dataset.json` is the benchmark, not an input. 50 recipes rebuilt
directly from the sample PDF and verified against it: no missing ingredients,
nothing fabricated, every derived field recomputed. It carries 229 evaluation
queries across seven families, plus substitutions, moods and a pinned corpus
hash.

It is evaluation-only truth. Nothing on the build side reads it, which is the
point -- a benchmark the system can see is a benchmark the system will fit.

I also found it wrong twice. Two queries listed fish and prawn recipes as valid
answers to "no meat". Both are corrected, and the correction is recorded in the
`provenance` block rather than quietly fixed.

## Known issues

Open items with enough diagnosis to pick up cold, in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md). The largest: the `constraint` query family
scores 0.000 for every retriever, because all 14 of its questions ask about
numbers -- minutes, ingredient counts -- and nothing in the pipeline compares
numbers. Deferred deliberately, with a sketch of the fix.

## What the UI does

**PDF tab** - drop a cookbook PDF, or ingest the sample in `data/`. Every page
read emits a live frame: pages read, recipes found, characters, elapsed.

**Web URLs tab** - one URL per line, a recipe page or a category index. With
"Expand category pages" on, an index is crawled for recipe links, ranked so the
page budget goes to real recipes rather than navigation.

**Ingested data tab** - every run in `data/ingested/`, with per-recipe
ingredient and step counts.

**RAG Lab** - the stage-by-stage comparison above, run live.

**Kitchen** - the chat, reading whatever `data/eval/pipeline.json` has locked.

## Output format

One shape for both sources, so later stages do not care where a recipe came
from:

    run_id, source_type (pdf|url), origin, started_at, finished_at,
    elapsed_seconds, n_units_total, n_units_read, n_units_empty, empty_units,
    n_recipes, warnings, recipes[]

Each recipe: `recipe_id, title, source_type, origin, pages[], url,
ingredient_lines[], step_lines[], servings_hint, time_hint, detected_by,
raw_text`.

`detected_by` records how it was found: `json-ld` (schema.org markup, most
reliable), `headings` (an Ingredients/Method heading), `prose-boundary` (no
heading, inferred from where instructions start), or `heuristic-text`.

## Layout

    src/cognitive_kitchen/
      config.py              settings from .env
      models.py              IngestionRun, RawRecipe, ProgressEvent
      jobs.py                job registry, per-job event queue
      api.py                 FastAPI endpoints + SSE stream
      ingest/pdf_ingest.py   page-by-page PDF reader
      ingest/web_ingest.py   Playwright crawler
      ui/app.py              Streamlit console
      ui/pages/2_RAG_Lab.py  per-stage strategy comparison
      ui/pages/3_Kitchen.py  chat, on the locked pipeline

      rag/
        registry.py          plugin registry, strategies auto-discovered
        corpus.py            render recipes to text, record spans
        telemetry.py         latency, tokens and dollars per stage
        memory.py            chat history, follow-up resolution
        loaders/             ingested JSON, PDF
        chunking/            8 chunkers, incl. recipe and recipe_sections
        embedding/           sentence-transformers, model from .env
        retrieval/           9 retrievers, incl. graph_hybrid and graph_only
        query/               passthrough, decompose, hyde
        generate/            local Qwen, and gpt-4o-mini for judging
        vocab/               ingredient identity  (ck-vocab)
        graph/               Neo4j build + traversal  (ck-graph)
        eval/                per-stage metrics, scored on the golden dataset

## Tests

    uv run pytest tests -q

60 tests. Each one pins a bug that shipped once: a source-prefix collision that
let a PDF recipe inherit a URL recipe's constraint eligibility, a diversity
metric that could exceed 1.0, a filter that returned all 168 recipes when it
could not resolve the include term.

## Setup from scratch

    uv sync
    uv run playwright install chromium
    cp .env.example .env      # then fill in real values
    # ingest a cookbook through the UI, then:
    uv run ck-vocab
    uv run ck-graph

## Security

The API has **no authentication**, and it both accepts file uploads and fetches
user-supplied URLs. It is bound to localhost for that reason. Do not expose it
to a network without adding auth.

Two guards are in place:

- **SSRF** - non-HTTP schemes are refused, and hosts resolving to loopback,
  private, link-local or reserved addresses are rejected before any fetch.
- **robots.txt** - checked per host and honoured; disallowed URLs are skipped
  with a warning rather than fetched.

Uploads are capped at 80 MB and must be `.pdf`.

## Cost

Everything on the answering path runs locally and costs nothing. Two things call
a paid model, both `gpt-4o-mini`:

- **constraint extraction** in `graph_hybrid`, about $0.0001 a question, cached
  to disk by question so replaying an evaluation is free
- **judged generation metrics**, faithfulness and cookability

`rag/telemetry.py` records tokens, latency and dollars per stage, so every
metrics table can show what a strategy cost as well as how well it scored.

## Environment notes

Neo4j needs `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` and `NEO4J_DATABASE`
in `.env`. On Neo4j Aura the username and the database are both the instance id,
something like `09bfbafe`, not the literal `neo4j`. Bolt uses port 7687, which
corporate VPNs commonly block; if you see "Unable to retrieve routing
information" while port 443 on the same host is open, that is the VPN and not the
instance.

The API defaults to port 8010 rather than 8000, because on my machine
AWSWorkDocsDriveClient already holds `0.0.0.0:8000`. Change `API_PORT` in `.env`
if you prefer another.

No GPU was used anywhere. 12 CPU cores, 15.6 GB RAM, and every model choice
follows from that.
