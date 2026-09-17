# Cognitive Kitchen

RAG-based cooking assistant grounded in recipes you own.

Phase 1 is ingestion: pull recipes out of PDFs and websites into one structured
format, streaming progress as pages are read. Phase 2 is the RAG lab: swappable
chunkers, retrievers, query transforms and a knowledge graph, each stage scored
so you can pick a strategy on evidence rather than taste.

## Run it

Two terminals, from the project root.

Terminal 1 - the API:

    uv run ck-api

Terminal 2 - the UI:

    uv run ck-ui

Then open http://127.0.0.1:8501

The API listens on http://127.0.0.1:8010 and serves interactive docs at
http://127.0.0.1:8010/docs

> Port 8010, not 8000: on this machine AWSWorkDocsDriveClient already holds
> 0.0.0.0:8000. Change `API_PORT` in `.env` if you prefer another.

## Build steps, in order

Ingestion first, then two one-off builds. Both read `data/ingested/`, so they
come after you have ingested something and before any retrieval or chat.

    uv run ck-vocab     ingredient vocabulary  -> data/eval/ingredient_map.json
    uv run ck-graph     knowledge graph        -> Neo4j

**`ck-vocab`** collapses 1,776 raw ingredient lines into 158 canonical
ingredients, so the graph has one `ginger` node instead of sixteen. It is
deterministic and needs no network: the decisions live in
`rag/vocab/curated.py`, hand-written for this corpus rather than generated,
because a model that files besan as gluten or treats coriander seeds and
coriander leaves as one thing produces a vocabulary that is wrong in ways that
matter to a cook. Re-run it after ingesting a new book; names it has never seen
keep their rule-normalised form, which is safe rather than silently wrong.

**`ck-graph`** rebuilds the graph from every run in `data/ingested/`. It wipes
first, so it is repeatable. Add `--dry-run` to shape and report without touching
the database.

    Recipe 195 (177 real)   Ingredient 156   Category 6
    Step 1358   USES 1658   NEXT 1164   PERFORMS 869   NEEDS 242

Needs `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` and `NEO4J_DATABASE` in
`.env`. On Neo4j Aura the username and the database are both the instance id,
something like `09bfbafe`, not the literal `neo4j`. Bolt uses port 7687, which
corporate VPNs commonly block; if you get "Unable to retrieve routing
information" while port 443 on the same host is open, that is the VPN, not the
instance.

## The stages

Each stage scores every option, you lock one, and the next stage is measured
using that choice rather than a default. Numbers below are from this corpus.

**Stage 2 · Chunking** — 6 chunkers. `word_purity`, `word_recall_best`,
`k_at_90`, `self_sufficiency`. `recursive` wins all four.

**Stage 3 · Retrieval** — four independent controls rather than one list:

    source     dense | bm25 | tfidf | hybrid | rrf      pick one
    diversity  MMR                                      optional
    rerank     cross-encoder                            optional
    graph      pre-filter by constraint                 optional

Scored on `hit@k`, `recall@k`, MAP, diversity — and `constraint_respected`,
which the other four cannot see:

    configuration        hit@5  recall     MAP  constraint   secs
    dense                0.500   0.297  0.2925      33.8%     0.3
    bm25                 0.350   0.239  0.2325      58.8%     0.1
    fusion:rrf           0.600   0.331  0.3258      38.3%     0.1
    rrf +rerank          0.700   0.363  0.3633      40.0%    25.8
    rrf +graph           0.700   0.351  0.3458     100.0%     1.1
    rrf +graph +rerank   0.800   0.388  0.3883     100.0%    62.5

Compliance is a safety axis, not a quality one. Asked "I am avoiding nuts",
`dense` returns Nut Milk and the cross-encoder returns Nut Milk, Cashew Chutney
and Almond Milk: the better the ranker, the more confidently wrong, because
"avoiding nuts" sits closest in embedding space to recipes about nuts. Negation
has no direction there. Only the graph rows reach 100%.

**Stage 4 · Query transform** — `passthrough`, `decompose`, `hyde`.
`decompose` scores best; `hyde` costs ~400s a query on CPU for no gain.

**Stage 5 · Pure graph** — standalone. No chunks, no embeddings, no ranking, so
it is scored on set precision and recall instead of MAP.

    set_precision 0.689 · set_recall 0.617 · exclusion family P/R 0.801 / 0.800
    constraint_extracted 1.000 · constraint_respected 1.000

Precision is measured only over the 52 recipes the golden dataset covers, out of
184. Asked "what can I cook with cumin" the graph returns every recipe that
really contains cumin; scoring the ones golden has no opinion about as errors
would measure the truth set's coverage rather than the graph.

This stage answers what no retriever can, because the answer is in no document:

    "what can I cook tonight?"   Masoor Dhal, missing 1 of 5: masoor dal
    "instead of ghee?"           yogurt, butter, milk, cream
    "how many are dairy-free?"   92 of 177

**Stage 6 · Generation** — four strategies, two computed metrics and two judged.

    strategy      NoInvent  Abstain  Faithful  Cookable  s/answer
    reordered        1.000    1.000     0.760     0.525       1.6
    stuff_strict     0.980    1.000     0.720     0.600       2.0
    structured       0.980    0.800     0.749     0.480       2.3
    map_reduce       1.000    0.800     0.550     0.420       5.2

`NoInvented` and `Abstention` are a floor: all four are grounded, so a score
below 1.0 is a defect rather than a reason to reject a strategy. `Faithfulness`
(DeepEval) and `Cookable` discriminate.

`reordered` beats `stuff_strict` on faithfulness with the *same chunks* — only
the order changes. That gap is the lost-in-the-middle effect, and no amount of
better retrieval would have revealed it.

Locking all four writes `data/eval/pipeline.json`, which the Kitchen page reads.

## What the UI does

**PDF tab** - drop a cookbook PDF, or ingest the sample already in `data/`.
Every page read emits a live frame: pages read, recipes found, characters,
elapsed. Uploads land in `data/uploads/`.

**Web URLs tab** - paste one URL per line. Either a direct recipe page or a
category/index page. With "Expand category pages" on, an index is crawled for
recipe links, ranked so the page budget is spent on real recipes rather than
navigation. Each page fetched emits a frame.

**Ingested data tab** - every run saved to `data/ingested/`, with per-recipe
ingredient and step counts.

## Output format

One shape for both sources, so later stages do not care where a recipe came
from. See `data/ingested/*.json`:

    run_id, source_type (pdf|url), origin, started_at, finished_at,
    elapsed_seconds, n_units_total, n_units_read, n_units_empty, empty_units,
    n_recipes, warnings, recipes[]

Each recipe: `recipe_id, title, source_type, origin, pages[], url,
ingredient_lines[], step_lines[], servings_hint, time_hint, detected_by,
raw_text`.

`detected_by` records how it was found: `json-ld` (schema.org markup, most
reliable), `headings` (an Ingredients/Method heading), `prose-boundary` (no
heading, inferred from where instructions start), or `heuristic-text`.

## Known issues

Open items, each with enough diagnosis to pick up cold, in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md). The main one: the `constraint` query family
scores 0.000 for every retriever because all 14 of its questions ask about
numbers -- minutes and ingredient counts -- and nothing in the pipeline compares
numbers. Deferred, with a sketch of the fix.

## The golden dataset

`data/golden_dataset.json` is the benchmark, not an input. 50 recipes rebuilt
directly from the sample PDF and verified against it: no missing ingredients,
nothing fabricated, every derived field recomputed. It carries 229 evaluation
queries across seven families, plus substitutions, moods and a pinned corpus
hash. All retrieval and evaluation work is scored against it.

Its `validation` and `provenance` blocks record what was checked and the
caveats that remain.

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

      rag/
        registry.py          plugin registry, strategies auto-discovered
        corpus.py            render recipes to text, record spans
        telemetry.py         latency, tokens and dollars per stage
        loaders/             ingested JSON, PDF
        chunking/            6 chunkers
        embedding/           sentence-transformers, model from .env
        retrieval/           8 retrievers, incl. graph_hybrid
        query/               passthrough, decompose, hyde
        generate/            local Qwen, and gpt-4o-mini for judging
        vocab/               ingredient identity  (ck-vocab)
        graph/               Neo4j build + traversal  (ck-graph)
        eval/                per-stage metrics, scored on the golden dataset

## Security

The API has **no authentication** and it both accepts file uploads and fetches
user-supplied URLs. It is bound to localhost for that reason. Do not expose it
to a network without adding auth.

Two guards are in place:

- **SSRF** - non-HTTP schemes are refused, and hosts resolving to loopback,
  private, link-local or reserved addresses are rejected before any fetch.
- **robots.txt** - checked per host and honoured; disallowed URLs are skipped
  with a warning rather than fetched.

Uploads are capped at 80 MB and must be `.pdf`.

## Setup from scratch

    uv sync
    uv run playwright install chromium
    cp .env.example .env      # then fill in real values
    # ingest a cookbook through the UI, then:
    uv run ck-vocab
    uv run ck-graph

## Cost

Everything on the answering path runs locally and costs nothing. Two things call
a paid model, both `gpt-4o-mini`:

- **constraint extraction** in `graph_hybrid`, roughly $0.0001 a question, cached
  to disk by question so replaying an evaluation is free
- **judged generation metrics**, faithfulness and cookability

`rag/telemetry.py` records tokens, latency and dollars per stage, so every
metrics table can show what a strategy cost as well as how well it scored.