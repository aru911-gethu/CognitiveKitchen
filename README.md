# Cognitive Kitchen 🍳⚡
**Enterprise RAG Strategy Benchmarking Testbed & AI Culinary Assistant**

[![tests](https://github.com/aru911-gethu/CognitiveKitchen/actions/workflows/tests.yml/badge.svg)](https://github.com/aru911-gethu/CognitiveKitchen/actions/workflows/tests.yml)

> *"A RAG test kitchen where AI Product Engineers and Technical Program Managers (TPMs) can play around with, benchmark, and taste-test diverse RAG strategies to lock the optimal production pipeline for their enterprise use-case — and as a culinary pun, use the exact same locked pipeline to power an interactive recipe-based kitchen assistant answering: **'What can I cook tonight?'***"*

---

## 📦 Package Installation & CLI Executables

`cognitive-kitchen` is packaged as a standard Python project configured in [pyproject.toml](file:///c:/Users/aru91/OneDrive/Documents/Workspace_AI/Projects_Vscode/CognitiveKitchen/pyproject.toml). Installing the package registers four dedicated `ck-*` CLI commands in your environment:

| CLI Command | Handler Function | Purpose & Output |
| :--- | :--- | :--- |
| **`uv run ck-api`** | `cognitive_kitchen:serve_api` | Launches FastAPI REST API & SSE server on `http://127.0.0.1:8010` |
| **`uv run ck-ui`** | `cognitive_kitchen:serve_ui` | Launches Streamlit multi-page UI console on `http://127.0.0.1:8501` |
| **`uv run ck-vocab`** | `cognitive_kitchen:build_vocabulary` | Normalizes ingredient strings to `data/eval/ingredient_map.json` |
| **`uv run ck-graph`** | `cognitive_kitchen:build_graph` | Ingests JSON runs & rebuilds Neo4j Knowledge Graph |

### Installation Modes

#### Option A: Using `uv` (Recommended)
`uv` automatically creates a virtual environment, installs dependencies, and links the `ck-*` executables in editable mode:
```bash
git clone https://github.com/aru911-gethu/CognitiveKitchen.git
cd CognitiveKitchen
uv sync
uv run playwright install chromium
```

#### Option B: Standard `pip` Editable Install
If using standard `pip` inside a virtual environment (`python -m venv .venv`):
```bash
source .venv/bin/activate  # Or `.venv\Scripts\activate` on Windows
pip install -e .
playwright install chromium
```
*When using standard `pip`, you can invoke commands directly as `ck-api`, `ck-ui`, `ck-vocab`, and `ck-graph`.*

---

## 🎯 Executive Overview & Product Strategy

When deploying Retrieval-Augmented Generation (RAG) into production, standard search metrics like $Hit@K$, $Recall@K$, and $MAP$ present a dangerous blind spot: **they measure topic relevance, not constraint safety**.

In our empirical testing on a hand-verified 184-recipe cookbook corpus, when a user asks *"I'm avoiding nuts, what can I make?"*, standard dense embedding search suffers from **Semantic Collapse**. Vectors for "avoiding nuts" sit closest in vector space to documents dense in nuts. As a result, standard top-performing retrievers returned *Nut Milk*, *Cashew Nut Chutney*, and *Almond Honey Milk*.

```
Strategy Configuration        Hit@5   Constraint Respected   Latency (sec)
---------------------------- ------- ---------------------- ---------------
Dense (bge-small-en-v1.5)     0.500           33.8%              0.3s
BM25 (Keyword Lexical)        0.350           58.8%              0.1s
Fusion (RRF)                  0.600           38.3%              0.1s
RRF + Cross-Encoder Rerank    0.700           40.0%             25.8s
RRF + Neo4j Graph Filter      0.700          100.0%              1.1s
RRF + Graph Filter + Rerank   0.800          100.0%             62.5s
```

### The Key Technical Takeaway
Standard metrics rank `RRF + Rerank` as a top performer ($Hit@5 = 0.700$), yet it violates user safety constraints **60% of the time**. 

By introducing **`constraint_respected`** as a core evaluation axis and leveraging **Neo4j Knowledge Graph pre-filtering**, `RRF + Graph Filter` achieves matching retrieval quality ($Hit@5 = 0.700$), **100% constraint compliance**, and operates **23x faster** than heavy neural re-rankers.

---

## 🏗️ Architecture & 6-Stage RAG Pipeline Mapping

Cognitive Kitchen features a modular plugin registry ([src/cognitive_kitchen/rag/registry.py](file:///c:/Users/aru91/OneDrive/Documents/Workspace_AI/Projects_Vscode/CognitiveKitchen/src/cognitive_kitchen/rag/registry.py)) that isolates every stage of the RAG lifecycle. Each stage is independently benchmarked against our Ground-Truth Golden Dataset ([data/golden_dataset.json](file:///c:/Users/aru91/OneDrive/Documents/Workspace_AI/Projects_Vscode/CognitiveKitchen/data/golden_dataset.json)).

```
                       ┌────────────────────────────────────────────────┐
                       │  STAGE 1: Ingestion & Live SSE Parser Stream    │
                       │  PDF (pdf_ingest.py) | Web Crawl (web_ingest.py)│
                       └───────────────────────┬────────────────────────┘
                                               │
                       ┌───────────────────────▼────────────────────────┐
                       │  STAGE 2: Structure-Aware Chunking Matrix     │
                       │  Recipe / Recursive / Semantic / Markdown      │
                       └───────────────────────┬────────────────────────┘
                                               │
                       ┌───────────────────────▼────────────────────────┐
                       │  STAGE 3: Hybrid Retrieval & Safety Pre-Filter │
                       │  Dense + BM25 + Hybrid RRF + Neo4j Graph       │
                       └───────────────────────┬────────────────────────┘
                                               │
                       ┌───────────────────────▼────────────────────────┐
                       │  STAGE 4: Query Transformation                 │
                       │  Passthrough / Sub-Query Decomposition / HyDE  │
                       └───────────────────────┬────────────────────────┘
                                               │
    ┌──────────────────────────────────────────┴──────────────────────────────────────────┐
    │                                                                                     │
┌───▼───────────────────────────────────────────────┐ ┌───▼───────────────────────────────────────────────┐
│ STAGE 5: Pure Knowledge Graph Traversal            │ │ STAGE 6: Local LLM Generation & DeepEval          │
│ Neo4j Graph Traversal & Vocabulary Canonicalization│ │ Reordered / Stuff Strict / Map-Reduce / Structured│
└───────────────────────────────────────────────────┘ └───────────────────────────────────────────────────┘
```

### Stage 1 — Multi-Source Ingestion & Streaming
- **PDF Ingest** ([src/cognitive_kitchen/ingest/pdf_ingest.py](file:///c:/Users/aru91/OneDrive/Documents/Workspace_AI/Projects_Vscode/CognitiveKitchen/src/cognitive_kitchen/ingest/pdf_ingest.py)): Page-by-page extraction emitting Server-Sent Events (`ProgressEvent`) tracking elapsed time, character counts, and recipes detected.
- **Web Crawler** ([src/cognitive_kitchen/ingest/web_ingest.py](file:///c:/Users/aru91/OneDrive/Documents/Workspace_AI/Projects_Vscode/CognitiveKitchen/src/cognitive_kitchen/ingest/web_ingest.py)): Playwright-driven crawler with SSRF safeguards, domain boundary rules, and Schema.org JSON-LD parsing (`detected_by: json-ld | headings | prose-boundary | heuristic-text`).

### Stage 2 — Chunking Strategy Evaluation
- **Plugins** ([src/cognitive_kitchen/rag/chunking/](file:///c:/Users/aru91/OneDrive/Documents/Workspace_AI/Projects_Vscode/CognitiveKitchen/src/cognitive_kitchen/rag/chunking)): 8 chunkers (`recipe`, `recipe_sections`, `recursive`, `semantic_adjacent`, `markdown_header`, `character`, `token`, `sentence`).
- **Benchmark Findings**: Structure-aware `recipe` chunking (keeping single recipes intact) yields $0.991$ recall and $1.000$ self-sufficiency. Arbitrary character-recursive chunking straddles recipe boundaries, causing LLMs to hallucinate ingredient quantities from unrelated recipes.

### Stage 3 — Retrieval & Safety Pre-Filtering
- **Plugins** ([src/cognitive_kitchen/rag/retrieval/](file:///c:/Users/aru91/OneDrive/Documents/Workspace_AI/Projects_Vscode/CognitiveKitchen/src/cognitive_kitchen/rag/retrieval)): 9 retrieval strategies combining Dense (`bge-small-en-v1.5`), Lexical (BM25, TF-IDF), Hybrid Reciprocal Rank Fusion (RRF), Maximal Marginal Relevance (MMR), Cross-Encoder Re-ranking, and Neo4j Graph Constraint Pre-filtering (`graph_hybrid`).
- **Evaluated On**: $Hit@K$, $Recall@K$, $MAP$, $Diversity$, and `constraint_respected`.

### Stage 4 — Query Transformations
- **Plugins** ([src/cognitive_kitchen/rag/query/](file:///c:/Users/aru91/OneDrive/Documents/Workspace_AI/Projects_Vscode/CognitiveKitchen/src/cognitive_kitchen/rag/query)): `passthrough`, `decompose`, `hyde`.
- **Benchmark Findings**: Sub-query `decompose` provides highest retrieval boost for multi-intent questions. Hypothetical Document Embeddings (`hyde`) cost ~400s per query on CPU without measurable accuracy gains.

### Stage 5 — Pure Knowledge Graph Traversal
- **Graph Traversal** ([src/cognitive_kitchen/rag/graph/](file:///c:/Users/aru91/OneDrive/Documents/Workspace_AI/Projects_Vscode/CognitiveKitchen/src/cognitive_kitchen/rag/graph)): Direct Cypher query engine on Neo4j for zero-embedding deterministic graph traversal.
- **Canonical Vocabulary Engine** ([src/cognitive_kitchen/rag/vocab/curated.py](file:///c:/Users/aru91/OneDrive/Documents/Workspace_AI/Projects_Vscode/CognitiveKitchen/src/cognitive_kitchen/rag/vocab/curated.py)): Deterministic rule engine executed via `uv run ck-vocab`, collapsing 1,776 raw ingredient variations into 158 canonical entities (e.g. mapping fresh ginger, ginger paste, and ground ginger safely).
- **Capability**: Answers complex graph logic that vector search cannot reach:
  - *"What can I cook tonight with my current pantry items?"* (Returns exact recipes and missing ingredients).
  - *"What can I substitute for ghee?"* (Traverses ingredient category nodes).
  - *"How many recipes are dairy-free?"* (Direct graph aggregation).

### Stage 6 — LLM Generation & Faithfulness
- **Plugins** ([src/cognitive_kitchen/rag/generate/](file:///c:/Users/aru91/OneDrive/Documents/Workspace_AI/Projects_Vscode/CognitiveKitchen/src/cognitive_kitchen/rag/generate)): Local Qwen generation supporting `reordered`, `stuff_strict`, `structured`, and `map_reduce` strategies.
- **Evaluated On**: DeepEval Faithfulness & G-Eval Cookability scoring.
- **Lost-in-the-Middle Verification**: `reordered` outperforms `stuff_strict` on identical context chunks simply by placing high-relevance chunks at context boundaries.

---

## 🖥️ Streamlit Interactive UI Console

Cognitive Kitchen delivers a 3-page interactive web application ([src/cognitive_kitchen/ui/](file:///c:/Users/aru91/OneDrive/Documents/Workspace_AI/Projects_Vscode/CognitiveKitchen/src/cognitive_kitchen/ui)):

```
                               ┌──────────────────────────────────────────┐
                               │       Streamlit Multi-Page Console       │
                               └────────────────────┬─────────────────────┘
                                                    │
         ┌──────────────────────────────────────────┼──────────────────────────────────────────┐
         │                                          │                                          │
┌────────▼─────────────────────────┐      ┌─────────▼────────────────────────┐      ┌──────────▼────────────────────────┐
│ 1. Data Ingest (app.py)          │      │ 2. RAG Lab (2_RAG_Lab.py)        │      │ 3. Kitchen Chat (3_Kitchen.py)    │
│ Live PDF upload & Web Crawler    │      │ Stage-by-stage benchmark sandbox │      │ Interactive Culinary Assistant    │
│ Streaming SSE ingestion telemetry│      │ Locked to Ground-Truth Benchmark │      │ Multi-dataset switch & Pantry Audit│
└──────────────────────────────────┘      └──────────────────────────────────┘      └───────────────────────────────────┘
```

1. **Ingest Tab** ([app.py](file:///c:/Users/aru91/OneDrive/Documents/Workspace_AI/Projects_Vscode/CognitiveKitchen/src/cognitive_kitchen/ui/app.py)): Ingest PDF cookbooks or crawl web recipe index pages with live SSE event progress.
2. **RAG Lab Sandbox** ([2_RAG_Lab.py](file:///c:/Users/aru91/OneDrive/Documents/Workspace_AI/Projects_Vscode/CognitiveKitchen/src/cognitive_kitchen/ui/pages/2_RAG_Lab.py)): Benchmark and compare strategies across all 6 stages live against the hand-verified Ground-Truth Golden Dataset (`load_gds_ingested()`). Lock your winning pipeline for production deployment.
3. **Kitchen Culinary Assistant** ([3_Kitchen.py](file:///c:/Users/aru91/OneDrive/Documents/Workspace_AI/Projects_Vscode/CognitiveKitchen/src/cognitive_kitchen/ui/pages/3_Kitchen.py)): Chat with your locked production pipeline.
   - **Active Dataset Selector**: Seamlessly switch context between the benchmark GDS PDF and custom uploaded user cookbooks.
   - **Pantry Audit Engine**: Ask *"Can I make this with what I have in my kitchen?"* with automatic fallback recipe card evaluation.

---

## 📊 Ground-Truth Golden Dataset & Benchmark Design

The evaluation benchmark lives in [data/golden_dataset.json](file:///c:/Users/aru91/OneDrive/Documents/Workspace_AI/Projects_Vscode/CognitiveKitchen/data/golden_dataset.json). It contains 50 hand-verified recipes extracted from the sample cookbook and 229 evaluation queries spanning 7 intent families.

```
Metric Matrix Tier        Required Benchmark Data         Evaluated Metrics
───────────────────────── ─────────────────────────────── ────────────────────────────────────────────────
Tier 1: Document-Only     Ingested Raw Data               Chunk Purity, Self-Sufficiency, Latency, Cost
Tier 2: Unlabeled Queries + data/questions.txt            Faithfulness, Relevancy, Cookability, Abstention,
                                                          Diversity, Constraint Compliance
Tier 3: Labeled Benchmark + data/golden_dataset.json      Hit@K, Recall@K, MAP, K@90, Set Precision/Recall
```

> [!NOTE]
> **Zero Benchmark Fitting**: The ingestion, indexing, and retrieval build pipelines never inspect `golden_dataset.json`. Ground truth is strictly isolated for evaluation. Refer to [docs/golden-dataset.md](docs/golden-dataset.md) for details on building custom truth sets.

---

## ⚡ Quickstart & Operations Runbook

### Step 1: Environment Setup
```bash
# Clone repository
git clone https://github.com/aru911-gethu/CognitiveKitchen.git
cd CognitiveKitchen

# Install dependencies & CLI entrypoints
uv sync
uv run playwright install chromium

# Create environment file
cp .env.example .env
```
*Configure `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and `OPENAI_API_KEY` inside `.env`.*

### Step 2: Build Vocabulary & Neo4j Knowledge Graph
```bash
# Process canonical ingredients (outputs data/eval/ingredient_map.json)
uv run ck-vocab

# Rebuild Neo4j Knowledge Graph from data/ingested/ (supports --dry-run)
uv run ck-graph
```

### Step 3: Launch Services
Open two terminal instances:

```bash
# Terminal 1: FastAPI REST & SSE Service (http://127.0.0.1:8010)
uv run ck-api

# Terminal 2: Streamlit Console UI (http://127.0.0.1:8501)
uv run ck-ui
```
API Documentation: `http://127.0.0.1:8010/docs`.

### Step 4: Verification Suite
Run all 60 automated unit tests:
```bash
uv run pytest tests -q
```

---

## 🔒 Security & Edge Deployment

- **SSRF Prevention**: The Playwright crawler rejects non-HTTP schemes and resolves destination IPs against loopback, link-local, private, and reserved subnets before fetching.
- **Robots.txt Adherence**: Checked dynamically per host; disallowed URLs are skipped cleanly.
- **Upload Guards**: File uploads restricted to `.pdf` under 80 MB.
- **Zero GPU Requirement**: Designed for CPU-bound environments (12 CPU cores, 16 GB RAM) with token, cost, and latency telemetry tracked per query stage in [src/cognitive_kitchen/rag/telemetry.py](file:///c:/Users/aru91/OneDrive/Documents/Workspace_AI/Projects_Vscode/CognitiveKitchen/src/cognitive_kitchen/rag/telemetry.py).

---

## 📚 Related Academic Literature

1. **Semantic Collapse in Vector Search**: *Negation is Not Semantic: Diagnosing Dense Retrieval Failure Modes* ([arXiv:2603.17580](https://arxiv.org/abs/2603.17580)) and *Exclusion-Sensitive Penalization for Negative-Constraint Retrieval* ([arXiv:2608.30130](https://arxiv.org/html/2608.30130v2)).
2. **Structure-Aware Chunking Efficiency**: *Evaluating Chunking Strategies for RAG in Enterprise Documents* ([arXiv:2603.24556](https://arxiv.org/abs/2603.24556)) and *Is Semantic Chunking Worth the Computational Cost?* ([arXiv:2410.13070](https://arxiv.org/html/2410.13070v1)).
3. **Lost-in-the-Middle Context Effects**: Liu et al., 2023 (*Lost in the Middle: How Language Models Use Long Contexts*).
