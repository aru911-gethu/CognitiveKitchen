<div align="center">

# 🍳 CognitiveKitchen

*A production-grade RAG recipe assistant — engineered iteratively from a naive MVP into an intelligent, multi-modal culinary AI system with graph reasoning, dietary constraints, and voice interaction.*

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-Lightning%20Fast-DE5FE9?style=for-the-badge&logo=python&logoColor=white)](https://github.com/astral-sh/uv)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Neo4j](https://img.shields.io/badge/Neo4j-4581C3?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com/)

**`Module 1 ✅ Complete`** · **`Module 2 🔧 In Progress`** · **`Module 3 🔜 Planned`** · **`Module 4 🔜 Planned`**

[Releases](https://github.com/aru911-gethu/CognitiveKitchen/releases) · [LinkedIn Build Journey](https://linkedin.com/in/yourprofile)

</div>

---

## 🧠 What is CognitiveKitchen?

CognitiveKitchen is an **end-to-end RAG (Retrieval-Augmented Generation) system** built around a recipe assistant — but the real story is the engineering journey. Each module introduces progressively advanced AI/ML techniques, evolving from a basic keyword-search chatbot into a production-deployed, multi-modal AI application with graph reasoning, dietary constraint satisfaction, voice interaction, and API-based ordering.

This repository is designed as an **iterative case study in applied AI engineering** — every phase is a self-contained milestone with its own architecture decisions, tradeoffs, and lessons learned.

---

## 🗺️ The Four-Module Journey

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  MODULE 1: Foundation                MODULE 2: Improve RAG                      │
│  ┌─────────────────────┐             ┌──────────────────────┐                   │
│  │ Phase 1: Ingestion  │             │ Phase 1: Semantic    │                   │
│  │   + Naive RAG       │             │   Chunking + MMR     │                   │
│  │   + BM25 + UI       │────────────▶│ Phase 2: Query       │                   │
│  │ Phase 2: LangChain  │             │   Engineering + HyDE │                   │
│  │   RAG + FAISS       │             │ Phase 3: Dense       │                   │
│  │ Phase 3: DeepEval   │             │   Vectors + Reranking│                   │
│  │   + Guardrails      │             └──────────┬───────────┘                   │
│  └─────────────────────┘                        │                               │
│                                                 ▼                               │
│  MODULE 4: Production-Ready          MODULE 3: Constraints                      │
│  ┌─────────────────────┐             ┌──────────────────────┐                   │
│  │ Phase 1: Self-Host  │             │ Phase 1: Neo4j Graph │                   │
│  │ Phase 2: Multi-Modal│◀────────────│   + Hybrid RAG       │                   │
│  │   (Voice + Vision)  │             │ Phase 2: Ingredient  │                   │
│  │ Phase 3: API-Based  │             │   & Mood Constraints │                   │
│  │   Ordering          │             └──────────────────────┘                   │
│  │ Phase 4: AWS Deploy │                                                        │
│  └─────────────────────┘                                                        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack (Evolving Per Module)

| Layer | Module 1 | Module 2 | Module 3 | Module 4 |
|-------|----------|----------|----------|----------|
| **Orchestration** | LangChain, LCEL | LangChain, LCEL | LangChain, LCEL | LangChain, LCEL |
| **Search / Retrieval** | BM25, FAISS similarity | MMR, HyDE, Cross-Encoder Reranking | Neo4j Graph + FAISS Hybrid | Hybrid + Constraints |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Semantic chunking-aware embeddings | Dense + Sparse vectors | Dense + Sparse |
| **LLM** | OpenAI `gpt-4o-mini` | OpenAI `gpt-4o-mini` | OpenAI `gpt-4o-mini` | OpenAI `gpt-4o` / `gpt-4o-mini` |
| **Vector DB** | FAISS (local) | FAISS (MMR-enabled) | Neo4j + FAISS | Neo4j + FAISS |
| **Evaluation** | DeepEval, LangSmith | — | — | — |
| **Web Scraping** | Playwright, BeautifulSoup | — | — | — |
| **PDF Extraction** | PyMuPDF (fitz) | — | — | — |
| **Frontend** | Streamlit | Streamlit | Streamlit | Streamlit + Voice UI |
| **Multi-Modal** | — | — | — | Whisper (voice), GPT-4o Vision |
| **Infrastructure** | Local (`uvicorn`) | Local | Local | Docker, Nginx, AWS (ECS Fargate) |
| **Package Manager** | `uv` (Rust-based) | `uv` | `uv` | `uv` |

---

## 📁 Project Structure

```text
CognitiveKitchen/
├── app.py                              # Streamlit frontend — multi-tab UI
├── pyproject.toml                      # Dependencies (managed by uv)
├── test_rag.py                         # Standalone FAISS retrieval test
│
├── data/                               # Centralized local data storage
│   ├── raw_recipes/                    # Normalized JSON recipe vault
│   ├── raw_pdfs/                       # Raw PDF cookbook uploads
│   └── vector_store/                   # FAISS index (index.faiss + index.pkl)
│
└── modules/
    ├── core_engine/                    # ── MODULE 1: FOUNDATION ──
    │   ├── main.py                     # FastAPI server — streaming ingest + PDF upload endpoints
    │   ├── part_a_ingestion/           # Data Collection & Ingestion Pipeline
    │   │   ├── schemas.py              # Pydantic data contracts (Recipe, Ingredient)
    │   │   ├── crawler.py              # Intelligent routing crawler (category vs single page)
    │   │   ├── ingest.py               # Playwright scraper + BeautifulSoup HTML cleaner
    │   │   └── pdf_ingest.py           # PyMuPDF page-by-page text extraction
    │   └── part_b_rag/                 # RAG Intelligence & Chat Engine
    │       ├── vector_builder.py       # Text splitting + HuggingFace embedding + FAISS index
    │       └── retriever.py            # LCEL chain: retriever → prompt → GPT-4o-mini → answer
    │
    ├── enhancements/                   # ── MODULE 2: IMPROVED RAG ──
    │   ├── semantic_chunker.py         # 🔜 Embedding-aware chunking (split on meaning, not chars)
    │   ├── mmr_retriever.py            # 🔜 Maximal Marginal Relevance — diversified retrieval
    │   ├── query_decomposer.py         # 🔜 Query deconstruction into sub-queries
    │   ├── hyde_engine.py              # 🔜 Hypothetical Document Embeddings — LLM query expansion
    │   ├── sparse_dense_retriever.py   # 🔜 Hybrid sparse (BM25) + dense (embedding) retrieval
    │   └── reranker.py                 # 🔜 Cross-encoder reranking for precision scoring
    │
    ├── constraints/                    # ── MODULE 3: GRAPH + CONSTRAINTS ──
    │   ├── graph_store.py              # 🔜 Neo4j graph schema (recipes, ingredients, tags, moods)
    │   ├── hybrid_retriever.py         # 🔜 Neo4j graph query + FAISS vector — hybrid RAG
    │   ├── ingredient_filter.py        # 🔜 Deterministic ingredient-based constraint satisfaction
    │   └── mood_engine.py              # 🔜 Mood/preference-aware recipe recommendation
    │
    ├── production/                     # ── MODULE 4: PRODUCTION-READY ──
    │   ├── Dockerfile                  # 🔜 Multi-stage container build
    │   ├── docker-compose.yml          # 🔜 Multi-container orchestration
    │   ├── nginx.conf                  # 🔜 Reverse proxy routing
    │   ├── multimodal/                 # 🔜 Voice + Vision modules
    │   │   ├── voice_chat.py           # 🔜 Whisper STT + TTS for voice interaction
    │   │   └── fridge_vision.py        # 🔜 Image-to-ingredients via GPT-4o Vision
    │   ├── ordering/                   # 🔜 API-based ingredient ordering
    │   │   └── order_api.py            # 🔜 Constraint-aware grocery ordering via external APIs
    │   └── aws/                        # 🔜 AWS deployment configs
    │       ├── ecs-task-def.json       # 🔜 ECS Fargate task definition
    │       ├── cloudformation.yml      # 🔜 Infrastructure-as-code (VPC, ALB, ECS, S3, EFS)
    │       └── ecr-push.sh             # 🔜 Docker image push to ECR
    │
    └── evaluation/                     # ── MODULE 1 PHASE 3: EVAL & GUARDRAILS ──
        ├── deepeval_suite.py           # 🔜 DeepEval test suite (faithfulness, relevancy, hallucination)
        └── guardrails.py              # 🔜 Input/output guardrails for safety + quality
```

---

## 🚀 Module 1 — Foundation: Naive RAG + Ingestion + Evaluation

> **Status:** ✅ Complete · **Release:** [`v1.0`](https://github.com/aru911-gethu/CognitiveKitchen/releases/tag/v1.0)

This module builds the complete foundation — from raw data ingestion to a working conversational recipe assistant.

### Phase 1 — Ingestion Pipeline + Naive RAG + BM25 + UI

**What was built:** A multi-source data ingestion pipeline that scrapes recipe websites and extracts text from PDFs, storing everything in a normalized JSON vault. Streamlit provides the frontend UI, Playwright handles JavaScript-rendered pages, FastAPI serves the backend with NDJSON streaming for real-time progress.

**Key technical decisions:**

| Decision | Choice | Why |
|----------|--------|-----|
| Scraping engine | Playwright (not `requests`) | Recipe sites use JavaScript for content rendering — `requests.get()` only fetches raw HTML |
| HTML cleaning | BeautifulSoup `decompose()` | Surgically removes `<script>`, `<nav>`, `<footer>`, `<aside>` — strips ads/clutter while preserving recipe text |
| Backend streaming | NDJSON via `StreamingResponse` | Each scraped recipe sends a progress event in real-time — no waiting for the entire batch |
| Smart routing | Link analysis in `crawler.py` | Auto-detects category index pages (bulk scrape) vs. single recipe pages (direct scrape) |
| PDF extraction | PyMuPDF (`fitz`) | Fast, page-level text extraction with reading-order preservation |
| Data format | JSON vault (`data/raw_recipes/`) | Unified storage — both web scrapes (`{url, title, raw_text}`) and PDFs (`{type: "pdf_document", pages: [...]}`) land in the same directory |

**Architecture:**
```
User (Streamlit) ──POST──▶ FastAPI ──▶ Crawler (Playwright) ──▶ JSON Vault
                                    ──▶ PDF Ingest (PyMuPDF) ──▶ JSON Vault
                  ◀──NDJSON stream──
```

### Phase 2 — LangChain RAG: Vectorization + LCEL Chain + Chat

**What was built:** The intelligence layer — converts raw recipe text into searchable vector embeddings and builds an LCEL (LangChain Expression Language) chain that retrieves relevant chunks and generates grounded answers via GPT-4o-mini with token-by-token streaming.

**Key technical decisions:**

| Decision | Choice | Why |
|----------|--------|-----|
| Text splitting | `RecursiveCharacterTextSplitter` (800 chars, 120 overlap) | Preserves paragraph structure; overlap prevents information loss at chunk boundaries |
| Embedding model | HuggingFace `all-MiniLM-L6-v2` (384-dim, 22M params) | Lightweight, fast on CPU, free — sufficient for recipe-scale content |
| Vector store | FAISS (`IndexFlatL2`) | Exact nearest-neighbor search; optimal for <100K vectors; zero infrastructure overhead |
| LLM | GPT-4o-mini (temp=0.2, streaming) | Low temperature for factual grounding; streaming enables typing-effect UX |
| Chain composition | LCEL pipe (`|`) operator | Declarative: `{context: retriever | formatter, question: passthrough} | prompt | llm | parser` |
| Caching | `@st.cache_resource` | Embedding model + FAISS index load once into RAM; zero lag on subsequent queries |
| Title prepending | `f"Recipe Title: {title}\n..."` injected into `page_content` | Ensures every chunk carries its recipe identity after splitting — dramatically improves retrieval relevance |

**The RAG pipeline:**
```
User question
  │
  ├──▶ Embed query (all-MiniLM-L6-v2) ──▶ FAISS top-4 similarity search
  │                                              │
  │                                              ▼
  │                                        Format chunks with source attribution
  │                                              │
  └──▶ RunnablePassthrough (question)            │
                    │                             │
                    ▼                             ▼
              ChatPromptTemplate ◀── {context: "...", question: "..."}
                    │
                    ▼
              GPT-4o-mini (streaming, temp=0.2)
                    │
                    ▼
              StrOutputParser ──▶ Streamlit chat bubble (live ▌ cursor)
```

### Phase 3 — Evaluation + Guardrails + LangSmith Tracing

**What will be built:** Systematic evaluation of RAG quality using DeepEval metrics, input/output guardrails for safety, and LangSmith integration for full pipeline observability.

| Component | Tool | What It Measures / Does |
|-----------|------|------------------------|
| **Faithfulness** | DeepEval | Does the answer stay within the retrieved context? (Anti-hallucination) |
| **Answer Relevancy** | DeepEval | Does the answer actually address the user's question? |
| **Contextual Precision** | DeepEval | Are the retrieved chunks actually relevant to the question? |
| **Contextual Recall** | DeepEval | Did retrieval capture all the information needed to answer? |
| **Hallucination Detection** | DeepEval | Does the answer contain claims not grounded in any retrieved chunk? |
| **Input Guardrails** | Custom | Block prompt injection, off-topic queries, PII in input |
| **Output Guardrails** | Custom | Block harmful content, enforce dietary safety disclaimers |
| **Tracing** | LangSmith | End-to-end latency breakdown (embedding time, FAISS search, LLM generation), token usage, cost tracking |

---

## 🔬 Module 2 — Improve RAG: Advanced Retrieval & Query Engineering

> **Status:** 🔧 In Progress · **Release:** `v2.0` *(upcoming)*

This module systematically upgrades every component of the RAG pipeline — how text is chunked, how queries are processed, and how results are ranked.

### Phase 1 — Semantic Chunking + MMR

**Problem being solved:** Module 1's `RecursiveCharacterTextSplitter` splits on character count — it might cut a recipe's ingredient list in half at the 800-char boundary. And standard similarity search can return 4 near-identical chunks from the same recipe.

| Upgrade | Before (Module 1) | After (Module 2) |
|---------|-------------------|-------------------|
| **Chunking** | Split at 800 chars (character boundary) | Split on embedding similarity drops (meaning boundary) |
| **Retrieval** | `search_type="similarity"` (top-4 closest) | `search_type="mmr"` (top-4 balanced: relevance + diversity) |

**Semantic Chunking** computes embedding similarity between consecutive sentences. When similarity drops sharply (e.g., switching from "Ingredients" to "Instructions"), a new chunk boundary is created. Result: each chunk contains one coherent topic.

**MMR (Maximal Marginal Relevance)** retrieves 20 candidates, then iteratively selects 4 that are both relevant to the query AND different from each other. Prevents the "4 identical dal chunks" problem.

```python
# MMR retriever:
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.7}
)
# fetch_k=20 → FAISS finds 20 candidates
# lambda_mult=0.7 → 70% relevance weight, 30% diversity weight
# k=4 → final selection after MMR reranking
```

### Phase 2 — Query Engineering: Decomposition + HyDE

**Problem being solved:** Users ask vague or complex questions. *"What's a good South Indian breakfast that's quick and doesn't use rice?"* contains multiple constraints that a single embedding can't capture well.

**Query Decomposition** breaks complex queries into focused sub-queries:
```
Original: "Quick South Indian breakfast without rice"
  → Sub-query 1: "South Indian breakfast recipes"
  → Sub-query 2: "Quick breakfast under 20 minutes"
  → Sub-query 3: "Breakfast recipes without rice"
Each sub-query retrieves independently → results are merged and deduplicated
```

**HyDE (Hypothetical Document Embeddings)** asks the LLM to imagine what a perfect answer would look like, then uses that synthetic document as the search query:
```
User: "something spicy for breakfast"
  → LLM generates: "Spicy Masala Dosa Recipe. Ingredients: dosa batter,
     green chillies, red chilli powder, onions, curry leaves..."
  → Embed this HYPOTHETICAL document (not the vague original query)
  → FAISS search finds real recipes matching the synthetic recipe vocabulary
```

### Phase 3 — Sparse + Dense Vectors + Cross-Encoder Reranking

**Problem being solved:** Pure semantic search (dense vectors) misses exact keyword matches. Searching for "Rava Idli" might return "Semolina Steamed Cake" (semantically similar) but miss the exact document titled "Rava Idli Recipe."

**Hybrid Retrieval** combines:
- **Sparse vectors (BM25):** Keyword-based — excels at exact term matching ("Rava Idli" → "Rava Idli")
- **Dense vectors (embeddings):** Semantic — excels at meaning matching ("sooji steamed breakfast" → "Rava Idli")
- Results are fused using **Reciprocal Rank Fusion (RRF)** — combining the best of both worlds

**Cross-Encoder Reranking** adds a precision layer:
```
Step 1: Hybrid retrieval returns top-20 candidates (fast, approximate)
Step 2: Cross-encoder scores each (query, candidate) pair with full attention (slow, precise)
Step 3: Top-4 after reranking are passed to the LLM
```

---

## 🕸️ Module 3 — Constraints: Graph Reasoning + Dietary Filters

> **Status:** 🔜 Planned · **Release:** `v3.0` *(upcoming)*

This module introduces structured knowledge (graph database) alongside unstructured retrieval (vectors), enabling constraint-based recipe recommendation.

### Phase 1 — Neo4j Graph + FAISS Vector: Hybrid RAG

**Why a graph?** Vector search finds *similar* recipes — but it can't reason about relationships. A graph database encodes structured knowledge: *"Dosa → requires → Rice Flour"*, *"Rice Flour → is_substitute_for → Rava"*, *"Dosa → belongs_to → South Indian Breakfast"*, *"Dosa → dietary_tag → Vegan"*.

**Neo4j Graph Schema:**
```
(:Recipe)-[:REQUIRES]->(:Ingredient)
(:Recipe)-[:BELONGS_TO]->(:Category)
(:Recipe)-[:HAS_TAG]->(:DietaryTag)
(:Ingredient)-[:SUBSTITUTE_FOR]->(:Ingredient)
(:Recipe)-[:PAIRS_WITH]->(:Recipe)
(:Recipe)-[:MOOD]->(:MoodTag)
```

**Hybrid RAG Pipeline:**
```
User: "Vegan South Indian breakfast without coconut"
  │
  ├──▶ Neo4j Cypher query:
  │    MATCH (r:Recipe)-[:HAS_TAG]->(:DietaryTag {name: "vegan"})
  │    WHERE NOT (r)-[:REQUIRES]->(:Ingredient {name: "coconut"})
  │    AND (r)-[:BELONGS_TO]->(:Category {name: "South Indian Breakfast"})
  │    RETURN r
  │    → Returns: [Rava Idli, Upma, Pesarattu, ...]
  │
  └──▶ FAISS vector search: "vegan South Indian breakfast"
       → Returns: [Dosa, Idli, Upma, Pongal, ...]

  Merge + Rerank → [Rava Idli, Upma, Pesarattu, Idli]
  → LLM generates final answer using merged context
```

### Phase 2 — Ingredient Constraints + Mood Preferences

**Ingredient-based filtering:** User specifies what's available in their kitchen or what they want to avoid. The graph query becomes a constraint satisfaction problem:
```
Available: ["rice flour", "urad dal", "oil", "salt"]
Avoid: ["coconut", "ghee"]
→ Cypher: MATCH (r:Recipe) WHERE ALL ingredients in r.required ARE IN available_list
          AND NONE of r.required ARE IN avoid_list
```

**Mood-based recommendation:** Recipes are tagged with mood/occasion nodes (`:MoodTag {name: "comfort food"}`, `"quick weeknight"`, `"festive"`, `"light and healthy"`). User says *"I want something comforting"* → graph traversal filters by mood → vector search ranks within the filtered set.

---

## 🚀 Module 4 — Production-Ready: Self-Host, Multi-Modal, AWS

> **Status:** 🔜 Planned · **Release:** `v4.0` *(upcoming)*

This module transforms CognitiveKitchen from a local development project into a deployed, multi-modal product.

### Phase 1 — Self-Hosting (Docker + Nginx)

**What gets built:** Multi-container Docker stack with `docker-compose.yml` orchestrating: FastAPI backend, Streamlit frontend, Neo4j graph database, and Nginx reverse proxy.

```yaml
# docker-compose.yml (planned):
services:
  backend:
    build: { context: ., target: backend }
    ports: ["8000:8000"]
    volumes: ["./data:/app/data"]
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}

  frontend:
    build: { context: ., target: frontend }
    ports: ["8501:8501"]

  neo4j:
    image: neo4j:5-community
    ports: ["7474:7474", "7687:7687"]
    volumes: ["neo4j_data:/data"]

  nginx:
    image: nginx:alpine
    ports: ["80:80"]
    volumes: ["./production/nginx.conf:/etc/nginx/nginx.conf"]
```

### Phase 2 — Multi-Modal: Voice Chat + Fridge Vision

**Voice Chat:** Integrates OpenAI Whisper (speech-to-text) and a TTS engine so users can talk to CognitiveKitchen hands-free while cooking. The pipeline: `Microphone → Whisper STT → RAG chain → LLM answer → TTS → Speaker`.

**Fridge Vision:** Users upload a photo of their fridge or pantry. GPT-4o Vision identifies visible ingredients, then feeds the ingredient list into the constraint-based retriever from Module 3:
```
Fridge photo → GPT-4o Vision → ["tomatoes", "onions", "rice", "yogurt"]
  → Module 3 ingredient filter → matching recipes
  → RAG chain → "You can make Tomato Rice, Curd Rice, or Onion Raita!"
```

### Phase 3 — API-Based Ordering with Constraints

**What gets built:** After CognitiveKitchen recommends a recipe and identifies missing ingredients (using the graph's `REQUIRES` relationships vs. user's available ingredients), it offers to order them via external grocery APIs (e.g., Instacart, BigBasket, or a mock API for demo). Constraint-aware: respects dietary restrictions, budget limits, and preferred brands.

```
Recipe: Masala Dosa
  Required: [rice flour, urad dal, fenugreek, oil, potatoes, onions, spices]
  Available: [rice flour, oil, salt]
  Missing: [urad dal, fenugreek, potatoes, onions, spices]
  → "Order these 5 items?" → API call → Order placed
```

### Phase 4 — AWS Cloud Deployment

**Infrastructure:** ECS Fargate (serverless containers) behind an Application Load Balancer. CloudFormation IaC for the full stack: VPC, subnets, ALB, ECS cluster, ECR image registry, S3 for data persistence, EFS for shared vector store. CloudWatch dashboards for latency, error rates, and LLM token cost monitoring.

---

## ⚡ Quick Start Guide

> **Powered by [uv](https://github.com/astral-sh/uv)** — Rust-based package manager for lightning-fast dependency resolution.

### Prerequisites

- Python 3.12+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- *(Optional)* [LangSmith API key](https://smith.langchain.com/) for tracing

### Setup

```bash
# 1. Clone
git clone https://github.com/aru911-gethu/CognitiveKitchen.git
cd CognitiveKitchen

# 2. Install dependencies (creates .venv automatically)
uv sync

# 3. Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate
# macOS/Linux:
# source .venv/bin/activate

# 4. Install Playwright browser (one-time)
uv run playwright install chromium

# 5. Configure environment
# Create .env file in project root:
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-key-here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_your-key-here
EOF

# 6. Launch backend (Terminal 1)
uvicorn modules.core_engine.main:app --reload

# 7. Launch frontend (Terminal 2)
streamlit run app.py

# 8. Build vector index (after ingesting recipes via UI)
python -m modules.core_engine.part_b_rag.vector_builder
```

> **API docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
> **Streamlit UI:** [http://localhost:8501](http://localhost:8501)

---

## 🔍 How the RAG Pipeline Works

```
User: "How long should I pressure cook the dal?"
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  1. EMBED — all-MiniLM-L6-v2 encodes the question   │
│     into a 384-dimensional vector                    │
│                                                      │
│  2. RETRIEVE — FAISS finds the 4 most semantically   │
│     similar chunks from the recipe vault              │
│                                                      │
│  3. FORMAT — Chunks get source attribution headers    │
│     ("--- Source: Dal Tadka Recipe ---")              │
│                                                      │
│  4. PROMPT — System message + formatted context       │
│     + user question → ChatPromptTemplate              │
│                                                      │
│  5. GENERATE — GPT-4o-mini reads context and produces │
│     a grounded answer (temperature=0.2)               │
│                                                      │
│  6. STREAM — Tokens flow to Streamlit chat UI with    │
│     live typing cursor (▌)                            │
└─────────────────────────────────────────────────────┘
  │
  ▼
Assistant: "Based on the recipes in your vault, cook the toor
dal for 3-4 whistles on medium flame until soft..."
```

---

## 🧪 Testing

```bash
# Test FAISS retrieval (no LLM, no API costs):
python test_rag.py

# Run DeepEval suite (Module 1 Phase 3):
# python -m modules.evaluation.deepeval_suite   # 🔜 coming soon
```

---

## 📋 Release History

| Release | Module | Phase | Highlights |
|---------|--------|-------|-----------|
| `v1.0` | Module 1 | Phase 1-2 | Playwright crawler, PDF ingestion, FAISS vector store, LCEL RAG chain, Streamlit chat UI |
| `v1.1` | Module 1 | Phase 3 | 🔜 DeepEval metrics, guardrails, LangSmith tracing |
| `v2.0` | Module 2 | Phase 1-3 | 🔜 Semantic chunking, MMR, HyDE, cross-encoder reranking |
| `v3.0` | Module 3 | Phase 1-2 | 🔜 Neo4j graph, hybrid RAG, ingredient/mood constraints |
| `v4.0` | Module 4 | Phase 1-4 | 🔜 Docker, multi-modal (voice + vision), API ordering, AWS |

---

## 👤 Author

Built with precision by **Arun** ([@aru911-gethu](https://github.com/aru911-gethu)) — an exploration into production-ready, modular RAG architectures from first principles.

---

<div align="center">

*Each module is a self-contained milestone with its own [LinkedIn write-up](https://linkedin.com/in/yourprofile) and [GitHub release](https://github.com/aru911-gethu/CognitiveKitchen/releases).*

*If this project helped you understand RAG pipelines, consider giving it a ⭐!*

</div>
