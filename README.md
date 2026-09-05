# CognitiveKitchen

An advanced, production-grade RAG recipe assistant engineered iteratively from a core MVP into an intelligent, multi-modal culinary system featuring semantic retrieval, HyDE query expansion, fridge-vision inventory, and cloud deployment.

---

## Tech Stack

- **Orchestration:** LangChain, Python
- **API Framework:** FastAPI, Uvicorn
- **Vector Database:** FAISS (Local)
- **Embeddings:** HuggingFace (all-MiniLM-L6-v2)
- **Web Scraping:** Playwright, BeautifulSoup
- **Frontend / UI:** Streamlit
- **Containerization & Cloud:** Docker, AWS

---

## Project Structure & Module Breakdown

CognitiveKitchen uses a modular, scalable architecture separating core ingestion/retrieval engines from advanced enhancements and production configs:

- **`app.py` (Root):** The primary Streamlit frontend entry point providing an interactive chat interface and ingestion dashboards.
- **`data/`:** Centralized storage for raw JSON recipe vaults (`raw_recipes/`) and unstructured document assets (`raw_pdfs/`).
- **`modules/core_engine/`:** Houses the foundational pipelines—FastAPI server (`main.py`), automated web scrapers (`crawler.py`, `ingest.py`), PDF text extractors (`pdf_ingest.py`), and baseline vector store vectorization/retrieval logic.
- **`modules/enhancements/`:** Reserved for advanced retrieval upgrades (semantic chunking, Maximal Marginal Relevance) and query engineering (HyDE, query expansion, and multi-modal vision).
- **`modules/production/`:** Houses containerization scripts (Dockerfile) and cloud deployment configurations for AWS.

---

## Architectural Evolution & Development Roadmap

- **[x] Phase 1: Core Data Ingestion & API Architecture**
  - **Multi-Source Ingestion Pipeline:** Built robust parsers to ingest user-shared URLs and raw PDFs, normalizing unstructured culinary data into clean JSON schemas.
  - **Intelligent Playwright Routing:** Leveraged Playwright to auto-recognize page structures, dynamically diverting execution between bulk category index pages and live single-recipe pages.
  - **FastAPI Live Streaming:** Engineered a high-performance FastAPI backend featuring NDJSON streaming endpoints to deliver real-time progress updates and recipe fetch stats directly to the UI.
- **[ ] Phase 2: Core Naive RAG & Chat UI**
  - Implementing local vector storage (FAISS) powered by lightweight HuggingFace embeddings (`all-MiniLM-L6-v2`).
  - Expanding the Streamlit interface into a real-time conversational assistant connected to the recipe vault.
- **[ ] Phase 3: Advanced Retrieval & Query Engineering**
  - Upgrading semantic search diversity using Maximal Marginal Relevance (MMR) scoring and optimized chunking.
  - Introducing Hypothetical Document Embeddings (HyDE) to bridge casual human cravings with structured recipe schemas.
- **[ ] Phase 4: Multi-Modal Vision & Constraints**
  - Integrating computer vision to parse fridge-inventory photo uploads and recommend recipes based on available ingredients.
  - Implementing deterministic dietary preference and allergen filtering.
- **[ ] Phase 5: Production Hardening & Cloud Deployment**
  - Containerizing the full stack via Docker and deploying to AWS for production-grade scalability.

---

## Quick Start

1. **Clone repository:** `git clone https://github.com/aru911-gethu/CognitiveKitchen.git`
2. **Navigate to folder:** `cd CognitiveKitchen`
3. **Create virtual environment:** `python -m venv .venv`
4. **Activate environment on Windows:** `.venv\Scripts\Activate`
5. **Install dependencies:** `pip install -r requirements.txt`
6. **Run FastAPI backend:** `uvicorn modules.core_engine.main:app --reload`
7. **Run Streamlit app (in a separate terminal):** `streamlit run app.py`

---

## Author
Built by [Arun](https://github.com/aru911-gethu) as an exploration into production-ready RAG architectures.