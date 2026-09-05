Markdown

<div align="center">

# 🍳 CognitiveKitchen

An advanced, production-grade RAG recipe assistant engineered iteratively from a core MVP into an intelligent, multi-modal culinary system featuring semantic retrieval, HyDE query expansion, fridge-vision inventory, and cloud deployment.

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-green.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-VectorDB-orange.svg)](https://github.com/facebookresearch/faiss)

</div>

---

## 🛠️ Tech Stack

- **Orchestration & Frameworks:** Python, LangChain, LCEL
- **Observability:** LangSmith (Tracing, latency, and token monitoring)
- **Backend API:** FastAPI, Uvicorn, NDJSON Streaming
- **Vector Database:** FAISS (Local Vector Index)
- **Embeddings Model:** HuggingFace (`all-MiniLM-L6-v2`)
- **LLM Engine:** OpenAI (`gpt-4o-mini`)
- **Web Automation & Scraping:** Playwright, BeautifulSoup
- **Frontend UI:** Streamlit
- **Infrastructure & Cloud:** Docker, AWS

---

## 📁 Project Structure & Module Breakdown

CognitiveKitchen implements a clean, modular architecture separating core data ingestion and UI controls (Part A) from semantic RAG intelligence and chat execution (Part B):

```text
CognitiveKitchen/
├── app.py                      # Streamlit frontend entry point (Chat UI & Dashboards)
├── data/                       # Centralized local storage
│   ├── raw_recipes/            # Structured JSON recipe vault
│   └── raw_pdfs/               # Unstructured document assets & PDF books
└── modules/
    ├── core_engine/            
    │   ├── part_a_ingestion/   # Data Collection & UI Control Plane
    │   │   ├── crawler.py      # Playwright intelligent routing scraper
    │   │   └── pdf_ingest.py   # PyMuPDF document loading pipeline
    │   └── part_b_rag/         # RAG Intelligence & Chatbot Engine
    │       ├── vector_builder.py # Recursive text splitter + HuggingFace + FAISS
    │       └── retriever.py    # LCEL chain + Prompt templates + LangSmith + OpenAI
    ├── enhancements/           # Advanced retrieval (Semantic chunking, MMR) & Query engineering (HyDE)
    └── production/             # Docker containerization & AWS deployment configs

🚀 Architectural Evolution & Development Roadmap

    [x] Phase 1: Modular Architecture & Multi-Source Ingestion Design

        Modular Separation: Divided core architecture between Part A (Ingestion & UI Controls) and Part B (RAG Intelligence & Chatbot).

        Multi-Source Ingestion Pipeline: Planned parsers to ingest user-shared URLs and raw PDF document uploads, normalizing unstructured culinary text into clean schemas.

        Intelligent Playwright Routing: Leveraged Playwright to auto-recognize page layouts, dynamically diverting execution between bulk category index listings and live single-recipe pages.

    [ ] Phase 2: Part A - Ingestion Pipeline & UI Controls

        Implementing Playwright scraping and PyMuPDF document loading managed seamlessly via Streamlit inputs.

    [ ] Phase 3: Part B - Vectorization, LCEL RAG, & LangSmith Chatbot

        Vector Database Indexing: Implementing local FAISS vector indexing powered by lightweight HuggingFace embeddings (all-MiniLM-L6-v2) and recursive text splitting.

        Conversational Assistant & Tracing: Integrating LangSmith observability with an OpenAI (gpt-4o-mini) LCEL chat chain connected to the Streamlit UI.

    [ ] Phase 4: Advanced Retrieval & Query Engineering

        Maximal Marginal Relevance (MMR): Upgrading semantic search diversity to prevent redundant recipe results.

        Hypothetical Document Embeddings (HyDE): Introducing query expansion to bridge casual human cravings with technical recipe schemas.

    [ ] Phase 5: Multi-Modal Vision & Constraints

        Fridge-Vision Inventory: Integrating computer vision to parse uploaded fridge-inventory photos and recommend recipes based on available ingredients.

        Dietary Guardrails: Implementing deterministic dietary preference and allergen filtering.

    [ ] Phase 6: Production Hardening & Cloud Deployment

        Containerization: Packaging the entire multi-container stack using Docker.

        Cloud Orchestration: Deploying the application pipeline onto AWS for production-grade scalability.

⚡ Quick Start Guide

    Clone the repository:
    Bash

git clone [https://github.com/aru911-gethu/CognitiveKitchen.git](https://github.com/aru911-gethu/CognitiveKitchen.git)
cd CognitiveKitchen

Create and activate a virtual environment:
Bash

python -m venv .venv
.venv\Scripts\Activate  # On Windows PowerShell

Install dependencies:
Bash

pip install -r requirements.txt

Launch the FastAPI backend:
Bash

uvicorn modules.core_engine.main:app --reload

Launch the Streamlit frontend (in a separate terminal tab):
Bash

    streamlit run app.py

👤 Author

Built with precision by Arun (@aru911-gethu) as an exploration into production-ready, modular RAG architectures.