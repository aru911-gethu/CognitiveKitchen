# CognitiveKitchen

An advanced, production-grade RAG recipe assistant engineered iteratively from a core MVP into an intelligent, multi-modal culinary system featuring semantic retrieval, HyDE query expansion, fridge-vision inventory, and cloud deployment.

---

## Architectural Evolution and Feature Roadmap

- Core Engine (Phase 1 and 2): Automated data structuring using FastAPI and Playwright, paired with a baseline RAG pipeline using local FAISS vector indexing and Streamlit chat.
- Enhancement 1 (Advanced Retrieval): Upgraded chunking strategies and Maximal Marginal Relevance scoring for high-diversity search results.
- Enhancement 2 (Query Engineering): Bridging human cravings and recipe schemas using HyDE and query expansion.
- Enhancement 3 (Multi-Modal Vision): Adding fridge-inventory photo uploads and dietary preference constraints.
- Enhancement 4 (Production): Containerization via Docker and deployment on AWS.

---

## Tech Stack

- Orchestration: LangChain, Python
- Vector Database: FAISS (Local)
- Embeddings: HuggingFace (all-MiniLM-L6-v2)
- Web Scraping: Playwright
- Frontend: Streamlit
- Deployment: Docker, AWS

---

## Quick Start

1. Clone repository: git clone [https://github.com/aru911-gethu/CognitiveKitchen.git](https://github.com/aru911-gethu/CognitiveKitchen.git)
2. Navigate to folder: cd CognitiveKitchen
3. Create virtual environment: python -m venv .venv
4. Activate environment on Windows: .venv\Scripts\Activate
5. Install dependencies: pip install -r requirements.txt
6. Run app: streamlit run app.py

---

## Author
Built by Arun ([https://github.com/aru911-gethu](https://github.com/aru911-gethu)) as an exploration into production-ready RAG architectures.