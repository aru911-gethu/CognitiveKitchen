# 🍳 Cognitive Kitchen — 8-Part LinkedIn Series (Production Edition)

**Author**: Arun Nakkeeran  
**GitHub**: https://github.com/aru911-gethu/CognitiveKitchen  
**Live App**: https://techideas.tech/  
**Media Assets**: Stored in `docs/assets/` (Live full-res screenshots & demo video)

---

## 📌 Episode 1: The 9:30 PM Tiffin Crisis

**Media Attachment**:  
🎥 Video Walkthrough: `docs/assets/live_demo_walkthrough.webm`  
📸 Image: `docs/assets/1_ingest_landing.png` (Live Console: 184 recipes · 1,746 graph nodes · 158 ingredients)

---

Every night at 9:30 PM, right after dinner, my house turns into a negotiation room.

*"What are we packing for tomorrow morning's school tiffin and office lunch?"* 🍱

The constraints are brutal:
→ Swiggy doesn't exist at 7:00 AM when the school bus rolls up at 8:00 AM.
→ The kids have declared war on monotony (*"Not Idli or Dosa AGAIN!"* 🙈).
→ We inspect the pantry, debate for 20 minutes, and inevitably default to the **same Idli/Dosa batter for the 4th time this week.**

So I did what any product person would do.

I asked ChatGPT. 🤖

It confidently recommended Paneer Tikka Wraps.
My pantry had rava, mustard seeds, and curry leaves.

Then I tried building a standard vector RAG app.
It failed even worse:

🔴 **The Allergy Trap (Semantic Collapse)**: A vector search for *"nut-free breakfast"* placed the query right next to Cashew Upma and Almond Milk. Vectors measure topic proximity — they have zero clue what *WITHOUT* means. The better your ranker, the more confidently it sends you to the ER. 🚑

🔴 **The Phantom Pantry**: Recommended elaborate dishes requiring 8 extra ingredients I didn't own. Zero inventory grounding.

🔴 **Relational Blindness**: *"What's the ONE ingredient I'm missing for Rava Upma?"* → silence. Vector similarity cannot calculate set-differences.

---

So I built **Cognitive Kitchen** — and created its biggest breakthrough: **The RAG Lab Sandbox** 🧪

Here is the twist: The real product isn't a cooking chatbot.

It's a **domain-agnostic RAG test workbench** where you can benchmark 8 chunkers, 9 retrievers, and 4 generation strategies against ground truth *before* locking your production pipeline.

The cooking domain? That's just the first proof point.

The workflow: **1 · Ingest → 2 · Experiment → 3 · Lock what wins → 4 · Chat.**

And the technical headline:
A **1.5B open-source model** (`Qwen2.5-1.5B-Instruct`) running on CPU — paired with a **Neo4j** knowledge graph + **FAISS** vector index — outperformed massive proprietary models on safety, accuracy, and latency.

🛠️ **The Marquee Stack**:
**Docker** · **Streamlit** · **FastAPI (SSE)** · **Neo4j Aura** · **FAISS** · **Qwen 1.5B (CPU)** · **DeepEval** · **LangSmith**

Check out the live app running on VPS:
🌐 https://techideas.tech/
💻 https://github.com/aru911-gethu/CognitiveKitchen

Over the next 7 posts, I'm opening the kitchen hood — benchmarks, failure modes, and lessons from every stage.

👇 What's your family's 9:30 PM tiffin debate? Drop your go-to quick breakfast in the comments!

---
#CognitiveKitchen #RAG #GraphRAG #AIProduct #OpenSource #GenAI #Qwen #Neo4j #FAISS #FastAPI #Streamlit #Docker #BuildInPublic

---
---

## 📌 Episode 2: The RAG Lab Sandbox

**Media Attachment**:  
📸 Image: `docs/assets/2_rag_lab.png` (RAG Lab: 8 chunkers · 9 retrievers · 4 strategies · 1,746 nodes)

---

Here's a dirty secret in AI product development:

**Most RAG benchmarks are cheating.** 🛑

If your chunkers, retrievers, or vocabulary scripts have ever *seen* your test queries — your AI isn't smart. It's just memorizing its own answer key.

In Ep. 1, I introduced the RAG Lab Sandbox as Cognitive Kitchen's core breakthrough. Today — how it enforces honest benchmarks.

I built a 50-recipe / 229-query hand-verified **Golden Data Set** (`golden_dataset.json`) covering 7 intent families, and locked it behind 3 strict evaluation tiers:

---

**The 3-Tier Evaluation Framework** 🧪

📌 **Tier 1 — Document & Extraction Health**
Measures raw document parsing quality before any LLM touches it.
→ Tracks Chunk Purity, Self-Sufficiency, System Latency ($s/query$), and Token Economics ($\$/query$) via our built-in telemetry ledger.

📌 **Tier 2 — Unlabeled Production Queries**
Tests real user questions *without* requiring manual labels.
→ Evaluated using **DeepEval** with `gpt-4o-mini` as an offline judge: Faithfulness, Answer Relevancy, G-Eval Cookability, and Abstention Rate. Traced end-to-end via **LangSmith**.

📌 **Tier 3 — Golden Data Set Benchmark**
Candidate pipelines scored against the hand-verified ground truth.
→ Hit@K, Recall@K, MAP, K@90, and Set Precision/Recall rendered live in the **Streamlit** console.

---

**The Non-Negotiable Rule** 🔒

No ingestion script, no vocabulary parser (`ck-vocab`), no graph builder (`ck-graph`), no indexing job is EVER allowed to inspect `golden_dataset.json`.

Ground truth stays strictly isolated.
Evaluation scores flow through **LangSmith** tracing.
Every experiment is reproducible.

The RAG Lab replaced *"this feels pretty good"* with metric-driven go/no-go quality gates.

Tomorrow in Ep. 3: I benchmarked 8 chunking strategies. The most expensive one lost. 📄✂️

---
#CognitiveKitchen #RAG #DeepEval #LangSmith #AIProduct #LLMOps #GoldenDataSet #Streamlit #BuildInPublic

---
---

## 📌 Episode 3: Chunking — The Expensive One Lost

**Media Attachment**:  
📸 Image: `docs/assets/2_rag_lab_expanded.png` (RAG Lab: Stage 2 Chunking benchmark selector)

---

"Semantic Chunking" is the default recommendation in almost every RAG tutorial.

So I benchmarked it against 7 alternatives.

It lost. 📉

---

In Stage 2 of Cognitive Kitchen, I tested **8 chunking strategies** across 184 recipe documents:

- **Structure-Aware (Full Recipe)** (`recipe`)
- **Recipe Sections (Ingredients & Method Split)** (`recipe_sections`)
- **Recursive Character Split** (`recursive`)
- **Semantic Adjacent (Embedding Distance Split)** (`semantic_adjacent`)
- **Markdown Header Split** (`markdown_header`)
- **Fixed Window Character Split** (`character` / `naive`)
- **Token Window Split** (`token`)
- **Sentence Boundary Split** (`sentence`)

**What Semantic Chunking (`semantic_adjacent`) did wrong:**

It required continuous embedding passes (running via `sentence-transformers`) — expensive — and frequently straddled recipe boundaries.

One chunk contained the tail of a Biryani recipe and the head of a Mutton Curry.

The LLM's answer to *"how much butter is needed?"* — it averaged butter quantities across 5 completely unrelated dishes in the same chunk! 🧈🙈

---

**The fix: Structure-Aware Recipe Chunking (`recipe`)**

Instead of embedding-based splitting, I used document boundary cues:
→ PDF page structures parsed by **PyMuPDF** & **pypdf**
→ Web recipe schema crawled by **Playwright**, extracting JSON-LD via BeautifulSoup

Each chunk = exactly one complete, self-contained recipe. Always.

📊 **The Scorecard:**

| Strategy | Recall | Self-Sufficiency |
|----------|--------|-----------------|
| **Structure-Aware (Full Recipe)** | **0.991** | **1.000** |
| Recursive Character Split | 0.745 | 0.932 |
| Semantic Adjacent Split | 0.612 | 0.780 |

---

**The product lesson:**

Before spending your compute budget on heavy semantic splitters, check if your document's own structure gives you better accuracy for free.

Ep. 4 tomorrow: Our top-scoring retriever had an allergy safety defect that would have sent someone to the hospital. 🚑

---
#CognitiveKitchen #RAG #Chunking #Playwright #PyMuPDF #VectorSearch #AIProduct #BuildInPublic

---
---

## 📌 Episode 4: The Top Retriever Failed Safety

**Media Attachment**:  
📸 Image: `docs/assets/3_kitchen_nut_free_result.png` (Live App: Allergy test query returning 0 sources / refusal to hallucinate)

---

Consider this query at 9:30 PM:

*"I'm avoiding nuts. What quick breakfast can I pack for tiffin?"* 🥜❌

My top-ranked vector retriever returned Cashew Chutney and Almond Milk.

With perfect confidence. 🚑

---

In Stage 3, I evaluated **9 retrieval configurations** combining **FAISS** (dense vectors) + BM25 (lexical) + Cross-Encoder Rerankers:

Standard metrics ranked **Hybrid Search (RRF: Dense + BM25) + Cross-Encoder Rerank (`bge-reranker-base`)** as #1 ($Hit@5 = 0.700$).

The catch? It scored **40% on constraint compliance.**

That means 6 out of 10 allergy-restricted queries returned the exact excluded ingredient!

---

**Why — Semantic Collapse:**

Vector embeddings for *"avoiding nuts"* map directly adjacent to documents dense in nuts. That's how vector embeddings work — topic proximity, not logical exclusion.

The cross-encoder reranker *amplifies* this defect. It is more confident about the wrong answer.

---

**The fix: Knowledge Graph Pre-Filter (Neo4j Cypher)**

Before **FAISS** vectors even run, a Cypher query on **Neo4j Aura** strips out recipes containing excluded ingredients from the candidate set.

📊 **Results:**

| Config | Hit@5 | Constraint Respected | Latency |
|--------|-------|---------------------|---------|
| Hybrid RRF + Cross-Encoder Rerank (`bge-reranker`) | 0.700 | 40% | 25.8s |
| **Hybrid RRF + Knowledge Graph Pre-Filter (`Neo4j`)** | **0.700** | **100%** | **1.1s** |

Same retrieval quality. Zero safety violations. **23x faster.**

---

**The product lesson:**

Retrieval accuracy without safety compliance is a liability, not a feature.

If you're building RAG for healthcare, legal, or dietary domains — add a `constraint_respected` metric to your eval framework. Today.

Ep. 5 next: Why I stopped forcing vector databases to answer relational questions. 🕸️

---
#CognitiveKitchen #GraphRAG #Neo4j #FAISS #AISafety #VectorSearch #AIProduct #BuildInPublic

---
---

## 📌 Episode 5: Not Everything is a Vector Problem

**Media Attachment**:  
📸 Image: `docs/assets/8_kitchen_pantry_comparison.png` (Live Pantry Audit: Missing ingredients set-difference calculation)

---

Ask any **FAISS** or vector database this:

*"I have rava, mustard seeds, and curry leaves. What can I cook, and what's the ONE ingredient I'm missing?"*

It won't answer. It can't. 🤷

"Missing ingredients" is a set-difference relational calculation.
It is not written in any document.
No amount of vector embedding dimensionality will solve this.

---

**Two things I built to handle relational queries:**

**1. Canonical Vocabulary Engine (`ck-vocab`)**

A deterministic, hand-written rule engine (`rag/vocab/curated.py`) that normalizes 1,776 raw ingredient strings → 158 canonical entities.

`fresh ginger root` → `ginger`
`coriander leaves` ≠ `coriander seeds` (kept separate — a cook knows the difference)

Why hand-written? Because a model that files *besan* as "gluten" produces a vocabulary that's wrong in the ways a cook notices instantly.

**2. Pure Neo4j Knowledge Graph Traversal (Stage 5)**

Zero vector embeddings. Zero FAISS. Direct Cypher queries on **Neo4j Aura**.

→ *"What can I cook tonight?"* → Masoor Dal (missing 1 of 5 ingredients)
→ *"Substitute for ghee?"* → Yogurt, butter, milk, cream
→ *"How many recipes are dairy-free?"* → 92 of 177

📊 Set Precision = 0.689 | Set Recall = 0.617 | **Exclusion Accuracy = 100%**

---

**One negative result worth sharing:**

I also tested **HyDE** (Hypothetical Document Embeddings) as a query transform in Stage 4.

It added ~400 seconds of CPU latency per query.
Accuracy gain: negligible.

Not every clever technique justifies its compute cost. Reporting negative results matters.

---

**The product lesson:**

Stop forcing vector databases to do relational graph logic. Use vectors for similarity. Use **Neo4j** for relationships. Use both when the query demands it.

Ep. 6 tomorrow: +5.5% faithfulness without changing the model or the prompt. 📈

---
#CognitiveKitchen #KnowledgeGraph #Neo4j #FAISS #Cypher #RAG #GraphRAG #AIProduct #BuildInPublic

---
---

## 📌 Episode 6: Free Faithfulness — No Model Change, No Prompt Change

**Media Attachment**:  
📸 Image: `docs/assets/7_kitchen_live_chat_response.png` (Live chat telemetry: 5 sources · 6.0s · $0.00077 cost)

---

Same model.
Same prompt.
Same retrieved chunks.

**+5.5% DeepEval Faithfulness. For free.**

How? I just changed the **order** of the chunks in the context window. 🤯

---

In Stage 6, I evaluated 4 generation strategies using **Qwen2.5-1.5B-Instruct** running 100% on CPU (12 cores, 16 GB RAM, zero GPU) — scored by **DeepEval** with `gpt-4o-mini` as judge:

| Strategy | Faithfulness | Cookability | s/answer |
|----------|-------------|-------------|----------|
| **Context-Reordered (`reordered` - Lost-in-the-Middle Fix)** | **0.760** | 0.525 | 1.6 |
| Strict Context Stuffing (`stuff_strict` - Grounded Refusal) | 0.720 | 0.600 | 2.0 |
| Structured JSON Output (`structured`) | 0.749 | 0.480 | 2.3 |
| Map-Reduce Chunk Summarization (`map_reduce`) | 0.550 | 0.420 | 5.2 |

`reordered` and `stuff_strict` used the **exact same chunks.**
Only the order changed.

---

**Why this works — Lost-in-the-Middle Effect** (Liu et al., 2023):

LLMs pay maximum attention to context at the **beginning and end** of the prompt window. Information buried in the middle gets overlooked, increasing hallucinations.

By placing high-relevance chunks at prompt boundaries:
✅ **Faithfulness: +5.5%**
✅ Additional cost: **$0**
✅ Additional latency: **0s**

All experiments traced through **LangSmith** for reproducibility.

---

**The product lesson:**

Before you rewrite prompts, fine-tune models, or upgrade to a bigger LLM — try rearranging your context.

Context layout strategy frequently has higher ROI than model scaling.

Ep. 7 next: Everything the sandbox proved — now applied to the real 9:30 PM tiffin problem. 🍳

---
#CognitiveKitchen #DeepEval #LangSmith #Qwen #PromptEngineering #GenAI #LLMOps #RAG #AIProduct #BuildInPublic

---
---

## 📌 Episode 7: The Kitchen in Action

**Media Attachment**:  
🎥 Video Walkthrough: `docs/assets/live_demo_walkthrough.webm`  
📸 Image: `docs/assets/8_kitchen_pantry_comparison.png` (Live Pantry Audit)

---

Six episodes of benchmarks, failure modes, and pipeline experiments.

Now — the payoff. 🍳

Here's what happens when you apply all of it to the actual 9:30 PM tiffin problem:

---

**The Full Architecture (Deployed Live on VPS via Docker Compose):**

| Component | Responsibility | Tech Hook |
|-----------|----------------|-----------|
| **Generation** | 100% CPU answering | `Qwen2.5-1.5B-Instruct` |
| **Embeddings** | Local vector passes | `Qwen3-Embedding-0.6B` |
| **Knowledge Graph** | Constraint filter & pantry audit | `Neo4j Aura` (Cypher) |
| **Vector Store** | Fast similarity retrieval | `FAISS` |
| **API Backend** | Async SSE progress streaming | `FastAPI` + `Pydantic` |
| **Frontend** | 3-page interactive console | `Streamlit` |
| **Ingestion** | PDF + Web crawling | `PyMuPDF` + `Playwright` |
| **Evaluation** | Offline judging & tracing | `DeepEval` + `LangSmith` |
| **Deployment** | Self-hosted container | `Docker Compose` (Hostinger VPS) |

---

**The Live User Experience:**

1️⃣ **Ingest**: Drop in a PDF or URL. `Playwright` crawls the pages, `FastAPI` streams real-time progress frames over SSE. 184 recipes ingested into the corpus.

2️⃣ **Experiment**: The RAG Lab lets you toggle 8 chunkers and 9 retrievers. Compare results side-by-side on our 50-recipe / 229-query Golden Data Set.

3️⃣ **Lock**: The winning config becomes the production pipeline.

4️⃣ **Chat**: Ask: *"What can I make with rice?"*  
The model answers in 5.2s, cites 5 source recipes, costs **$0.00077** in tokens, and runs a **live pantry audit** with 1 click:  
*`Plain Savoury Rice — missing 3 of 4 ingredients. Buy: ghee, rice, water.`*

---

🌐 **Try the live app**: https://techideas.tech/  
💻 **GitHub source**: https://github.com/aru911-gethu/CognitiveKitchen  

Tomorrow — the final episode: where Cognitive Kitchen goes next. 🔮

---
#CognitiveKitchen #FullStack #RAG #GraphRAG #OpenSource #Streamlit #FastAPI #Neo4j #FAISS #Docker #Playwright #DeepEval #LangSmith #BuildInPublic

---
---

## 📌 Episode 8: What's Next

**Media Attachment**:  
📸 Image: `docs/assets/1_landing_full.png` (Full Landing Console & Roadmap)

---

Over 7 posts, I shared:

✅ Why naive vector RAG fails on safety-critical domains (Ep. 1, 4)  
✅ How to benchmark RAG honestly with a 3-tier framework (Ep. 2)  
✅ Why the most expensive semantic chunker lost (Ep. 3)  
✅ Why your top retriever might be a liability (Ep. 4)  
✅ When to use graphs instead of vectors for relational logic (Ep. 5)  
✅ A free faithfulness boost from context reordering (Ep. 6)  
✅ A live, deployed product on **Docker** proving all of it (Ep. 7)  

All running on a **1.5B open-source model** on CPU. Zero GPU spend.

---

**What's next for Cognitive Kitchen:**

🔮 **Vision-Driven Pantry Audit**  
Snap a photo of your fridge shelf. A multimodal vision model extracts ingredients → normalizes via `ck-vocab` → auto-populates the **Neo4j** pantry graph. Zero typing.

🎙️ **Hands-Free Voice UX**  
Real-time step-by-step cooking guidance via audio streaming. No touching the screen with messy hands.

📚 **Domain Expansion**  
The RAG Lab Sandbox is domain-agnostic by design. Recipes were first. Legal contracts, medical policies, technical manuals — same **Streamlit** + **FastAPI** + **DeepEval** framework, different corpus.

---

**What building this taught me:**

→ Small models + right architecture > large models + wrong architecture.  
→ Safety compliance is not a quality score. It's a non-negotiable gate.  
→ Negative results (HyDE, Semantic Chunking) are as valuable as positive ones.  
→ Ship it, measure it, share the data. Not just the wins.  

---

💻 https://github.com/aru911-gethu/CognitiveKitchen  
🌐 https://techideas.tech/  

Thank you to everyone who followed this series.

👇 What RAG challenges are you navigating right now? What domain would you test the sandbox on? Drop your thoughts below!

---
#CognitiveKitchen #BuildInPublic #OpenSource #RAG #GraphRAG #AIProduct #VisionAI #MultiModal #GenAI #Qwen #Neo4j #FAISS #Docker #Streamlit #FastAPI #DeepEval #LangSmith
