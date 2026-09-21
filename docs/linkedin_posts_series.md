# 🍳 Cognitive Kitchen — 8-Part LinkedIn Series (Production Edition)

**Author**: Arun Nakkeeran  
**GitHub**: https://github.com/aru911-gethu/CognitiveKitchen  
**Live App**: https://techideas.tech/  
**Media Assets**: Stored in `docs/assets/` (Live full-res screenshots & demo video)

---

## 📌 Episode 1: The 9:30 PM Tiffin Crisis & The Birth of the RAG Sandbox

**Media Attachments**:  
🎥 **Option A (Video Upload — Recommended)**: `docs/assets/live_demo_walkthrough.mp4`  
*(Trimmed 720p walkthrough, 1280x720, 2m 56s — also at `C:\Users\aru91\Downloads\Video Project 1.mp4`)*  

📸 **Option B (Multi-Image Carousel — 4-Slide Story)**:  
1. Slide 1: `docs/assets/1_ingest_landing.png` — *The Production Console (184 recipes · 1,746 graph nodes · 158 ingredients · 229 benchmark queries)*  
2. Slide 2: `docs/assets/2_rag_lab.png` — *The RAG Lab Sandbox (8 chunkers & 9 retrievers competing head-to-head)*  
3. Slide 3: `docs/assets/4_retrieval_benchmark.png` — *The Benchmark Proof (Hybrid RRF + Neo4j hitting 0.850 Hit@5 in 0.7s with 100% safety)*  
4. Slide 4: `docs/assets/8_kitchen_pantry_comparison.png` — *The Payoff (Live chat & Neo4j graph pantry set-difference audit)*  

---

Every night at 9:30 PM, right after dinner, my house turns into an intense negotiation room. 🍱

"What are we packing for tomorrow morning's school tiffin and office lunch?"

The real-world constraints are unforgiving:
→ Swiggy doesn't deliver at 7:00 AM when the school bus rolls up at 8:00 AM.
→ The kids have declared war on monotony: *"Not Idli or Dosa AGAIN, please!"* 🙈
→ We stare into the fridge, debate for 20 minutes, and inevitably default to the **same Idli/Dosa batter for the 4th time this week.**

So I did what any AI engineer would do.

I opened ChatGPT. 🤖
It confidently recommended: *"Paneer Tikka Wraps with fresh mint chutney!"*
My pantry had: semolina (rava), mustard seeds, and curry leaves. Zero paneer. Zero wraps.

Then I built a standard vector RAG app using popular framework tutorials.
It failed even more catastrophically:

🔴 **The Allergy Trap (Semantic Collapse)**: When I queried *"nut-free breakfast"*, vector similarity placed the query closest to Cashew Upma and Almond Milk. Vectors measure topic proximity — they have zero clue what logical exclusion (*WITHOUT*) means. The higher your vector similarity score, the faster it sends someone to the emergency room. 🚑

🔴 **The Phantom Pantry**: It retrieved elaborate recipes requiring 8 obscure ingredients I didn't own. Zero inventory grounding.

🔴 **Relational Blindness**: *"What is the ONE ingredient I'm missing for Rava Upma?"* → Total silence. Vector embeddings cannot calculate set-differences.

---

### The Spark: Why We Built a Sandbox, Not Just Another App 💡

During that late-night debate, an uncomfortable truth struck us:

*Why are AI teams building production RAG systems blindly by stitching together whatever tutorials recommend?*

One blog swears by "Semantic Chunking". Another claims "HyDE is king". Another says "just throw a Cross-Encoder reranker at it".
Yet, almost no one measures these choices against a hand-verified ground truth before pushing to production. They build an entire pipeline on gut feeling — and wonder why users complain about hallucinations, latency spikes, and safety leaks.

That discussion triggered the core breakthrough: **The RAG Lab Sandbox.** 🧪

We decided: Don't just build a cooking assistant.
Build an **industrial, domain-agnostic RAG experimentation workbench** where every architectural layer — chunkers, retrievers, query transforms, and generators — is forced to compete head-to-head on the same ground truth before a single pipeline is locked into production.

The workflow: **1 · Ingest → 2 · Experiment in Sandbox → 3 · Lock What Wins → 4 · Chat.**

The cooking domain was simply our toughest testing ground. Because if your RAG can't reliably distinguish coriander leaves from coriander seeds, or exclude peanuts from a child's breakfast, it has no business handling medical records, legal contracts, or customer support.

---

### 🛠️ The Production Stack Behind the Sandbox

We deployed this live to a self-hosted VPS, proving that an open-source **1.5B model** running 100% on CPU — paired with a Knowledge Graph — can outperform massive proprietary LLMs on safety, speed, and cost.

Here is the engineering architecture:

• **Interactive Frontend**: **Streamlit** multi-page console (`app.py`, `2_RAG_Lab.py`, `3_Kitchen.py`) providing live telemetry, parameter sliders, and side-by-side metric tables.
• **Asynchronous Backend**: **FastAPI** streaming real-time ingestion chunking and token events over **Server-Sent Events (SSE)**.
• **Knowledge Graph (Relational Logic)**: **Neo4j Aura** running Cypher queries to enforce strict allergy exclusion pre-filtering (100% compliance) and graph set-difference calculations (pantry inventory matching).
• **Dense + Lexical Retrieval**: **FAISS** vector store paired with **BM25** lexical search, fused via **Reciprocal Rank Fusion (RRF)**.
• **100% Local CPU Inference**: Open-source **Qwen2.5-1.5B-Instruct** running entirely on CPU (12 cores, 16 GB RAM, $0 GPU cost) answering in 1.6s to 1.9s.
• **Document Ingestion**: **PyMuPDF** & **pypdf** for structural PDF slicing + **Playwright** headless browser for dynamic recipe web crawling with JSON-LD schema parsing.
• **Evaluation & Observability**: **DeepEval** offline LLM-as-a-judge (evaluating Faithfulness, Relevancy, G-Eval Cookability) scored with `gpt-4o-mini`, with 100% execution trace visibility in **LangSmith**.
• **Deployment**: Fully self-hosted container orchestration via **Docker Compose** on Hostinger VPS.

---

### What the RAG Lab Decides (Before You Lock Production)

In the sandbox, every choice is backed by hard numbers:
1. **8 Chunking Strategies**: Fixed-window, Recursive, Sentence, Structure-Aware, and 4 Semantic variations benchmarked on Purity, Recall, and Self-Sufficiency. *(Spoiler: The most expensive semantic chunker lost).*
2. **9 Retrieval Configurations**: Dense vector, Lexical BM25, Hybrid RRF, Cross-Encoders, and Neo4j Graph Pre-Filters compared on Hit@K, MAP, and safety compliance. *(Spoiler: Graph pre-filtering was 23x faster than cross-encoders with 100% allergy safety).*
3. **4 Generation Strategies**: Strict Stuffing, JSON Schema, Map-Reduce, and Context-Reordering scored on DeepEval Faithfulness. *(Spoiler: Rearranging context order gave +10% faithfulness for free).*
4. **The Lock**: With one click, your winning benchmark configuration is saved as the production pipeline.

---

🌐 **Try the live app**: https://techideas.tech/  
💻 **GitHub source**: https://github.com/aru911-gethu/CognitiveKitchen  

Over the next 7 posts, I'll break down the exact benchmark data, the unexpected negative results (HyDE & Semantic Chunking), and the design patterns from every stage.

👇 What's your family's 9:30 PM tiffin debate? And what RAG architecture choice has given your team the biggest headache? Let's discuss in the comments!

#CognitiveKitchen #RAG #GraphRAG #AIProduct #OpenSource #GenAI #Qwen #Neo4j #FAISS #FastAPI #Streamlit #Docker #DeepEval #LangSmith #BuildInPublic

---
---

## 📌 Episode 2: The RAG Lab Sandbox

**Media Attachment**:  
📸 Image: `docs/assets/2_rag_lab.png` (RAG Lab Console: Stage 2 Chunking showing all 8 chunking strategies in clear nomenclature)

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
📸 Image: `docs/assets/3_chunking_benchmark.png` (RAG Lab: Stage 2 Chunking benchmark table showing Structure-Aware hitting 0.991 recall vs Semantic Adjacent 0.469)

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
📸 Image: `docs/assets/4_retrieval_benchmark.png` (Stage 3 Retrieval benchmark: Hybrid RRF + Knowledge Graph Pre-Filter hitting 0.850 Hit@5, 100% constraint compliance in 0.7s)

---

Consider this query at 9:30 PM:

*"I'm avoiding nuts. What quick breakfast can I pack for tiffin?"* 🥜❌

My top-ranked vector retriever returned Cashew Chutney and Almond Milk.

With perfect confidence. 🚑

---

In Stage 3, I evaluated **9 retrieval configurations** combining **FAISS** (dense vectors) + BM25 (lexical) + Cross-Encoder Rerankers:

Standard metrics ranked **Hybrid Search (RRF: Dense + BM25) + Cross-Encoder Rerank (`bge-reranker-base`)** high on Hit@5.

The catch? Without graph filtering, it scored abysmal marks on constraint compliance (returning excluded ingredients 6-9 times out of 10).

Why? The cross-encoder reranker *amplifies* topic proximity. It is more confident about the wrong answer.

---

**The fix: Knowledge Graph Pre-Filter (Neo4j Cypher)**

Before **FAISS** vectors even run, a Cypher query on **Neo4j Aura** strips out recipes containing excluded ingredients from the candidate set.

📊 **Live Benchmark Scorecard:**

| Config | Hit@5 | Constraint Respected | Latency |
|--------|-------|---------------------|---------|
| Keyword Search (BM25) | 0.500 | 1% | 0.2s |
| Keyword Search (BM25) + Knowledge Graph Pre-Filter (Neo4j) | 0.600 | 1% | 0.8s |
| Hybrid Search (RRF: Dense + BM25) | 0.700 | 1% | 0.2s |
| **Hybrid Search (RRF: Dense + BM25) + Knowledge Graph Pre-Filter (Neo4j)** | **0.850** | **100%** | **0.7s** |

Highest retrieval quality (0.850 Hit@5). Zero safety violations (100% compliance). **Sub-second latency (0.7s).**

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
📸 Image: `docs/assets/8_kitchen_pantry_comparison.png` (Live Pantry Audit: Missing ingredients set-difference & category-safe swap: wheat flour for all-purpose flour)

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
📸 Image: `docs/assets/6_generation_benchmark.png` (RAG Lab Console: Stage 6 Generation benchmark table showing Context-Reordered hitting 0.867 Faithfulness vs Strict Context Stuffing & Map-Reduce)

---

Same model.
Same prompt.
Same retrieved chunks.

**+10.0% DeepEval Faithfulness. For free.**

How? I just changed the **order** of the chunks in the context window. 🤯

---

In Stage 6, I evaluated 4 generation strategies using **Qwen2.5-1.5B-Instruct** running 100% on CPU (12 cores, 16 GB RAM, zero GPU) — scored by **DeepEval** with `gpt-4o-mini` as judge:

| Strategy | Faithfulness | Cookability | s/answer |
|----------|-------------|-------------|----------|
| **Context-Reordered (`reordered` - Lost-in-the-Middle Fix)** | **0.867** | **0.720** | **1.9s** |
| Strict Context Stuffing (`stuff_strict` - Grounded Refusal) | 0.767 | 0.694 | 1.6s |
| Structured JSON Schema Output (`structured`) | 0.750 | 0.632 | 3.4s |
| Map-Reduce Chunk Summarization (`map_reduce`) | 0.550 | 0.546 | 5.8s |

`reordered` and `stuff_strict` used the **exact same chunks.**
Only the order changed.

---

**Why this works — Lost-in-the-Middle Effect** (Liu et al., 2023):

LLMs pay maximum attention to context at the **beginning and end** of the prompt window. Information buried in the middle gets overlooked, increasing hallucinations.

By placing high-relevance chunks at prompt boundaries:
✅ **Faithfulness: +10.0%** (0.867 vs 0.767)
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
🎥 Video Walkthrough: `docs/assets/live_demo_walkthrough.mp4` (720p Trimmed for LinkedIn, 1280x720, 2m 56s — also at `C:\Users\aru91\Downloads\Video Project 1.mp4`)  
📸 Image: `docs/assets/7_kitchen_live_chat_response.png` (Live Kitchen Chat: "suggest something to cook with rava , like a dosa?" · 5 sources · 9.9s · $0.00026)

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

4️⃣ **Chat**: Ask: *"suggest something to cook with rava , like a dosa?"*  
The model answers in 9.9s, cites 5 source recipes, costs **$0.00026** in tokens, and runs a **live pantry audit** with 1 click:  
*`Didir Onion Rava Dosa — missing 8 of 11 ingredients. Buy: asafoetida, cashew, cumin seeds, ginger, green chilli, rice flour, semolina. Swap: no all-purpose flour — use wheat flour, already on your shelf.`*

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
