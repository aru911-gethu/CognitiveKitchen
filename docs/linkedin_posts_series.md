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

## 📌 Episode 2: The RAG Lab Sandbox & The 3-Tier Evaluation Framework

**Media Attachments**:  
📸 **Option A (Multi-Image Carousel — 3-Slide Framework)**:  
1. Slide 1: `docs/assets/2_rag_lab.png` — *The RAG Lab Sandbox (Interactive Stage 2 Chunking selector with 8 modular strategies)*  
2. Slide 2: `docs/assets/1_landing_full.png` — *The Ground Truth Anchor (50 golden recipes · 229 verified queries · Service online)*  
3. Slide 3: `docs/assets/4_retrieval_benchmark.png` — *The Live Benchmark Table (Hit@K, Recall, MAP, Latency, and Constraint Compliance)*  

📸 **Option B (Single Image)**: `docs/assets/2_rag_lab.png`  

---

Here is an uncomfortable dirty secret in enterprise AI development:

**Most RAG benchmarks are cheating.** 🛑

If your chunking scripts, vocabulary mappings, or indexing pipelines have ever *seen* your test queries during development — your AI isn't intelligent. It's just memorizing its own answer key.

In Ep. 1, I introduced the RAG Lab Sandbox as Cognitive Kitchen's core breakthrough. Today, I'm opening the hood on how it enforces honest, leak-proof benchmarks before locking any production pipeline.

When building an enterprise AI product, "this prompt feels pretty good" is not an engineering metric. You need deterministic, automated go/no-go quality gates.

Here is the **3-Tier Evaluation Framework** I built into the sandbox:

---

### The 3-Tier Evaluation Framework 🧪

📌 **Tier 1 — Document & Extraction Health (Pre-LLM)**  
Measures raw document parsing quality *before* any LLM touches the text.  
→ **Chunk Purity**: Does this chunk contain text from exactly one recipe, or does it drag in remnants of another? (Low purity poisons every downstream retrieval stage).  
→ **Self-Sufficiency**: Can an agent answer a question using *only* this chunk, without needing missing headers or severed instructions?  
→ **Telemetry Ledger**: Real-time tracking of ingestion latency ($s/doc$) and token storage footprint.

📌 **Tier 2 — Unlabeled Production Traffic (DeepEval + LangSmith)**  
Evaluates real conversational queries without requiring costly manual human labels.  
→ Scored using **DeepEval** with `gpt-4o-mini` acting as an offline judge:  
  • **Faithfulness**: Are claims strictly supported by retrieved context?  
  • **Answer Relevancy**: Did the model directly address user intent?  
  • **G-Eval Cookability**: Does the generated response provide actionable, step-by-step culinary instructions?  
  • **Abstention Rate**: Does the model safely refuse when information is absent (`NOT_IN_CONTEXT`)?  
→ Every generation and evaluation run is traced 100% in **LangSmith** for granular latency and cost observability.

📌 **Tier 3 — Curated Golden Data Set Benchmark (Ground Truth)**  
A hand-verified, frozen benchmark of **50 recipes** and **229 queries** (`data/golden_dataset.json`) spanning 7 distinct intent families:  
1. Single ingredient lookups  
2. Multi-ingredient constraints  
3. Hard allergen exclusions ("avoiding nuts")  
4. Pantry set-difference ("missing ingredients")  
5. Time-bound requests ("under 20 minutes")  
6. Course-specific classifications (Breakfast/Dinner)  
7. Ingredient substitutions ("ghee substitute")  

Every candidate pipeline is evaluated across **Hit@K**, **Recall@K**, **MAP (Mean Average Precision)**, and **Constraint Compliance** — rendered live in the **Streamlit** dashboard.

---

### 🔒 The Non-Negotiable Rule of the Sandbox

No ingestion script, no canonical vocabulary parser (`ck-vocab`), no knowledge graph builder (`ck-graph`), and no embedding indexing job is EVER allowed to inspect `golden_dataset.json`.

Ground truth stays strictly air-gapped from pipeline construction.

If you change an embedding model or rewrite a chunker, you run the benchmark. The numbers move. You see the trade-offs immediately. If the new pipeline passes the quality gate, you click **"Lock"** — and it instantly becomes your production runtime.

No guesswork. No blind deployments.

Tomorrow in Ep. 3: I put 8 chunking strategies into the ring. The most expensive semantic chunker lost. 📄✂️

👇 How does your team evaluate RAG pipelines today? Do you have an air-gapped golden dataset, or are you still evaluating on gut feel? Let's talk in the comments!

#CognitiveKitchen #RAG #DeepEval #LangSmith #AIProduct #LLMOps #Evaluation #GoldenDataSet #Streamlit #FastAPI #Docker #BuildInPublic

---
---

## 📌 Episode 3: Chunking — Why the Most Expensive Strategy Lost

**Media Attachments**:  
📸 **Option A (Multi-Image Carousel — 2-Slide Comparison)**:  
1. Slide 1: `docs/assets/2_rag_lab.png` — *Stage 2 Chunking Selector (8 modular chunking strategies in clear nomenclature)*  
2. Slide 2: `docs/assets/3_chunking_benchmark.png` — *The Chunking Benchmark Table (Structure-Aware hitting 0.991 recall vs Semantic Adjacent at 0.469)*  

📸 **Option B (Single Image)**: `docs/assets/3_chunking_benchmark.png`  

---

"Semantic Chunking" is the default darling of almost every RAG tutorial on the internet.

The pitch sounds irresistible: *"Don't split on arbitrary character counts! Use embedding distance drift to detect when the topic changes and split dynamically!"*

So in Stage 2 of Cognitive Kitchen, I put it to the test against 7 other chunking strategies across 184 recipes.

It lost. Badly. 📉

---

### The 8 Chunking Strategies in the Sandbox 📄✂️

1. **Structure-Aware (Full Recipe)** (`recipe`) — 1 chunk = 1 complete recipe based on document schema.
2. **Recipe Sections (Split)** (`recipe_sections`) — Splitting each recipe into an Ingredients chunk and a Method chunk.
3. **Recursive Character Split** (`recursive`) — Paragraph, line, and word hierarchy (LangChain/LlamaIndex default).
4. **Sentence Boundary Split** (`sentence`) — NLTK-style grammatical sentence boundaries.
5. **Fixed Window Character Split** (`character`) — Naive fixed character window with overlap.
6. **Token Window Split** (`token`) — Fixed token counts.
7. **Semantic Adjacent (Embedding Distance Split)** (`semantic_adjacent`) — Splits when cosine distance between consecutive sentences exceeds a threshold.
8. **Semantic Centroid (Running Drift Split)** (`semantic_centroid`) — Splits when sentences drift from a running centroid.

---

### What Went Wrong with Semantic Chunking? 🧈🙈

Semantic chunking failed on two massive axes: **Compute Overhead** and **Boundary Poisoning**.

🔴 **The Compute Cost**: Generating sentence embeddings via `sentence-transformers/all-MiniLM-L6-v2` for thousands of sentences on CPU inflated ingestion time by over 400%.

🔴 **Boundary Straddling**: Embedding models look for topical similarity, not entity boundaries. A sentence describing the end of a Biryani recipe sounded semantically adjacent to the opening steps of a Mutton Curry. 
The result? It merged the tail of one dish into the head of another!

When I tested: *"How much butter do I need?"*, the downstream LLM blended the ingredients together and hallucinated an averaged butter quantity across completely unrelated recipes.

---

### The Fix: Structure-Aware Document Slicing (`recipe`)

Instead of throwing expensive embeddings at raw text, I used the physical and structural cues inherent in the source formats:
→ **PDFs**: Parsed layout cues and recipe boundary headings using **PyMuPDF** & **pypdf**.
→ **Web Crawls**: Extracted structured **JSON-LD Schema** via **Playwright** headless browser and BeautifulSoup.

Every chunk corresponds to exactly **one complete, atomic recipe** (averaging 807 characters). It has 100% self-sufficiency: the LLM never needs to look upstream or downstream to find missing ingredients or steps.

📊 **The Live Benchmark Scorecard (from our Streamlit Console):**

| Strategy | Chunks | Avg Chars | Chunk Purity | Recall@K | Self-Sufficiency |
|---|---|---|---|---|---|
| **Structure-Aware (Full Recipe)** | **184** | **807.6** | **0.939** | **0.991** | **1.000** |
| Recipe Sections (Split) | 412 | 358.6 | 0.953 | 0.820 | 0.976 |
| Recursive Character Split | 345 | 438.4 | 0.932 | 0.745 | 0.933 |
| Semantic Centroid Split | 787 | 188.3 | 0.642 | 0.548 | 0.608 |
| Semantic Adjacent Split | 1,016 | 145.6 | 0.748 | 0.469 | 0.511 |

Look at that gap: **0.991 recall vs 0.469 recall.**
The simplest, cheapest structural chunker outperformed the most expensive semantic model by **2.1x.**

---

### The Product Lesson 💡

Before you burn your compute budget on heavy semantic splitters, inspect your document's inherent structure.
Schema-aware parsing almost always gives you cleaner purity, higher recall, and faster latency for $0.

Ep. 4 tomorrow: Our top-scoring retriever had an allergy safety defect that would have sent someone to the hospital. 🚑

👇 What chunking strategy does your production app use today? Have you measured chunk purity vs recall on your corpus? Drop your thoughts below!

#CognitiveKitchen #RAG #Chunking #PyMuPDF #Playwright #NLP #AIProduct #Evaluation #DeepEval #BuildInPublic

---
---

## 📌 Episode 4: The Top Retriever Failed Safety — The "Semantic Collapse" Trap

**Media Attachments**:  
📸 **Option A (Multi-Image Carousel — 2-Slide Deep Dive)**:  
1. Slide 1: `docs/assets/4_retrieval_benchmark.png` — *Stage 3 Retrieval Benchmark (Hybrid RRF + Neo4j Graph Pre-Filter hitting 0.850 Hit@5 with 100% safety in 0.7s)*  
2. Slide 2: `docs/assets/3_kitchen_chat.png` — *The Production Chain (Knowledge Graph Pre-Filter locked into the live runtime)*  

📸 **Option B (Single Image)**: `docs/assets/4_retrieval_benchmark.png`  

---

Consider this real 9:30 PM user query:

*"I am avoiding nuts. What quick breakfast can I pack for my allergic kid?"* 🥜❌

My top-ranked vector retriever returned Cashew Chutney and Almond Milk.

With 99% cosine similarity confidence. 🚑

In healthcare, legal, or dietary AI — the better your vector ranker, the faster it sends someone to the emergency room.

---

### What is "Semantic Collapse"? 🧠💥

Most engineers assume that if an LLM or embedding model is large enough, it understands negation.

It doesn't.

Vector embeddings map text into continuous geometric space based on **topic co-occurrence**.
The query *"avoiding nuts"* shares almost identical semantic coordinates with *"cashews, almonds, and peanuts"*. Vector similarity measures topical proximity — it is completely blind to boolean logic (*WITHOUT*).

Worse: When you add a state-of-the-art Cross-Encoder Reranker (`bge-reranker-base`), it scores the document pair even higher because the word "nut" appears prominently in both the prompt and the recipe. The cross-encoder is *more confident about the fatal answer*.

In our tests, the cross-encoder scored **40% on constraint compliance** (breaching safety 6 times out of 10) and added **25.8 seconds of CPU latency** per query!

---

### The Fix: Knowledge Graph Pre-Filtering (Neo4j Aura + Cypher) 🕸️🛡️

We stopped asking vector embeddings to do boolean logic.

Instead, before **FAISS** vector search is even invoked, a deterministic Cypher query runs on our **Neo4j Aura** knowledge graph:

```cypher
MATCH (r:Recipe)-[:USES_INGREDIENT]->(i:Ingredient)
WHERE i.category = 'nut' OR i.canonical_name = 'cashew'
WITH collect(DISTINCT r.id) AS blocked_recipes
MATCH (clean:Recipe)
WHERE NOT clean.id IN blocked_recipes
RETURN clean.id
```

This strips out 100% of allergen-tainted recipes *before* dense retrieval runs. The vector search is restricted strictly to the candidate set of guaranteed safe recipes.

📊 **Live Benchmark Scorecard (9 Retrieval Configurations Evaluated):**

| Retrieval Pipeline | Hit@5 | Recall@K | MAP | Constraint Compliance | Latency |
|---|---|---|---|---|---|
| Keyword Search (BM25) | 0.500 | 0.286 | 0.281 | 1% | 0.2s |
| Keyword (BM25) + Neo4j Graph Pre-Filter | 0.600 | 0.386 | 0.356 | **100%** | 0.8s |
| Hybrid Search (RRF: Dense + BM25) | 0.700 | 0.393 | 0.393 | 1% | 0.2s |
| **Hybrid Search (RRF) + Neo4j Graph Pre-Filter** | **0.850** | **0.503** | **0.503** | **100%** | **0.7s** |

Look at the results:
✅ **Hit@5 jumped to 0.850** (best overall retrieval quality).
✅ **Constraint compliance reached 100%** (zero breaches across all test queries).
✅ **Latency: 0.7 seconds** (sub-second on CPU — **36x faster** than heavy cross-encoders).

---

### The Product Lesson 💡

Retrieval accuracy without safety compliance is not a feature; it is an existential liability.

If you are building RAG for medical regimens, compliance policies, or dietary safety:
**Do not trust vector similarity for negative constraints.**
Enforce hard symbolic boundaries with a knowledge graph first; then use vectors to rank what's safe.

Ep. 5 tomorrow: Why I stopped forcing vector databases to calculate missing pantry items. 🕸️

👇 Has your team encountered "Semantic Collapse" with negative queries in RAG? How are you handling hard exclusions? Let's discuss!

#CognitiveKitchen #GraphRAG #Neo4j #FAISS #AISafety #VectorSearch #Cypher #LLMOps #AIProduct #BuildInPublic

---
---

## 📌 Episode 5: Not Everything is a Vector Problem — Knowledge Graphs & Relational Logic

**Media Attachments**:  
📸 **Option A (Multi-Image Carousel — 2-Slide Relational Proof)**:  
1. Slide 1: `docs/assets/8_kitchen_pantry_comparison.png` — *Live Pantry Audit (Missing ingredients set-difference & category-safe swap: wheat flour for all-purpose flour)*  
2. Slide 2: `docs/assets/5_query_transform.png` — *Stage 4 & Stage 5 Logic (Query decomposition & pure Neo4j Cypher traversals)*  

📸 **Option B (Single Image)**: `docs/assets/8_kitchen_pantry_comparison.png`  

---

Ask any vector database on earth (FAISS, Pinecone, Chroma, or Weaviate) this simple question:

*"I have rava, mustard seeds, and curry leaves. What can I cook, and what is the ONE ingredient I'm missing?"*

It won't answer. It can't. 🤷

Why? Because "missing ingredients" is an algebraic set-difference:

$$\text{Missing} = \text{Recipe Ingredients} \setminus \text{Pantry Inventory}$$

That information is **not written in any document chunk**.
No recipe text says: *"Cook this if you are missing mustard seeds."*
No amount of vector embedding dimensionality or prompt gymnastics can solve relational set-differences without hallucinating.

---

### Two Engines Built for Relational Precision 🕸️

To solve this, Stage 5 bypasses vectors completely and switches to structured graph intelligence:

#### 1. Canonical Vocabulary Engine (`ck-vocab`)
Before building relationships, you need entity resolution. Real cookbooks contain messy, inconsistent strings (`freshly grated ginger`, `ginger root`, `ginger paste`).

We built a deterministic rule engine (`rag/vocab/curated.py`) that normalizes **1,776 raw strings → 158 canonical ingredient entities**.
→ `fresh ginger root` → `ginger`  
→ `coriander leaves` ≠ `coriander seeds` (kept strictly separate — every Indian cook knows their culinary chemistry is worlds apart).

*Why deterministic rules instead of an LLM?*  
Because an LLM that casually classifies *besan* (gram flour) as "wheat/gluten" creates safety defects that an experienced cook notices in one second.

#### 2. Pure Neo4j Knowledge Graph Traversal (Stage 5)
We mapped our entire corpus into **Neo4j Aura**: **1,746 nodes** and **5,277 relationships** (`:USES_INGREDIENT`, `:IN_CATEGORY`, `:SUBSTITUTES_FOR`).

Direct Cypher traversals answer what vector search can never touch:
→ *"What can I cook tonight?"* → Matches recipes against your pantry shelf.  
→ *"How many recipes are dairy-free?"* → **92 of 177** (Exact count, zero hallucinations).  
→ *"Pantry Audit on Didir Onion Rava Dosa"*:  
  • Missing 8 of 11 ingredients.  
  • Buy list: `asafoetida, cashew, cumin seeds, ginger, green chilli, rice flour, semolina`.  
  • Smart Swap: `no all-purpose flour — use wheat flour, already on your shelf`.  

Notice the swap: **wheat flour for all-purpose flour.**
By enforcing strict taxonomic categories in Cypher (`WHERE c.category = t.category`), the graph guarantees swaps stay strictly within family (flours with flours, nuts with nuts) — completely eliminating absurd hallucinations like swapping cashew with tomato!

📊 **Graph Benchmark:** Set Precision = 0.689 | Set Recall = 0.617 | **Exclusion Accuracy = 100%**

---

### An Honest Negative Result: Why We Dropped HyDE 📉

In Stage 4, I tested **HyDE (Hypothetical Document Embeddings)**. The idea: Ask the LLM to write a hypothetical recipe, then embed that hallucinated recipe to find real recipes.

The result was an operational disaster:
🔴 On local CPU inference, generating hypothetical recipes added **~400 seconds of latency per query**.
🔴 Retrieval gain over hybrid search: **< 1.2%**.

We killed HyDE immediately. Reporting negative results is just as vital as sharing wins — it stops your team from shipping expensive, low-value complexity.

---

### The Product Lesson 💡

Stop forcing vector databases to do relational graph algebra.
→ Use **vectors** for unstructured semantic discovery.  
→ Use **Neo4j knowledge graphs** for entity relationships, set-differences, and inventory math.  
→ Use **both** when production demands it.

Ep. 6 tomorrow: +10% faithfulness without changing the model or the prompt. 📈

👇 Has your team tried using Knowledge Graphs with RAG? Where do you draw the boundary between Vector search and Graph traversal?

#CognitiveKitchen #KnowledgeGraph #Neo4j #FAISS #Cypher #GraphRAG #AIProduct #LLMOps #BuildInPublic

---
---

## 📌 Episode 6: Free Faithfulness — No Model Change, No Prompt Change

**Media Attachments**:  
📸 **Option A (Multi-Image Carousel — 2-Slide Generation Proof)**:  
1. Slide 1: `docs/assets/6_generation_benchmark.png` — *Stage 6 Generation Benchmark Table (Context-Reordered hitting 0.867 Faithfulness & 0.720 Cookability vs Map-Reduce 0.550)*  
2. Slide 2: `docs/assets/7_kitchen_live_chat_response.png` — *Live Chat Telemetry (Grounding verified across 5 sources in 9.9s for $0.00026)*  

📸 **Option B (Single Image)**: `docs/assets/6_generation_benchmark.png`  

---

Same open-source model (`Qwen2.5-1.5B-Instruct` running 100% on CPU).  
Same prompt instructions.  
Same 5 retrieved context chunks.  

**+10.0% DeepEval Faithfulness. For free.** 🤯  

Total implementation cost: **$0**.  
Total latency added: **0 seconds**.  

How? We simply changed the **physical order** of the chunks inside the context window.

---

### The Science: The "Lost-in-the-Middle" Effect 🧠📉

In 2023, Liu et al. (Stanford/Berkeley) demonstrated a fundamental limitation of transformer attention mechanisms:
LLMs attend with laser focus to text at the **very beginning** (primacy bias) and the **very end** (recency bias) of their context window.

Information buried in the middle of a long prompt suffers from massive attention decay. The model quietly overlooks middle documents, leading to hallucinations and ignored constraints.

When you stuff retrieved chunks in descending order of relevance $[c_1, c_2, c_3, c_4, c_5]$, chunk $c_3$ lands squarely in the dead zone.

---

### The Fix: Context-Reordered Interleaving (`reordered`) 🔄

We implemented a simple interleaving algorithm:

Given chunks ranked 1 to 5 by relevance, we place them at the window boundaries:
$$\text{Layout: } [c_1, c_3, c_5, c_4, c_2]$$

• Chunk 1 sits at the top (maximum attention).  
• Chunk 2 sits at the bottom (immediate recency before generation).  
• Less critical chunks sit in the middle trough.  

---

### The 4 Generation Strategies Evaluated in the Sandbox 🧪

In Stage 6, we tested 4 strategies on local CPU inference, scored by **DeepEval** with `gpt-4o-mini` as judge and traced in **LangSmith**:

📊 **The Live Scorecard (from our Streamlit Console):**

| Strategy | Faithfulness | Cookability (G-Eval) | Answer Relevancy | Latency | Judge Spend |
|---|---|---|---|---|---|
| **Context-Reordered (`reordered`)** | **0.867** | **0.720** | 0.565 | **1.9s** | $0.0006 |
| Strict Context Stuffing (`stuff_strict`) | 0.767 | 0.694 | **0.796** | **1.6s** | $0.0002 |
| Structured JSON Schema (`structured`) | 0.750 | 0.632 | 0.818 | 3.4s | $0.0011 |
| Map-Reduce Summarization (`map_reduce`) | 0.550 | 0.546 | 0.800 | 5.8s | $0.0006 |

---

### Why Did Map-Reduce Fail So Miserably? (0.550 Faithfulness)

In document summarization, Map-Reduce is standard practice.
In recipe RAG, it was catastrophic.

When a chunk contained only ingredients or only cooking steps (due to section chunking), evaluating it in isolation caused individual map calls to return `NOT_IN_CONTEXT`. The reduce phase had almost nothing to synthesize!
Recipe context is **atomic** — slicing it across multiple isolated LLM map calls fragments reasoning.

---

### The Product Lesson 💡

Before you:
❌ Spend weeks fine-tuning custom models  
❌ Upgrade to 70B parameter models with giant GPU bills  
❌ Rewrite prompts for the 20th time  

**Try optimizing your context window layout.**
Prompt layout engineering frequently delivers higher ROI, lower hallucination rates, and zero extra cost.

Ep. 7 tomorrow: Taking all these locked benchmark winners to the live kitchen! 🍳

👇 Have you tested Lost-in-the-Middle mitigation in your RAG prompts? What context reordering strategy has worked best for your team?

#CognitiveKitchen #DeepEval #LangSmith #Qwen #PromptEngineering #GenAI #LLMOps #RAG #AIProduct #BuildInPublic

---
---

## 📌 Episode 7: The Kitchen in Action — From Benchmarks to the Dining Table

**Media Attachments**:  
🎥 **Option A (Video Walkthrough — Recommended)**: `docs/assets/live_demo_walkthrough.mp4`  
*(Trimmed 720p walkthrough, 1280x720, 2m 56s — also at `C:\Users\aru91\Downloads\Video Project 1.mp4`)*  

📸 **Option B (Multi-Image Carousel — 3-Slide Live Journey)**:  
1. Slide 1: `docs/assets/3_kitchen_chat.png` — *The Production Console (Locked pipeline badge: Structure-Aware → Hybrid RRF → Neo4j Pre-Filter → Sub-Query → Context-Reordered)*  
2. Slide 2: `docs/assets/7_kitchen_live_chat_response.png` — *Live Assistant Response ("suggest something to cook with rava , like a dosa?" · Didir Onion Rava Dosa · 5 sources · 9.9s · $0.00026)*  
3. Slide 3: `docs/assets/8_kitchen_pantry_comparison.png` — *1-Click Pantry Audit (Missing 8 of 11 ingredients · Buy list · Category-safe shelf swap)*  

---

Six episodes of benchmarks, failure modes, and architectural trade-offs.

Now — the payoff. 🍳

What happens when a parent types: *"suggest something to cook with rava , like a dosa?"* at 9:30 PM on a live, self-hosted container running on a modest VPS?

Here is the exact execution trace running under the hood:

---

### The Live Execution Trace in the Kitchen ⚡

1️⃣ **The Pipeline Gate**: The assistant immediately activates our locked production pipeline (visible in the top banner):  
`Structure-Aware (Full Recipe) → Hybrid Search (RRF) → Neo4j Graph Pre-Filter → Sub-Query Decomposition → Context-Reordered`

2️⃣ **Allergen Pre-Filtering**: Before retrieving documents, **Neo4j Aura** queries the knowledge graph to ensure zero conflict with active user dietary restrictions.

3️⃣ **Hybrid Dense + Lexical Fusion**: **FAISS** dense vectors retrieve semantic equivalents for "rava" (semolina), while **BM25** scores exact lexical matches for "dosa". **Reciprocal Rank Fusion (RRF)** merges the candidate lists.

4️⃣ **Lost-in-the-Middle Context Interleaving**: The winning candidate — `Didir Onion Rava Dosa` — is placed directly at the context boundary (position 0) of the prompt.

5️⃣ **100% Local CPU Inference**: Open-source **Qwen2.5-1.5B-Instruct** generates the complete recipe:  
• 1 cup semolina/rava  
• 1 cup maida  
• 1/2 cup rice flour  
• 4-5 green chillies, ginger, jeera, asafoetida, onions, cashews  
• Full step-by-step batter preparation  
📊 **Telemetry**: **5 sources cited · 9.9s CPU latency · $0.00026 token spend.**

6️⃣ **1-Click Relational Pantry Audit**:  
Clicking **"Compare with my pantry"** triggers a pure **Neo4j Cypher set-difference traversal** against the user's shelf:  
• *Matched 6 pantry items*  
• *Didir Onion Rava Dosa — missing 8 of 11 ingredients*  
• *Buy: asafoetida, cashew, cumin seeds, ginger, green chilli, rice flour, semolina*  
• *Safe Swap: no all-purpose flour — use wheat flour, already on your shelf*  

No negotiation room. No 4th-night-in-a-row frozen batter. Breakfast is solved in 10 seconds flat.

---

### 🛠️ The Full Production Architecture (Deployed via Docker Compose)

| Layer | Responsibility | Technology Hook |
|---|---|---|
| **CPU Generation** | 100% local answering ($0 GPU) | `Qwen2.5-1.5B-Instruct` |
| **Local Embeddings** | Sentence vectorization | `all-MiniLM-L6-v2` / `Qwen` |
| **Knowledge Graph** | Relational logic & safety pre-filter | `Neo4j Aura` (Cypher) |
| **Vector Index** | Dense similarity retrieval | `FAISS` |
| **Streaming Backend** | Async progress frames | `FastAPI` (Server-Sent Events) |
| **Frontend UI** | 3-page interactive console | `Streamlit` |
| **Ingestion Engine** | PDF & Web recipe extraction | `PyMuPDF` + `Playwright` |
| **Observability** | Offline quality gates & tracing | `DeepEval` + `LangSmith` |
| **Deployment** | Self-hosted Linux container | `Docker Compose` (Hostinger VPS) |

---

🌐 **Try the live app**: https://techideas.tech/  
💻 **GitHub repository**: https://github.com/aru911-gethu/CognitiveKitchen  

Tomorrow — the final episode: What building Cognitive Kitchen taught me about the future of AI engineering, and our multimodal roadmap. 🔮

👇 Would your family trust an AI that audits their pantry shelf? What’s the biggest barrier to deploying local small models in your company?

#CognitiveKitchen #FullStack #RAG #GraphRAG #OpenSource #Streamlit #FastAPI #Neo4j #FAISS #Docker #Playwright #DeepEval #LangSmith #BuildInPublic

---
---

## 📌 Episode 8: What's Next — Engineering Principles & The Multimodal Roadmap

**Media Attachments**:  
📸 **Option A (Multi-Image Carousel — 3-Slide Finale)**:  
1. Slide 1: `docs/assets/1_landing_full.png` — *The Complete Production Console (Benchmark health & architecture roadmap)*  
2. Slide 2: `docs/assets/2_rag_lab.png` — *The 6-Stage Empirical RAG Lifecycle (From ingest to locked runtime)*  
3. Slide 3: `docs/assets/8_kitchen_pantry_comparison.png` — *From Multimodal Vision to Grounded Graph Action*  

📸 **Option B (Single Image)**: `docs/assets/1_landing_full.png`  

---

Over 7 posts, I opened the hood on **Cognitive Kitchen**:
• 184 recipe documents ingested  
• 1,746 knowledge graph nodes & 5,277 relationships mapped  
• 158 canonical ingredients curated  
• 50 golden recipes & 229 verified benchmark queries air-gapped  
• 100% running on a 1.5B open-source model on CPU — with **$0 GPU spend**.

The real breakthrough was never just a cooking app.
It was proving that **empirical benchmarking before production locking** beats guesswork every single time.

Here are the 4 non-negotiable principles this build taught me:

---

### 4 Core Engineering Principles from the Sandbox 📐

1️⃣ **Small Models + Right Architecture > Giant LLMs + Default Tutorials**  
A 1.5B model on CPU — paired with a Neo4j knowledge graph and context reordering — outperformed proprietary models on safety, speed, and deterministic inventory logic. Architecture always beats parameter bloat.

2️⃣ **Safety & Constraint Compliance is a Gate, Not a Quality Score**  
If a user is allergic to nuts, a retriever that returns cashew milk 1 time out of 10 is not "90% accurate". It is dangerous. Enforce hard symbolic constraints with a knowledge graph *before* probabilistic neural ranking.

3️⃣ **Publish Your Negative Results**  
We tested HyDE: it added 400s of CPU latency for zero gain. We tested Semantic Chunking: it straddled recipe boundaries and blended butter quantities. We tested Map-Reduce: it fragmented recipes into `NOT_IN_CONTEXT` errors. Sharing failures saves your team hundreds of engineering hours.

4️⃣ **Empirical Benchmarks Before Production Locks**  
If your chunker, retriever, or vocabulary has ever seen your test queries, your AI is cheating. An air-gapped 3-tier evaluation framework transforms RAG from *"this feels good"* into quantifiable engineering.

---

### 🔮 The Roadmap: Where Cognitive Kitchen Goes Next

1. 📸 **Vision-Driven Multimodal Pantry Audit**  
Snap a photo of your fridge shelf. A multimodal vision model extracts bounding boxes → normalizes strings through `ck-vocab` → auto-populates your Neo4j pantry graph. Zero typing.

2. 🎙️ **Hands-Free Bidirectional Voice UX**  
Real-time step-by-step culinary guidance via streaming WebSockets. Ask questions and get timing alerts without touching screens with messy, floured hands.

3. 📚 **Domain Expansion (Beyond Cooking)**  
The RAG Lab Sandbox is domain-agnostic by design. The same 6-stage Streamlit + FastAPI + DeepEval framework can benchmark:  
• **Healthcare**: Clinical protocol extraction with strict pharmacological contraindication pre-filtering.  
• **Legal & Contracts**: Clause extraction with regulatory compliance gates.  
• **Enterprise Manuals**: SOP compliance with set-difference tooling audits.

---

💻 **GitHub source**: https://github.com/aru911-gethu/CognitiveKitchen  
🌐 **Live app running on VPS**: https://techideas.tech/  

Thank you to everyone who followed this 8-part journey!

👇 What RAG challenges is your engineering team currently navigating? What domain would you test the RAG Lab Sandbox on next? Let's connect and discuss below!

#CognitiveKitchen #BuildInPublic #OpenSource #RAG #GraphRAG #AIProduct #VisionAI #MultiModal #GenAI #Qwen #Neo4j #FAISS #Docker #Streamlit #FastAPI #DeepEval #LangSmith #LLMOps
