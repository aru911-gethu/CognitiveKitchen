"""RAG Lab - compare strategies stage by stage, then lock a pipeline.

Each stage shows every option scored, you pick one, and the choice locks. The
next stage is then measured USING that choice rather than a default, so the
numbers you read at Stage 3 belong to the chunker you actually chose. Locking
forward is the whole point: a table of Stage 3 results computed against some
other chunker would not tell you anything about your pipeline.

Stage 5 sits outside the chain. The graph on its own returns unordered sets
rather than ranked chunks, so it has nothing to hand to Stage 6 and is scored on
its own terms.
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from cognitive_kitchen.ui import models as MODELS
from cognitive_kitchen.ui import warmup

from cognitive_kitchen.config import settings
from cognitive_kitchen.rag import pipeline as P
from cognitive_kitchen.rag import registry
from cognitive_kitchen.rag.corpus import build_corpus, summarise
from cognitive_kitchen.rag.loaders import load_latest_ingested
from cognitive_kitchen.ui import explain
from cognitive_kitchen.ui.progress import Commentary, per_question

st.set_page_config(page_title="RAG Lab", page_icon="🧪", layout="wide")

for package in ("chunking", "embedding", "retrieval", "query", "generate",
                "strategies"):
    registry.discover(f"cognitive_kitchen.rag.{package}")

SS = st.session_state
SS.setdefault("locked", {})
SS.setdefault("results", {})

# Idempotent: no-op if the console already started it.
warmup.start()


# ---------------------------------------------------------------- resources
@st.cache_resource(show_spinner=False)
def corpus_for(source: str):
    return build_corpus(load_latest_ingested(source_type=source or None),
                        source=source)


@st.cache_resource(show_spinner=False)
def embedder():
    return registry.build("embedder", "st")


@st.cache_data(show_spinner=False)
def chunk(source: str, chunker: str, size: int, overlap: int):
    """Build a chunker, passing only parameters it actually accepts.

    The previous version guessed at "size" and "overlap" and caught TypeError to
    recover. The factories take "chunk_size", so the guess always failed and the
    fallback quietly used defaults -- the size and overlap controls in Stage 2
    did nothing at all, silently. Asking the registry what a factory takes is
    both correct and self-maintaining as chunkers are added.
    """
    accepted = registry.signature("chunker", chunker)
    kwargs: dict = {}
    if "chunk_size" in accepted:
        kwargs["chunk_size"] = size
    if "overlap" in accepted:
        kwargs["overlap"] = overlap
    if "embedder" in accepted:
        kwargs["embedder"] = embedder()
    return registry.build("chunker", chunker, **kwargs).split(corpus_for(source))


def lock(stage: str, value) -> None:
    SS["locked"][stage] = value
    st.rerun()


def is_locked(stage: str) -> bool:
    return stage in SS["locked"]


def show_models(stage: str) -> None:
    """Name the models a stage actually runs, so nothing is a black box."""
    pairs = explain.models_in_play(stage)
    if pairs:
        st.caption("  ·  ".join(f"**{role}** {name}" for role, name in pairs))


def show_options(title: str, table: dict, keys) -> None:
    with st.expander(f"What each {title} does"):
        for key in keys:
            if key in table:
                headline, detail = table[key]
                st.markdown(f"**`{key}`** — {headline}  \n{detail}")


def show_metrics(group: str) -> None:
    with st.expander("What the metrics mean, and how each is computed"):
        for name, (formula, detail) in explain.METRICS.get(group, {}).items():
            st.markdown(f"**{name}**")
            st.code(formula, language="text")
            st.caption(detail)


def graph_offline(exc: Exception) -> None:
    """Say the graph is unreachable, instead of printing a routing stack trace.

    Bolt is port 7687, which corporate VPNs routinely block, so this is the most
    common failure on this page and it has nothing to do with the query.
    """
    st.warning("The graph is not reachable, so this panel cannot answer right "
               "now. Every stage that does not use the graph still works.")
    with st.expander("Details"):
        st.caption(f"{type(exc).__name__}: {exc}")
        st.caption("Neo4j speaks Bolt on port 7687. A VPN blocking that port "
                   "produces exactly this, even when the instance is healthy "
                   "and port 443 on the same host is open.")


def best_of(frame: pd.DataFrame, column: str, high: bool = True):
    if column not in frame or frame[column].isna().all():
        return None
    return frame[column].max() if high else frame[column].min()


# ---------------------------------------------------------------- header
st.title("RAG Lab")

source = st.sidebar.selectbox("Corpus", ["pdf", "url", ""],
                              format_func=lambda s: {"pdf": "PDF run",
                                                     "url": "Web run",
                                                     "": "newest run"}[s])

# ---------------------------------------------------------------- models
# Every factory already takes model_name; this is the control that was missing.
# Switching one clears the resource cache, because a cached embedder would
# otherwise outlive the change and label a table with a model it did not use.
with st.sidebar:
    st.divider()
    st.subheader("Models")
    st.caption("Re-measure any stage on a different model. Verify before a long "
               "run, or a bad id fails forty seconds in.")
    for role in ("embedding_model", "generation_model", "reranker_model"):
        options = MODELS.KNOWN[role] + [MODELS.OTHER]
        now = MODELS.current(role)
        picked = st.selectbox(
            MODELS.LABELS[role], options,
            index=options.index(now) if now in options else len(options) - 1,
            key="pick-" + role, format_func=lambda s: s.split("/")[-1])
        if picked == MODELS.OTHER:
            picked = st.text_input("model id", value=now, key="free-" + role,
                                   placeholder="org/model-name")
        left, right = st.columns(2)
        if left.button("Use", key="use-" + role, width="stretch"):
            if MODELS.apply(role, picked):
                st.cache_resource.clear()
                st.cache_data.clear()
                SS["results"] = {}
                st.rerun()
        if right.button("Verify", key="ver-" + role, width="stretch"):
            SS["verdict-" + role] = MODELS.verify(role, picked)
        verdict = SS.get("verdict-" + role)
        if verdict:
            ok, detail = verdict
            (st.success if ok else st.error)(detail)

# The header used to call client.health() on every render, which is a Neo4j
# round trip. Streamlit re-runs the whole script on any navigation, so opening
# the Lab meant waiting on the network before the page appeared. Both badges are
# cached now: the numbers move rarely, and a stale-by-two-minutes node count is
# worth far more than a page that stalls every time you come back.
@st.cache_data(ttl=300, show_spinner=False)
def _vocab_badge() -> str:
    try:
        from cognitive_kitchen.rag.vocab import ingredients as vocab

        return f"{len(vocab.all_canonical())} ingredients"
    except Exception:
        return "not built"


@st.cache_data(ttl=120, show_spinner=False)
def _graph_badge() -> str:
    try:
        from cognitive_kitchen.rag.graph import client

        health = client.health()
        return f"{health['nodes']:,} nodes" if health["ok"] else "offline"
    except Exception:
        return "offline"


info = summarise(corpus_for(source))
cols = st.columns(4)
cols[0].metric("Recipes", info["recipes"])
cols[1].metric("Characters", f"{info['characters']:,}")
cols[2].metric("Vocabulary", _vocab_badge())
cols[3].metric("Graph", _graph_badge())

crumbs = []
for stage, label in (("chunker", "Chunker"), ("retrieval", "Retrieval"),
                     ("transform", "Query"), ("strategy", "Generation")):
    value = SS["locked"].get(stage)
    crumbs.append(f"**{label}: {value}**" if value else f"{label}: —")
st.caption("  ·  ".join(crumbs))
if SS["locked"]:
    if st.button("Reset pipeline"):
        SS["locked"] = {}
        SS["results"] = {}
        st.rerun()
st.divider()


# ================================================== STAGE 2 · CHUNKING
with st.expander("STAGE 2 · Chunking",
                 expanded=not is_locked("chunker")):
    st.caption("How the recipes are cut up. Everything downstream ranks these "
               "pieces, so a chunk that mixes two dishes poisons every later stage.")
    show_models("stage2")
    if is_locked("chunker"):
        st.success(f"Locked: **{SS['locked']['chunker']}**")
    else:
        available = registry.available("chunker")
        chosen = st.multiselect("Compare", available, default=available)
        show_options("chunker", explain.CHUNKERS, chosen or available)
        show_metrics("stage2")
        c1, c2, c3 = st.columns(3)
        size = c1.number_input("chunk size", 200, 3000, 600, 50,
                               help="Target characters per chunk. Used by naive "
                                    "and recursive; the semantic chunkers "
                                    "decide their own boundaries and only take "
                                    "this as an upper bound.")
        overlap = c2.number_input("overlap", 0, 400, 80, 10,
                                  help="Characters repeated between "
                                       "consecutive chunks, so a sentence cut "
                                       "in half still appears whole somewhere. "
                                       "Costs storage and can inflate "
                                       "diversity. naive and recursive only.")
        sample = c3.number_input("self-sufficiency sample", 10, 184, 40, 10,
                                 help="How many recipes to test self-"
                                      "sufficiency on. Each one embeds its "
                                      "chunks and searches all 184 recipe "
                                      "cards, so this is the slow part.")

        if st.button("Run comparison", type="primary", key="run2"):
            from cognitive_kitchen.rag.eval.golden import load_golden
            from cognitive_kitchen.rag.eval.stage2_chunking import evaluate

            note = Commentary("Comparing chunkers", total=len(chosen))
            note.say(f"corpus {info['recipes']} recipes, "
                     f"{info['characters']:,} characters")
            note.say(f"golden fingerprints loaded for scoring")
            note.say(f"embeddings: {settings.embedding_model.split('/')[-1]}")
            rows = []
            try:
                for name in chosen:
                    note.step(f"chunking with {name}")
                    started = time.perf_counter()
                    passages = chunk(source, name, size, overlap)
                    note.result(f"{len(passages)} chunks, mean "
                                f"{sum(len(p.text) for p in passages) // max(len(passages), 1)}"
                                f" chars, {time.perf_counter() - started:.1f}s")
                    note.result(f"scoring purity and recall, then embedding "
                                f"{int(sample)} recipes for self-sufficiency")
                    metrics = evaluate(corpus_for(source), passages,
                                       embedder=embedder(), golden=load_golden(),
                                       self_suff_sample=int(sample))
                    note.result(f"purity {metrics.get('word_purity')} · "
                                f"recall {metrics.get('word_recall_best')} · "
                                f"k@90 {metrics.get('k_at_90')} · "
                                f"self-suff {metrics.get('self_sufficiency')}")
                    rows.append({"chunker": name, "chunks": len(passages),
                                 **{k: v for k, v in metrics.items()
                                    if k != "elapsed_s"},
                                 "secs": round(time.perf_counter() - started, 1)})
                SS["results"]["stage2"] = rows
                best = max(rows, key=lambda r: r.get("word_purity") or 0)
                note.finish(f"{len(rows)} chunkers compared · best purity "
                            f"{best['chunker']}")
            except Exception as exc:
                note.fail(f"{type(exc).__name__}: {exc}")

        rows = SS["results"].get("stage2")
        if rows:
            frame = pd.DataFrame(rows)
            st.dataframe(frame, width="stretch", hide_index=True)
            top = frame.sort_values("word_purity", ascending=False).iloc[0]
            st.info(
                f"**{top['chunker']}** has the highest purity "
                f"({top['word_purity']:.3f}). Purity is how much of a chunk "
                f"belongs to one recipe; a low score means the retriever cannot "
                f"return one dish without dragging in another.\n\n"
                "A chunker that fragments recipes scores high purity and low "
                "recall, which reads well and retrieves badly. If you plan to "
                "use the graph pre-filter, prefer purity: a chunk spanning two "
                "recipes inherits both recipe ids and slips the gate if either "
                "one qualifies.")
            pick = st.selectbox("Lock", frame["chunker"].tolist(),
                                index=int(frame["word_purity"].argmax()))
            if st.button(f"Lock {pick} → Stage 3", type="primary", key="lock2"):
                SS["chunk_params"] = {"size": int(size), "overlap": int(overlap)}
                lock("chunker", pick)


# ================================================== STAGE 3 · RETRIEVAL
if is_locked("chunker"):
    with st.expander("STAGE 3 · Retrieval", expanded=not is_locked("retrieval")):
        st.caption("Four independent decisions, not one list of eight. The graph "
                   "decides which recipes are eligible; the rest decide order.")
        show_models("stage3")
        if is_locked("retrieval"):
            st.success(f"Locked: **{SS['locked']['retrieval']}**")
        else:
            passages = chunk(source, SS["locked"]["chunker"],
                             *(SS.get("chunk_params", {"size": 600,
                                                       "overlap": 80}).values()))
            st.caption(f"{len(passages)} chunks from "
                       f"{SS['locked']['chunker']}")

            c1, c2 = st.columns([2, 3])
            with c1:
                st.markdown("**① Candidate source**")
                sources = st.multiselect("compare",
                                         ["dense", "bm25", "tfidf", "hybrid", "rrf"],
                                         default=["bm25", "rrf"])
                sparse = st.radio("sparse half of a fuser", ["bm25", "tfidf"],
                                  horizontal=True,
                                  help="hybrid and rrf each combine dense with "
                                       "ONE sparse retriever. This picks which.")
            with c2:
                st.markdown("**② ③ ④ Modifiers** — compared on and off")
                try_mmr = st.checkbox("diversity (MMR)", value=False)
                try_rerank = st.checkbox("cross-encoder rerank", value=True)
                try_graph = st.checkbox("graph pre-filter", value=True)
                k = st.slider("k", 3, 10, 5,
                              help="Chunks handed to generation. Stage 2's "
                                   "k_at_90 tells you what this should be for "
                                   "your chunker rather than guessing.")
                n_questions = st.slider("questions", 5, 40, 20, 5,
                                        help="Sampled evenly across all seven "
                                             "golden query families. Only about "
                                             "a fifth restrict anything, so "
                                             "raise this if you want the "
                                             "compliance column to be stable.")
            show_options("candidate source", explain.SOURCES, sources or [])
            show_options("modifier", explain.MODIFIERS, ["mmr", "rerank", "graph"])
            show_metrics("stage3")

            if st.button("Run comparison", type="primary", key="run3"):
                from cognitive_kitchen.rag.eval.stage3_ranking import evaluate

                combos = []
                for base in sources:
                    combos.append((base, False, False, False))
                    if try_mmr:
                        combos.append((base, True, False, False))
                    if try_rerank:
                        combos.append((base, False, True, False))
                    if try_graph:
                        combos.append((base, False, False, True))
                    if try_rerank and try_graph:
                        combos.append((base, False, True, True))

                note = Commentary("Comparing retrievers", total=len(combos))
                note.say(f"{len(passages)} chunks from {SS['locked']['chunker']}"
                         f" · k={k} · {int(n_questions)} questions")
                note.say(f"embeddings {settings.embedding_model.split('/')[-1]}"
                         + (f" · reranker {settings.reranker_model.split('/')[-1]}"
                            if try_rerank else "")
                         + (f" · constraints {settings.judge_model}"
                            if try_graph else ""))
                # The sweep takes a minute or more. Rendering only at the
                # end leaves the screen blank for all of it, which reads as a
                # hang. This paints every configuration up front as queued and
                # fills each row as it lands, so progress and scope are both
                # visible and partial results are readable before the end.
                rows = []
                labels = [base + (" +mmr" if mmr else "")
                          + (" +rerank" if rerank else "")
                          + (" +graph" if graph else "")
                          for base, mmr, rerank, graph in combos]
                status = ["queued"] * len(labels)
                landed: dict[str, dict] = {}
                slot = st.empty()

                def paint_partial() -> None:
                    slot.dataframe(pd.DataFrame([
                        {"configuration": name, "status": status[i],
                         "hit@k": landed.get(name, {}).get("hit@k"),
                         "constraint": landed.get(name, {}).get("constraint"),
                         "secs": landed.get(name, {}).get("secs")}
                        for i, name in enumerate(labels)]),
                        width="stretch", hide_index=True)

                paint_partial()
                for index, (base, mmr, rerank, graph) in enumerate(combos):
                    status[index] = "running"
                    paint_partial()
                    spec = P.Pipeline(source=base, sparse=sparse, use_mmr=mmr,
                                      use_rerank=rerank, use_graph=graph)
                    name, params = spec.retriever_spec()
                    if name in ("dense", "hybrid", "rrf", "mmr", "cross_encoder",
                                "graph_hybrid"):
                        params["embedder"] = embedder()
                    label = (base + (" +mmr" if mmr else "")
                             + (" +rerank" if rerank else "")
                             + (" +graph" if graph else ""))
                    note.step(label, f"composed as {name}")
                    started = time.perf_counter()
                    try:
                        retriever = registry.build("retriever", name, **params)
                        note.result("building the index")
                        retriever.index(passages)
                        note.result("scoring questions")
                        metrics = evaluate(retriever, corpus_for(source), k=k,
                                           limit=int(n_questions),
                                           progress=per_question(note, every=5))
                        cr = metrics["constraint_respected"]
                        rows.append({
                            "source": base,
                            "mmr": "✓" if mmr else "", "rerank": "✓" if rerank else "",
                            "graph": "✓" if graph else "",
                            "hit@k": metrics["hit_at_k"],
                            "recall@k": metrics["recall_at_k"],
                            "MAP": metrics["map"],
                            "diversity": metrics["diversity_at_k"],
                            "constraint": cr,
                            "secs": round(time.perf_counter() - started, 1),
                            "_spec": (base, mmr, rerank, graph),
                            "_breaches": metrics.get("breaches") or [],
                        })
                        landed[labels[index]] = {
                            "hit@k": metrics["hit_at_k"], "constraint": cr,
                            "secs": round(time.perf_counter() - started, 1)}
                        status[index] = "done"
                        paint_partial()
                        note.result(f"hit@{k} {metrics['hit_at_k']} · "
                                    f"recall {metrics['recall_at_k']} · "
                                    f"MAP {metrics['map']} · constraint "
                                    f"{metrics['constraint_respected']} · "
                                    f"{time.perf_counter() - started:.1f}s")
                    except Exception as exc:
                        note.warn(f"{name}: {type(exc).__name__}: {exc}")
                        status[index] = "failed"
                        paint_partial()
                # the full table renders below; drop the running preview
                slot.empty()
                SS["results"]["stage3"] = rows
                note.finish(f"{len(rows)} configurations compared")

            rows = SS["results"].get("stage3")
            if rows:
                frame = pd.DataFrame(rows).drop(columns=["_spec", "_breaches"])
                st.dataframe(frame, width="stretch", hide_index=True,
                             column_config={
                                 "constraint": st.column_config.NumberColumn(
                                     "constraint", format="%.0f%%",
                                     help="Of the recipes handed to generation, "
                                          "how many satisfy what the question "
                                          "ruled out. Only scored on restricting "
                                          "questions.")})
                quality = frame.loc[frame["hit@k"].idxmax()]
                safe = frame[frame["constraint"] == frame["constraint"].max()]
                st.info(
                    f"**Best quality:** {quality['source']} "
                    f"{'+rerank' if quality['rerank'] else ''}"
                    f"{'+graph' if quality['graph'] else ''} at hit@k "
                    f"{quality['hit@k']:.3f} in {quality['secs']}s.\n\n"
                    f"**Compliance is a separate axis.** The four quality "
                    f"metrics cannot see it. Asked \"I am avoiding nuts\", a "
                    f"dense retriever returns Nut Milk and a cross-encoder "
                    f"returns Nut Milk, Cashew Chutney and Almond Milk: the "
                    f"better the ranker, the more confidently wrong, because "
                    f"\"avoiding nuts\" sits closest in embedding space to "
                    f"recipes about nuts. Only the graph rows reach 100%.")
                breaches = [b for r in rows for b in r["_breaches"]]
                if breaches:
                    with st.expander(f"{len(breaches)} constraint breaches"):
                        st.dataframe(pd.DataFrame(breaches), width="stretch",
                                     hide_index=True)
                labels = [f"{r['source']}"
                          f"{' +mmr' if r['mmr'] else ''}"
                          f"{' +rerank' if r['rerank'] else ''}"
                          f"{' +graph' if r['graph'] else ''}" for r in rows]
                pick = st.selectbox("Lock", range(len(labels)),
                                    format_func=lambda i: labels[i])
                if st.button(f"Lock {labels[pick]} → Stage 4", type="primary",
                             key="lock3"):
                    base, mmr, rerank, graph = rows[pick]["_spec"]
                    SS["retrieval_spec"] = {"source": base, "sparse": sparse,
                                            "use_mmr": mmr, "use_rerank": rerank,
                                            "use_graph": graph, "k": int(k)}
                    lock("retrieval", labels[pick])


# ================================================== STAGE 4 · QUERY
if is_locked("retrieval"):
    with st.expander("STAGE 4 · Query transform",
                     expanded=not is_locked("transform")):
        st.caption("Rewrite the question before it reaches the retriever.")
        show_models("stage4")
        if is_locked("transform"):
            st.success(f"Locked: **{SS['locked']['transform']}**")
        else:
            options = registry.available("query")
            chosen = st.multiselect("Compare", options,
                                    default=[o for o in options if o != "hyde"])
            show_options("transform", explain.TRANSFORMS, chosen or options)
            show_metrics("stage3")
            n = st.slider("questions", 3, 20, 5, key="n4")
            if st.button("Run comparison", type="primary", key="run4"):
                from cognitive_kitchen.rag.eval.stage3_ranking import evaluate

                passages = chunk(source, SS["locked"]["chunker"],
                                 *(SS.get("chunk_params",
                                          {"size": 600, "overlap": 80}).values()))
                spec = P.Pipeline(**SS["retrieval_spec"])
                name, params = spec.retriever_spec()
                if name in ("dense", "hybrid", "rrf", "mmr", "cross_encoder",
                            "graph_hybrid"):
                    params["embedder"] = embedder()
                retriever = registry.build("retriever", name, **params)
                retriever.index(passages)

                note = Commentary("Comparing query transforms", total=len(chosen))
                note.say(f"retriever {SS['locked']['retrieval']} · "
                         f"{int(n)} questions")
                rows = []
                for transform_name in chosen:
                    note.step(transform_name,
                              "one generation call per query"
                              if transform_name == "hyde" else "")
                    transform = (None if transform_name == "passthrough"
                                 else registry.build("query", transform_name))
                    started = time.perf_counter()
                    try:
                        metrics = evaluate(retriever, corpus_for(source),
                                           query_transform=transform, k=spec.k,
                                           limit=int(n),
                                           progress=per_question(note))
                        note.result(f"hit {metrics['hit_at_k']} · MAP "
                                    f"{metrics['map']} · "
                                    f"{time.perf_counter() - started:.1f}s")
                        rows.append({"transform": transform_name,
                                     "hit@k": metrics["hit_at_k"],
                                     "recall@k": metrics["recall_at_k"],
                                     "MAP": metrics["map"],
                                     "constraint": metrics["constraint_respected"],
                                     "secs": round(time.perf_counter() - started, 1)})
                    except Exception as exc:
                        note.warn(f"{transform_name}: {exc}")
                SS["results"]["stage4"] = rows
                note.finish(f"{len(rows)} transforms compared")

            rows = SS["results"].get("stage4")
            if rows:
                frame = pd.DataFrame(rows)
                st.dataframe(frame, width="stretch", hide_index=True)
                st.info("decompose splits a compound question and retrieves for "
                        "each part. hyde writes a hypothetical recipe first and "
                        "searches with that, which costs around 400s per query "
                        "on CPU for no measured gain here — leave it until the "
                        "GPU machine.")
                pick = st.selectbox("Lock", frame["transform"].tolist(),
                                    index=int(frame["hit@k"].argmax()))
                if st.button(f"Lock {pick} → Stage 6", type="primary", key="lock4"):
                    lock("transform", pick)


# ================================================== STAGE 6 · GENERATION
if is_locked("transform"):
    with st.expander("STAGE 6 · Generation", expanded=not is_locked("strategy")):
        st.caption("Four ways to turn the retrieved chunks into an answer.")
        show_models("stage6")
        if is_locked("strategy"):
            st.success(f"Locked: **{SS['locked']['strategy']}**")
        else:
            options = registry.available("strategy")
            chosen = st.multiselect("Compare", options, default=options)
            show_options("strategy", explain.STRATEGIES, chosen or options)
            show_metrics("stage6")
            c1, c2, c3 = st.columns(3)
            gen_name = c1.selectbox("generator", registry.available("generator"))
            tokens = c2.number_input("max_new_tokens", 60, 512, 120, 20,
                                     help="Answer length cap. On CPU the local "
                                          "model produces roughly 3 tokens a "
                                          "second, so this is the main lever on "
                                          "how long an answer takes.")
            n = c3.slider("questions", 3, 10, 5, key="n6")
            judged = st.checkbox("Judge faithfulness and cookability "
                                 "(gpt-4o-mini, about $0.0003 a strategy)",
                                 value=True)
            if gen_name == "qwen":
                st.warning("The local model answers in roughly a minute per "
                           "question on CPU, and map_reduce makes one call per "
                           "chunk. Expect a long run.")

            if st.button("Run comparison", type="primary", key="run6"):
                from cognitive_kitchen.rag.eval import stage6_generation as S6
                from cognitive_kitchen.rag.eval.judge import Judge

                passages = chunk(source, SS["locked"]["chunker"],
                                 *(SS.get("chunk_params",
                                          {"size": 600, "overlap": 80}).values()))
                spec = P.Pipeline(**SS["retrieval_spec"])
                name, params = spec.retriever_spec()
                if name in ("dense", "hybrid", "rrf", "mmr", "cross_encoder",
                            "graph_hybrid"):
                    params["embedder"] = embedder()
                retriever = registry.build("retriever", name, **params)
                retriever.index(passages)
                transform = (None if SS["locked"]["transform"] == "passthrough"
                             else registry.build("query",
                                                 SS["locked"]["transform"]))
                generator = registry.build("generator", gen_name,
                                           max_new_tokens=int(tokens))

                note = Commentary("Comparing generation strategies",
                                  total=len(chosen))
                note.say(f"pipeline {SS['locked']['chunker']} -> "
                         f"{SS['locked']['retrieval']} -> "
                         f"{SS['locked']['transform']}")
                note.say(f"generator {gen_name} · max_new_tokens {int(tokens)} · "
                         f"{int(n)} questions"
                         + (f" · judge {settings.judge_model}" if judged else ""))
                if gen_name == "qwen":
                    note.say("local model on CPU: expect around a minute an "
                             "answer, and map_reduce makes one call per chunk")
                rows = []
                for strategy_name in chosen:
                    note.step(strategy_name,
                              f"{int(n)} x {spec.k + 1} calls"
                              if strategy_name == "map_reduce"
                              else f"{int(n)} calls")
                    strategy = registry.build("strategy", strategy_name,
                                              generator=generator)
                    started = time.perf_counter()
                    metrics = S6.evaluate(strategy, retriever, corpus_for(source),
                                          query_transform=transform, k=spec.k,
                                          limit=int(n),
                                          judge=Judge() if judged else None,
                                          progress=per_question(note))
                    note.result(f"invented {metrics['no_invented_ingredients']} · "
                                f"abstain {metrics['honest_abstention']} · "
                                f"faithful {metrics.get('faithfulness')} · "
                                f"relevancy {metrics.get('answer_relevancy')} · "
                                f"relevancy {metrics.get('answer_relevancy')} · "
                                f"cookable {metrics.get('cookable')} · "
                                f"{time.perf_counter() - started:.1f}s")
                    for failure in (metrics.get("judge_failures") or []):
                        note.warn(failure)
                    for failure in (metrics.get("judge_failures") or []):
                        note.warn(failure)
                    rows.append({
                        "strategy": strategy_name,
                        "NoInvented": metrics["no_invented_ingredients"],
                        "Abstention": metrics["honest_abstention"],
                        "Faithful": metrics.get("faithfulness"),
                        "Relevancy": metrics.get("answer_relevancy"),
                        "Cookable": metrics.get("cookable"),
                        "refusals": metrics["refusal_rate"],
                        "s/answer": metrics["mean_answer_seconds"],
                        "judge $": metrics.get("judge_cost_usd", 0.0),
                        "_rows": metrics["rows"],
                    })
                SS["results"]["stage6"] = rows
                SS["gen_choice"] = {"generator": gen_name,
                                    "max_new_tokens": int(tokens)}
                note.finish(f"{len(rows)} strategies compared")

            rows = SS["results"].get("stage6")
            if rows:
                frame = pd.DataFrame(rows).drop(columns=["_rows"])
                st.dataframe(frame, width="stretch", hide_index=True)
                st.info(
                    "NoInvented and Abstention are a safety floor, not a "
                    "ranking: all four strategies are grounded, so a score "
                    "below 1.0 is a defect to fix rather than a strategy to "
                    "reject. Faithfulness and Cookable do the discriminating.\n\n"
                    "Watch reordered against stuff_strict. Same chunks, same "
                    "count, same cost — only the order changes. A gap between "
                    "them is the lost-in-the-middle effect, and no amount of "
                    "better retrieval would have revealed it.")
                with st.expander("Answers"):
                    for row in rows:
                        st.markdown(f"**{row['strategy']}**")
                        st.dataframe(pd.DataFrame(row["_rows"])[
                            ["question", "refused", "no_invented",
                             "abstention"]], width="stretch", hide_index=True)
                pick = st.selectbox("Lock", frame["strategy"].tolist())
                if st.button(f"Lock {pick} and save pipeline", type="primary",
                             key="lock6"):
                    lock("strategy", pick)


# ================================================== SAVE
if is_locked("strategy"):
    st.divider()
    config = P.Pipeline(
        chunker=SS["locked"]["chunker"],
        chunker_params=SS.get("chunk_params", {}),
        **SS["retrieval_spec"],
        query_transform=SS["locked"]["transform"],
        strategy=SS["locked"]["strategy"],
        corpus_source=source,
        **SS.get("gen_choice", {}))
    st.subheader("Pipeline")
    st.code(config.label(), language="text")
    if st.button("Save and open the Kitchen", type="primary"):
        P.save(config)
        st.success(f"Saved to {P.path()}")
        st.page_link("pages/3_Kitchen.py", label="Open the Kitchen", icon="🍲")


# ================================================== STAGE 5 · standalone
st.divider()
with st.expander("STAGE 5 · Pure graph — standalone, not part of the chain"):
    st.caption("No chunks, no embeddings, no ranking. A traversal returns every "
               "recipe satisfying the condition, unordered and complete, so it "
               "is scored on set precision and recall rather than MAP.")
    show_models("stage5")
    show_metrics("stage5")
    tab_scored, tab_gen, tab_pantry, tab_subs = st.tabs(
        ["Retrieval scored", "End to end", "Try it: pantry",
         "Try it: substitutes"])

    with tab_scored:
        n5 = st.slider("questions", 5, 40, 20, 5, key="n5")
        if st.button("Run", type="primary", key="run5"):
            from cognitive_kitchen.rag.eval import stage5_graph as S5

            note = Commentary("Traversing the graph", total=int(n5))
            note.say("no chunks, no embeddings, no ranking")
            note.say(f"constraint reader {settings.judge_model}, cached by question")
            try:
                note.step("mapping golden recipes onto pipeline ids by content")
                SS["results"]["stage5"] = S5.evaluate(
                    corpus_for(source), limit=int(n5),
                    progress=per_question(note))
                r = SS["results"]["stage5"]
                note.finish(f"precision {r['set_precision']} · recall "
                            f"{r['set_recall']} · respected "
                            f"{r['constraint_respected']}")
            except Exception as exc:
                note.fail(f"{type(exc).__name__}: {exc}")
        result = SS["results"].get("stage5")
        if result:
            cols = st.columns(5)
            cols[0].metric("set precision", result["set_precision"])
            cols[1].metric("set recall", result["set_recall"])
            cols[2].metric("set F1", result["set_f1"])
            cols[3].metric("constraint extracted", result["constraint_extracted"])
            cols[4].metric("constraint respected", result["constraint_respected"])
            st.caption(
                f"Precision is scored only over the {result['judged_recipes']} "
                f"recipes the golden dataset covers, out of "
                f"{result['corpus_recipes']} in the corpus. Asked \"what can I "
                f"cook with cumin\" the graph returns every recipe that really "
                f"contains cumin; counting the ones golden simply has no opinion "
                f"about as errors would measure the truth set's coverage rather "
                f"than the graph.")
            st.dataframe(pd.DataFrame(result["rows"])[
                ["question", "n_returned", "n_judged_returned", "n_expected",
                 "precision", "recall"]], width="stretch", hide_index=True)

    with tab_gen:
        st.caption("Answer from the graph alone, then score on exactly the "
                   "metrics Stage 6 uses. Set precision says the traversal found "
                   "the right recipes; it says nothing about whether an answer "
                   "built from them is faithful, relevant or followable. Same "
                   "metrics, same judge, one table — so the two routes are "
                   "genuinely comparable.")
        c1, c2, c3 = st.columns(3)
        g_strategy = c1.selectbox("strategy", registry.available("strategy"),
                                  key="g5strat")
        g_gen = c2.selectbox("generator", registry.available("generator"),
                             key="g5gen")
        g_n = c3.slider("questions", 3, 10, 5, key="g5n")
        also_retrieval = st.checkbox(
            "Also run the locked retrieval pipeline, for comparison",
            value=is_locked("retrieval"),
            disabled=not is_locked("retrieval"),
            help="Needs a locked Stage 3 choice to compare against.")
        g_judged = st.checkbox("Judge faithfulness, relevancy and cookability",
                               value=True, key="g5judge")

        if st.button("Run", type="primary", key="run5gen"):
            from cognitive_kitchen.rag.eval import stage5_graph as S5
            from cognitive_kitchen.rag.eval import stage6_generation as S6
            from cognitive_kitchen.rag.eval.judge import Judge

            total = 2 if also_retrieval else 1
            note = Commentary("Answering from the graph", total=total)
            note.say(f"strategy {g_strategy} · generator {g_gen} · "
                     f"{int(g_n)} questions")
            note.say("the graph has no relevance ranking, so which k of the "
                     "matching recipes reach the generator is a heuristic: "
                     "fewest ingredients first")
            rows = []
            try:
                generator = registry.build("generator", g_gen)
                note.step("graph_only route")
                result = S5.evaluate_generation(
                    corpus_for(source), strategy_name=g_strategy,
                    generator=generator, k=5, limit=int(g_n),
                    judge=Judge() if g_judged else None,
                    progress=per_question(note))
                rows.append({"route": result["route"], **{
                    "NoInvented": result["no_invented_ingredients"],
                    "Abstention": result["honest_abstention"],
                    "Faithful": result.get("faithfulness"),
                    "Relevancy": result.get("answer_relevancy"),
                    "Cookable": result.get("cookable"),
                    "refusals": result["refusal_rate"],
                    "s/answer": result["mean_answer_seconds"]}})
                for failure in (result.get("judge_failures") or []):
                    note.warn(failure)
                SS["results"]["stage5gen_trace"] = result.get("trace") or []

                if also_retrieval and is_locked("retrieval"):
                    note.step("locked retrieval route")
                    passages = chunk(source, SS["locked"]["chunker"],
                                     *(SS.get("chunk_params",
                                              {"size": 600, "overlap": 80}).values()))
                    spec = P.Pipeline(**SS["retrieval_spec"])
                    name, params = spec.retriever_spec()
                    if name in ("dense", "hybrid", "rrf", "mmr", "cross_encoder",
                                "graph_hybrid"):
                        params["embedder"] = embedder()
                    retriever = registry.build("retriever", name, **params)
                    retriever.index(passages)
                    strategy = registry.build("strategy", g_strategy,
                                              generator=generator)
                    other = S6.evaluate(strategy, retriever, corpus_for(source),
                                        k=spec.k, limit=int(g_n),
                                        judge=Judge() if g_judged else None,
                                        progress=per_question(note))
                    rows.append({
                        "route": f"{SS['locked']['chunker']} -> "
                                 f"{SS['locked']['retrieval']} -> {g_strategy}",
                        "NoInvented": other["no_invented_ingredients"],
                        "Abstention": other["honest_abstention"],
                        "Faithful": other.get("faithfulness"),
                        "Relevancy": other.get("answer_relevancy"),
                        "Cookable": other.get("cookable"),
                        "refusals": other["refusal_rate"],
                        "s/answer": other["mean_answer_seconds"]})
                SS["results"]["stage5gen"] = rows
                note.finish(f"{len(rows)} route(s) compared")
            except Exception as exc:
                note.fail(f"{type(exc).__name__}: {exc}")

        rows = SS["results"].get("stage5gen")
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.info(
                "A dish name is not a graph pattern. Asked \"give me the recipe "
                "for Samosa\", the graph can narrow to course=snack and no "
                "further, so the generator sees five snacks that are not samosa "
                "and correctly refuses. That shows up as a lower Abstention "
                "score and is the honest limit of answering from structure "
                "alone: it is strong on absence, weak on identity.")
            trace = SS["results"].get("stage5gen_trace") or []
            if trace:
                with st.expander("What the traversal matched"):
                    st.dataframe(pd.DataFrame([
                        {"question": x.get("query", "")[:52],
                         "matched": x.get("matched"), "shown": x.get("shown"),
                         "include": ", ".join((x.get("constraints") or {})
                                              .get("include") or []),
                         "excluded families": ", ".join(
                             (x.get("constraints") or {})
                             .get("exclude_categories") or []),
                         "course": (x.get("constraints") or {}).get("course", "")}
                        for x in trace]), width="stretch", hide_index=True)

    with tab_pantry:
        st.caption("Not scored - a demonstration. The answer here appears in no "
                   "document: it is (ingredients needed) minus (ingredients you "
                   "have), which is arithmetic over the graph and not something "
                   "any retriever can find.")
        have = st.text_input("I have", "onion, tomato, oil, salt, turmeric")
        missing = st.slider("missing at most", 0, 4, 2)
        if st.button("Find", key="pantry"):
            try:
                from cognitive_kitchen.rag.graph import traverse

                names = [x.strip() for x in have.split(",") if x.strip()]
                found = traverse.pantry_gap(names, max_missing=int(missing),
                                            limit=15)
                if found:
                    st.dataframe(pd.DataFrame(found)[
                        ["title", "n_missing", "n_needed", "missing"]],
                        width="stretch", hide_index=True)
                else:
                    st.info("Nothing is within that many missing ingredients.")
            except Exception as exc:
                graph_offline(exc)

    with tab_subs:
        st.caption("Not scored - a demonstration. Substitutes come from "
                   "distributional similarity: two ingredients are alike when "
                   "they keep the same company, even if they never meet. Plain "
                   "co-occurrence gives the wrong answer, because the things most "
                   "often found next to ghee are water and salt.")
        target = st.text_input("Instead of", "ghee")
        if st.button("Suggest", key="subs"):
            try:
                from cognitive_kitchen.rag.graph import traverse

                found = traverse.substitutes(target, limit=8)
                if found:
                    st.dataframe(pd.DataFrame(found), width="stretch",
                                 hide_index=True)
                else:
                    st.warning(f"{target!r} is not in the ingredient vocabulary.")
            except Exception as exc:
                graph_offline(exc)
