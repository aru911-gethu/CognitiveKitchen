"""Kitchen - the chat page.

No strategy pickers: it reads the pipeline the Lab saved and runs that. Choosing
happens in the Lab; this is where the choice gets used.

Conversations persist to data/chat/, and a follow-up is rewritten against the
recent turns before retrieval. That second part is the one that matters -- "what
about without dairy?" has no subject, so a retriever handed it alone returns
nothing useful. Without resolution, memory would be a transcript and nothing
more.
"""
from __future__ import annotations

import time

import streamlit as st

from cognitive_kitchen.config import settings
from cognitive_kitchen.rag import memory as M
from cognitive_kitchen.rag import pipeline as P

st.set_page_config(page_title="Kitchen", page_icon="🍲", layout="centered")
st.title("Kitchen")

config = P.load()
if config is None:
    st.warning("No pipeline saved yet. Open the RAG Lab, work through the "
               "stages, and lock a choice at each one.")
    st.page_link("pages/2_RAG_Lab.py", label="Go to the RAG Lab", icon="🧪")
    st.stop()

model_label = (settings.generation_model.split("/")[-1]
               if config.generator == "qwen" else config.generator)
st.caption(f"using **{config.label()}**  ·  k={config.k}  ·  {model_label}")


@st.cache_resource(show_spinner="Loading the pipeline...")
def runtime(signature: str):
    return P.build_runtime(P.load())


parts = runtime(config.label() + str(config.k) + str(config.max_new_tokens))

# ------------------------------------------------------------------ session
if "conversation_id" not in st.session_state:
    saved = M.listing()
    if saved:
        st.session_state["conversation_id"] = saved[0]["conversation_id"]
    else:
        fresh = M.Conversation(pipeline=config.label())
        M.save(fresh)
        st.session_state["conversation_id"] = fresh.conversation_id

conversation = M.load(st.session_state["conversation_id"])
if conversation is None:
    conversation = M.Conversation(pipeline=config.label())
    M.save(conversation)
    st.session_state["conversation_id"] = conversation.conversation_id

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.subheader("Conversations")
    if st.button("New conversation", width="stretch"):
        fresh = M.Conversation(pipeline=config.label())
        M.save(fresh)
        st.session_state["conversation_id"] = fresh.conversation_id
        st.rerun()

    for entry in M.listing()[:12]:
        active = entry["conversation_id"] == conversation.conversation_id
        label = ("● " if active else "") + entry["title"][:34]
        cols = st.columns([5, 1])
        if cols[0].button(label, key=f"open-{entry['conversation_id']}",
                          width="stretch",
                          help=f"{entry['turns']} messages"):
            st.session_state["conversation_id"] = entry["conversation_id"]
            st.rerun()
        if cols[1].button("×", key=f"del-{entry['conversation_id']}"):
            M.delete(entry["conversation_id"])
            st.session_state.pop("conversation_id", None)
            st.rerun()

    st.divider()
    st.subheader("Pantry")
    st.caption("What you have in. Answered by the graph, not by retrieval — "
               "'missing' appears in no document.")
    pantry_text = st.text_area("one per line",
                               value=st.session_state.get(
                                   "pantry", "onion\ntomato\noil\nsalt\nturmeric"),
                               height=130)
    st.session_state["pantry"] = pantry_text
    pantry = [line.strip() for line in pantry_text.splitlines() if line.strip()]

    if st.button("What can I make?", width="stretch"):
        try:
            from cognitive_kitchen.rag.graph import traverse

            st.session_state["pantry_results"] = traverse.pantry_gap(
                pantry, max_missing=2, limit=8)
        except Exception as exc:
            st.error(f"Graph unavailable: {exc}")

    for row in st.session_state.get("pantry_results", []) or []:
        st.markdown(f"**{row['title'][:36]}**  \nmissing {row['n_missing']} of "
                    f"{row['n_needed']}: {', '.join(row['missing']) or 'nothing'}")

# ------------------------------------------------------------------ history
for turn in conversation.turns:
    with st.chat_message(turn.role):
        st.markdown(turn.content)
        if turn.resolved:
            st.caption(f"↳ retrieved as: _{turn.resolved}_")
        if turn.meta:
            st.caption(turn.meta)

# ------------------------------------------------------------------ new turn
question = st.chat_input("Ask about a recipe, or what you can cook")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        started = time.perf_counter()

        with st.spinner("Reading the question..."):
            retrieval_query, rewritten = M.resolve(question, conversation)
        if rewritten:
            st.caption(f"↳ retrieved as: _{retrieval_query}_")

        conversation.add("user", question,
                         resolved=retrieval_query if rewritten else "")

        with st.spinner("Searching your recipes..."):
            queries = ([retrieval_query] if parts["transform"] is None
                       else parts["transform"].expand(retrieval_query))
            retriever = parts["retriever"]
            hits = (retriever.search_multi(queries, config.k)
                    if hasattr(retriever, "search_multi") and len(queries) > 1
                    else retriever.search(queries[0], config.k))
            contexts = [hit.passage for hit in hits]

        if not contexts:
            text = ("I could not find anything in your recipes that answers "
                    "that.")
            st.markdown(text)
            meta = f"0 sources · {time.perf_counter() - started:.1f}s"
        else:
            with st.spinner(f"Writing the answer ({len(contexts)} sources)..."):
                answer = parts["strategy"].generate(retrieval_query, contexts)
            text = answer.text
            st.markdown(text)

            badges = [f"{len(contexts)} sources",
                      f"{time.perf_counter() - started:.1f}s"]
            cost = getattr(parts["generator"], "cost_usd", 0.0)
            if cost:
                badges.append(f"${cost:.5f}")

            try:
                from cognitive_kitchen.rag.eval.stage3_ranking import (
                    recipe_ingredients)
                from cognitive_kitchen.rag.eval.stage6_generation import (
                    constraint_respected)
                from cognitive_kitchen.rag.graph.constraints import (
                    ConstraintExtractor)

                constraints = ConstraintExtractor().extract(retrieval_query)
                if constraints.restricting:
                    score = constraint_respected(
                        answer, constraints,
                        recipe_ingredients(config.corpus_source))
                    ruled_out = ", ".join(constraints.exclude_categories
                                          + constraints.exclude)
                    badges.append(f"✓ respects: {ruled_out}" if score == 1.0
                                  else f"⚠ may include {ruled_out}")
            except Exception:
                pass

            meta = " · ".join(badges)
            with st.expander("Sources"):
                for index, passage in enumerate(contexts, start=1):
                    ids = ", ".join(passage.meta.get("recipe_ids") or [])
                    st.markdown(f"**{index}. {ids}**")
                    st.text(passage.text[:600])

        st.caption(meta)
        conversation.add("assistant", text, meta=meta)
        M.save(conversation)