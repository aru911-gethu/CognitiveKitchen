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

            # The graph key of every cited recipe, kept for the pantry check
            # below. Streamlit reruns the script on a button press, so this has
            # to outlive the turn that produced it.
            st.session_state["last_keys"] = [
                f"{passage.meta.get('source')}:{(passage.meta.get('recipe_ids') or [''])[0]}"
                for passage in contexts
                if passage.meta.get("source") and passage.meta.get("recipe_ids")]

            meta = " · ".join(badges)
            with st.expander("Sources"):
                for index, passage in enumerate(contexts, start=1):
                    ids = ", ".join(passage.meta.get("recipe_ids") or [])
                    st.markdown(f"**{index}. {ids}**")
                    st.text(passage.text[:600])

        st.caption(meta)
        conversation.add("assistant", text, meta=meta)
        M.save(conversation)


# ------------------------------------------------- can I actually make it?
# The generator answers from retrieved text. This asks the graph whether the
# cook can act on that answer, which retrieval structurally cannot: "missing"
# appears in no document. The recipe is identified by the answer's cited source,
# never by parsing the answer text.
keys = st.session_state.get("last_keys") or []
if keys:
    st.divider()
    st.subheader("Can I actually make it?")
    st.caption("Checks the recipe this answer came from against your pantry. "
               "Answered by the graph — retrieval has no notion of what you lack.")

    if st.button("Check against my pantry", width="stretch"):
        with st.spinner("Resolving your pantry, then asking the graph..."):
            try:
                from cognitive_kitchen.rag.graph import pantry as PN

                resolution = PN.resolve_pantry(pantry)
                st.session_state["pantry_check"] = {
                    "resolution": resolution,
                    "checks": [PN.can_i_make(key, resolution.canonical)
                               for key in keys[:3]]}
            except Exception as exc:
                st.session_state["pantry_check"] = {"error": str(exc)}

    check = st.session_state.get("pantry_check") or {}
    if check.get("error"):
        st.error(f"Graph unavailable: {check['error']}")
    elif check.get("checks"):
        resolution = check["resolution"]
        note = f"read {len(resolution.canonical)} ingredients from your pantry"
        if resolution.unknown:
            note += f" · could not place: {', '.join(resolution.unknown)}"
        if resolution.llm_used:
            note += f" · a model resolved the leftovers (${resolution.cost_usd:.5f})"
        st.caption(note)

        for row in check["checks"]:
            if row["can_make"]:
                st.success(f"**{row['title']}** — you have everything, or "
                           f"something that stands in.")
            else:
                st.warning(f"**{row['title']}** — missing "
                           f"{len(row['missing'])} of {len(row['needed'])}")
            if row["buy"]:
                st.markdown("**Buy:** " + ", ".join(row["buy"]))
            for missing, covers in (row["swaps"] or {}).items():
                st.markdown(f"**Swap:** no {missing} — use "
                            f"{' or '.join(covers)}, already on your shelf")

