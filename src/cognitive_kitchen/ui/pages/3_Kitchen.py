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
from cognitive_kitchen.ui import chrome

chrome.page("Kitchen", icon="🍲", layout="centered")

config = P.load()
if config is None:
    st.warning("No pipeline saved yet. Open the RAG Lab, work through the "
               "stages, and lock a choice at each one.")
    st.page_link("pages/2_RAG_Lab.py", label="Go to the RAG Lab", icon="🧪")
    st.stop()

from cognitive_kitchen.rag.loaders import list_runs

# ------------------------------------------------------------------ sidebar datasets
runs = list_runs()
selected_dataset = None
if runs:
    options = {r["file"]: f"{r.get('origin') or r['file']} ({r.get('n_recipes', 0)} recipes)" for r in runs}
    files_list = list(options.keys())

    default_idx = 0
    active = st.session_state.get("active_dataset")
    if active and active in options:
        default_idx = files_list.index(active)
    else:
        for idx, f in enumerate(files_list):
            if "indian-dishes-for-you-to-try-at-home.pdf" in options[f]:
                default_idx = idx
                break

    with st.sidebar:
        st.subheader("Ingested Cookbook")
        selected_dataset = st.selectbox(
            "Active Dataset",
            options=files_list,
            index=default_idx,
            format_func=lambda f: options[f],
            help="Choose an ingested PDF or Web dataset to chat with."
        )
        st.divider()

model_label = (settings.generation_model.split("/")[-1]
               if config.generator == "qwen" else config.generator)
chrome.header("Kitchen",
              "Answers grounded in the recipes you ingested, using the pipeline "
              "you locked in the Lab.",
              pills=[config.label(), f"k={config.k}", model_label])


@st.cache_resource(show_spinner="Loading the pipeline...")
def runtime(signature: str, dataset_file: str | None = None):
    return P.build_runtime(P.load(), dataset_file=dataset_file)


parts = runtime(config.label() + str(config.k) + str(config.max_new_tokens) + str(selected_dataset), selected_dataset)

def meta_chips(meta: str) -> None:
    """Render the answer footer as chips.

    The footer is stored as a plain " . "-joined string because it is persisted
    with the conversation and re-read on load. Presentation is derived from it
    rather than stored, so old conversations pick up the new look for free.
    """
    if not meta:
        return
    out = ['<div class="metachips">']
    for part in [x.strip() for x in meta.split("\u00b7") if x.strip()]:
        kind = "ok" if part.startswith("\u2713") else (
            "warn" if part.startswith("\u26a0") else "")
        out.append(f'<span class="metachip {kind}">{part}</span>')
    out.append("</div>")
    st.markdown("".join(out), unsafe_allow_html=True)


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
    st.caption("What you actually have in. Used further down to check whether "
               "you can cook an answer.")
    pantry_text = st.text_area(
        "one per line",
        value=st.session_state.get("pantry",
                                   "onion\ntomato\noil\nsalt\nturmeric"),
        height=150)
    st.session_state["pantry"] = pantry_text
    pantry = [line.strip() for line in pantry_text.splitlines() if line.strip()]

# ------------------------------------------------------------------ history
STARTERS = ["what can I make with rice?",
            "something with no dairy",
            "what can I use instead of ghee?",
            "I am avoiding nuts, what can I make?"]

if not conversation.turns:
    st.markdown('<div class="emptychat"><h4>Ask about your recipes</h4>'
                "<p>Every answer is built only from what you ingested. If the "
                "answer is not in there, it says so rather than inventing "
                "one.</p></div>", unsafe_allow_html=True)
    st.caption("Or start with one of these:")
    for column, starter in zip(st.columns(2) + st.columns(2), STARTERS):
        if column.button(starter, key=f"start-{starter}", width="stretch"):
            st.session_state["pending_question"] = starter
            st.rerun()

for turn in conversation.turns:
    with st.chat_message(turn.role):
        st.markdown(turn.content)
        if turn.resolved:
            st.caption(f"↳ retrieved as: _{turn.resolved}_")
        if turn.meta:
            meta_chips(turn.meta)

# ------------------------------------------------------------------ new turn
question = st.chat_input("Ask about a recipe, or what you can cook")
question = question or st.session_state.pop("pending_question", None)
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
            # Key plus a human label, taken from the passage itself rather
            # than from the graph -- reading a title should not need a network
            # round trip, and this list is built on every answer.
            cited = []
            for passage in contexts:
                src = passage.meta.get("source")
                ids = passage.meta.get("recipe_ids") or []
                if not src or not ids:
                    continue
                head = next((ln.strip() for ln in passage.text.splitlines()
                             if ln.strip()), ids[0])
                cited.append({"key": f"{src}:{ids[0]}",
                              "label": head[:58]})
            seen, unique = set(), []
            for row in cited:
                if row["key"] not in seen:
                    seen.add(row["key"])
                    unique.append(row)
            st.session_state["last_sources"] = unique

            meta = " · ".join(badges)
            with st.expander("Sources"):
                for index, passage in enumerate(contexts, start=1):
                    ids = ", ".join(passage.meta.get("recipe_ids") or [])
                    st.markdown(f"**{index}. {ids}**")
                    st.text(passage.text[:600])

        meta_chips(meta)
        conversation.add("assistant", text, meta=meta)
        M.save(conversation)


# ------------------------------------------------- can I actually make it?
# The generator answers from retrieved text. This asks the graph whether the cook
# can act on that answer, which retrieval structurally cannot: "missing" appears
# in no document. Recipes are identified by the answer's cited sources, never by
# parsing the answer text.
sources = st.session_state.get("last_sources") or []
if sources:
    st.divider()
    st.subheader("Can I actually make it?")
    st.caption("Compare a recipe from the answer above against your pantry. "
               "Answered by the graph, not by retrieval.")

    labels = [row["label"] for row in sources]
    picked = st.selectbox("Which recipe?", range(len(labels)),
                          format_func=lambda i: labels[i])

    if not pantry:
        st.info("Add what you have to the Pantry in the sidebar first.")
    elif st.button("Compare with my pantry", width="stretch"):
        with st.spinner("Reading your pantry, then asking the graph..."):
            try:
                from cognitive_kitchen.rag.graph import pantry as PN

                resolution = PN.resolve_pantry(pantry)
                st.session_state["pantry_check"] = {
                    "resolution": resolution,
                    "row": PN.can_i_make(sources[picked]["key"],
                                         resolution.canonical)}
            except Exception as exc:
                st.session_state["pantry_check"] = {
                    "error": f"{type(exc).__name__}: {exc}"}

    check = st.session_state.get("pantry_check") or {}
    if check.get("error"):
        # Bolt is port 7687, which corporate VPNs routinely block. Say that
        # rather than showing a routing-table stack trace.
        st.warning("The graph is not reachable, so this cannot be answered right "
                   "now. Everything else on this page still works.")
        with st.expander("Details"):
            st.caption(check["error"])
            st.caption("Neo4j speaks Bolt on port 7687. A VPN blocking that "
                       "port produces exactly this, even when the instance is "
                       "healthy and port 443 on the same host is open.")
    elif check.get("row"):
        resolution, row = check["resolution"], check["row"]

        note = f"matched {len(resolution.canonical)} pantry items"
        if resolution.unknown:
            note += f" · not recognised: {', '.join(resolution.unknown)}"
        if resolution.llm_used:
            note += f" · a model resolved the leftovers (${resolution.cost_usd:.5f})"
        st.caption(note)

        if row["can_make"]:
            st.success(f"**{row['title']}** — you have everything, or something "
                       f"that stands in for it.")
        else:
            st.warning(f"**{row['title']}** — missing {len(row['missing'])} of "
                       f"{len(row['needed'])} ingredients")
        if row["buy"]:
            st.markdown("**Buy:** " + ", ".join(row["buy"]))
        for missing, covers in (row["swaps"] or {}).items():
            st.markdown(f"**Swap:** no {missing} — use {' or '.join(covers)}, "
                        f"already on your shelf")
        with st.expander("What the recipe needs"):
            st.caption(", ".join(row["needed"]) or "nothing recorded")

