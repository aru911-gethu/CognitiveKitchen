"""Cognitive Kitchen - ingestion console."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

from cognitive_kitchen.config import settings
from cognitive_kitchen.ui import warmup

API = settings.api_base
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

st.set_page_config(page_title="Cognitive Kitchen", page_icon="C",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown((HERE / "theme.css").read_text(encoding="utf-8"), unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><h1>Cognitive Kitchen</h1>'
    '<p>A workbench for building a question-answering system over documents you '
    'own &mdash; and for proving its answers can be trusted.</p>'
    '<div style="margin-top:14px">'
    '<span class="pill">1 &middot; Ingest</span>'
    '<span class="pill">2 &middot; Experiment</span>'
    '<span class="pill">3 &middot; Lock what wins</span>'
    '<span class="pill">4 &middot; Chat</span>'
    '</div></div>', unsafe_allow_html=True)


def api_up() -> bool:
    try:
        return httpx.get(f"{API}/health", timeout=2.5).status_code == 200
    except Exception:
        return False


def metric_row(cols, values):
    for col, (label, val) in zip(cols, values):
        col.markdown(f'<div class="card"><div class="metric-label">{label}</div>'
                     f'<div class="metric">{val}</div></div>', unsafe_allow_html=True)


def consume(job_id: str, unit_word: str):
    """Show a live dish count and nothing else.

    The per-URL log was noise: it named pages the user did not ask about and
    buried the one number that matters.
    """
    headline = st.empty()
    bar = st.progress(0.0)
    found = 0
    final = None

    def paint(running: bool) -> None:
        word = "dish" if found == 1 else "dishes"
        if running:
            headline.markdown(
                f'<div class="runline"><span class="livedot"></span>'
                f'Running &mdash; <b>{found} {word}</b> found</div>',
                unsafe_allow_html=True)
        else:
            headline.markdown(
                f'<div class="runline done">Finished &mdash; '
                f'<b>{found} {word}</b> saved</div>', unsafe_allow_html=True)

    paint(True)

    with httpx.stream("GET", f"{API}/ingest/{job_id}/stream", timeout=None) as resp:
        for raw in resp.iter_lines():
            if not raw or not raw.startswith("data:"):
                continue
            body = raw[5:].strip()
            if body in ("", "{}"):
                continue
            try:
                f = json.loads(body)
            except json.JSONDecodeError:
                continue

            total, idx = f.get("unit_total") or 0, f.get("unit_index") or 0
            if total:
                bar.progress(min(idx / total, 1.0))
            found = max(found, f.get("recipes_found", 0))

            if f.get("kind") == "done":
                final = f.get("payload", {})
                bar.progress(1.0)
                paint(False)
            elif f.get("kind") == "error":
                headline.error(f.get("message", "Failed"))
            else:
                paint(True)
    return final

with st.sidebar:
    st.subheader("Service")
    if api_up():
        st.markdown('<span class="livedot"></span>API online', unsafe_allow_html=True)
        st.caption(API)
    else:
        st.error("API offline. Start the backend first.")
    st.divider()
    st.subheader("Benchmark")
    gp = ROOT / "data" / "golden_dataset.json"
    if gp.exists():
        g = json.loads(gp.read_text(encoding="utf-8"))
        st.metric("Golden recipes", len(g["recipes"]))
        st.metric("Golden queries", len(g["evaluation"]["queries"]))
        st.caption("Ground truth for all evaluation.")

# ---------------------------------------------------------------- the lede
# Earlier versions opened on a file uploader, then on a pasted metrics table.
# Neither told a newcomer what the tool is for. This explains the job first,
# offers the loaded sample as a way in, and keeps the measurement story in an
# expander for whoever wants it. The nuts example is shown as a before/after
# rather than described, because that contrast is the whole point.


# Fire the slow loads onto a background thread before anything renders. The
# first visit to the Lab or the Kitchen otherwise pays for the embedding model
# and the first Neo4j round trip at once, which reads as the app being slow when
# it is only cold.
warmup.start(st.session_state.get("warm_generator", False))


@st.cache_data(ttl=300, show_spinner=False)
def local_stats() -> dict:
    """Counts from disk only. No network, so it is safe to block on."""
    runs = sorted((ROOT / "data" / "ingested").glob("*.json"))
    recipes = 0
    for f in runs:
        try:
            recipes += len(json.loads(f.read_text(encoding="utf-8")).get("recipes") or [])
        except Exception:
            pass
    ingredients = 0
    try:
        from cognitive_kitchen.rag.vocab import ingredients as _v

        ingredients = len(_v.all_canonical())
    except Exception:
        pass
    return {"runs": len(runs), "recipes": recipes, "ingredients": ingredients}


@st.cache_data(ttl=300, show_spinner=False)
def graph_nodes() -> int:
    """One network round trip, so it is called last and never gates the page."""
    try:
        from cognitive_kitchen.rag.graph import client as _c

        health = _c.health()
        return int(health.get("nodes") or 0) if health.get("ok") else 0
    except Exception:
        return 0


S = local_stats()

# The pills hold their place in the layout and are written at the very end of the
# script, once the graph has answered. The page paints without waiting for it.
pills_slot = st.empty()

st.markdown('<div class="sectionhead">Build a chatbot over your own documents '
            "&mdash; and prove it works</div>"
            '<p class="sectionsub">Most tools let you build one. This one shows '
            "you, with numbers, which way of building it is actually best. The "
            "example here is cookbooks, so the questions are about cooking. None "
            "of the machinery is.</p>", unsafe_allow_html=True)

st.markdown('<div class="sectionhead">How it works</div>'
            '<p class="sectionsub">Four steps. Each one is measured before you '
            "move on.</p>", unsafe_allow_html=True)

STEPS = [
    ("Ingest", "Drop in a cookbook PDF or a few recipe links. It reads them "
               "page by page and pulls out every recipe, its ingredients and "
               "its steps."),
    ("Experiment", "In the Lab, try different ways of slicing the text, "
                   "searching it and writing the answer. Every option is "
                   "scored side by side on the same questions."),
    ("Lock what wins", "Pick the winner at each step. Your choice is saved, "
                       "and the next step is measured using it rather than "
                       "some default."),
    ("Chat", "The Kitchen runs the pipeline you built. Ask it anything, and "
             "check with one click whether you can actually cook the answer."),
]
for col, (index, (title, body)) in zip(st.columns(4), enumerate(STEPS, start=1)):
    col.markdown(f'<div class="step"><div class="step-num">{index}</div>'
                 f"<h4>{title}</h4><p>{body}</p></div>", unsafe_allow_html=True)

st.markdown('<div class="sectionhead">Try it right now</div>'
            '<p class="sectionsub">Two ways in. Neither needs any setup.</p>',
            unsafe_allow_html=True)

if S["recipes"]:
    # Chips are inline-block, so the HTML needs real whitespace between them or
    # they render as one run-on line.
    chips = "\n".join(f'<span class="chip">{q}</span>' for q in (
        "what can I make with rice?",
        "something with no dairy",
        "what can I use instead of ghee?",
        "I am avoiding nuts, what can I make?",
        "what can I cook tonight?"))
    st.markdown(
        f'<div class="trybox"><h4>A cookbook is already loaded &mdash; '
        f'{S["recipes"]} recipes</h4>'
        f'<p class="sub">Nothing to install, nothing to ingest. Try one of '
        f"these:</p>{chips}</div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown(
            '<div class="cta"><div class="tag">Just want to use it</div>'
            "<h4>Ask the Kitchen</h4>"
            "<p>A working chatbot over the loaded cookbook. Ask what to cook, "
            "what to leave out, what to use instead &mdash; then click once to "
            "check whether your kitchen actually has the ingredients.</p></div>",
            unsafe_allow_html=True)
        if st.button("Open the Kitchen", key="cta-kitchen", width="stretch"):
            st.switch_page("pages/3_Kitchen.py")
    with right:
        st.markdown(
            '<div class="cta"><div class="tag">Want to see the working</div>'
            "<h4>Open the Lab</h4>"
            "<p>Every choice behind that chatbot, scored side by side: how to "
            "split the text, how to search it, how to write the answer. Change "
            "one, watch the numbers move, lock what wins.</p></div>",
            unsafe_allow_html=True)
        if st.button("Open the Lab", key="cta-lab", width="stretch"):
            st.switch_page("pages/2_RAG_Lab.py")
else:
    st.info("Nothing ingested yet. Add a cookbook PDF or some recipe URLs "
            "below, then head to the Lab.")

with st.expander("Why this needed measuring in the first place"):
    st.markdown("Someone with a nut allergy asks a perfectly ordinary question. "
                "The **highest-scoring** search in the whole lab answers it like "
                "this:")
    st.markdown(
        '<div class="bad"><strong>Best search quality &middot; allergy ignored'
        "</strong>Nut Milk &nbsp;&middot;&nbsp; Cashew Nut Chutney "
        "&nbsp;&middot;&nbsp; Almond Honey Milk</div>",
        unsafe_allow_html=True)
    st.markdown(
        '<div class="good"><strong>Same question &middot; knowledge graph route'
        "</strong>Lemon Rice &nbsp;&middot;&nbsp; Tomato Rice "
        "&nbsp;&middot;&nbsp; Masoor Dhal &mdash; and 23&times; faster</div>",
        unsafe_allow_html=True)
    st.markdown("Nothing was broken. Searching by similarity has no way to "
                "express *without*, so *avoiding nuts* lands closest to the "
                "recipes that are mostly nuts. **The better the search got, the "
                "more confidently wrong it became** &mdash; and none of the "
                "standard quality scores could see it. A fifth measure had to be "
                "written:")
    st.dataframe(
        pd.DataFrame([
            {"how the search was built": "meaning-based search", "quality": 0.500, "kept to the restriction": 0.338, "seconds": 0.3},
            {"how the search was built": "keyword search", "quality": 0.350, "kept to the restriction": 0.588, "seconds": 0.1},
            {"how the search was built": "both, blended", "quality": 0.600, "kept to the restriction": 0.383, "seconds": 0.1},
            {"how the search was built": "blended + re-ranked", "quality": 0.700, "kept to the restriction": 0.400, "seconds": 25.8},
            {"how the search was built": "blended + knowledge graph", "quality": 0.700, "kept to the restriction": 1.000, "seconds": 1.1},
            {"how the search was built": "graph + re-ranked", "quality": 0.800, "kept to the restriction": 1.000, "seconds": 62.5},
        ]),
        width="stretch", hide_index=True,
        column_config={
            "quality": st.column_config.ProgressColumn(
                "search quality", min_value=0.0, max_value=1.0, format="%.2f",
                help="Did a correct recipe make the top five?"),
            "kept to the restriction": st.column_config.ProgressColumn(
                "kept to the restriction", min_value=0.0, max_value=1.0,
                format="percent",
                help="How often the recipes it found actually avoided what the "
                     "question ruled out."),
            "seconds": st.column_config.NumberColumn("secs", format="%.1f")})
    st.caption("Quality climbs as you read down. Staying within the restriction "
               "does not follow it. The graph works out which recipes are even "
               "allowed before anything is ranked, which is why it gets there "
               "without the slow re-ranking step.")

st.divider()
st.markdown('<div class="sectionhead">Add your own recipes</div>'
            '<p class="sectionsub">A PDF cookbook, or recipe pages from the web. '
            "Everything above is measured on whatever you ingest.</p>",
            unsafe_allow_html=True)

tab_pdf, tab_url, tab_data = st.tabs(["  PDF  ", "  Web URLs  ", "  Ingested data  "])

with tab_pdf:
    st.markdown("#### Ingest a cookbook PDF")
    st.caption("Pages stream as they are read. Uploads are saved under data/uploads.")
    left, right = st.columns([3, 2])
    with left:
        up = st.file_uploader("Drop a PDF", type=["pdf"], label_visibility="collapsed")
        if up is not None and st.button("Ingest upload", key="go_up"):
            r = httpx.post(f"{API}/ingest/pdf",
                           files={"file": (up.name, up.getvalue(), "application/pdf")},
                           timeout=180)
            r.raise_for_status()
            info = r.json()
            st.info(f"Saved to {info['saved_pdf']} ({info['bytes']:,} bytes)")
            consume(info["job_id"], "Pages")
    with right:
        # The source PDF may sit in data/ or, once uploaded, in data/uploads/
        sample = next((p for p in [
            ROOT / "data" / "indian-dishes-for-you-to-try-at-home.pdf",
            *sorted((ROOT / "data" / "uploads").glob("*.pdf")),
        ] if p.exists()), ROOT / "data" / "indian-dishes-for-you-to-try-at-home.pdf")
        st.markdown("**Sample already on disk**")
        st.caption(sample.name if sample.exists() else "sample missing")
        if sample.exists() and st.button("Ingest sample PDF", key="go_sample"):
            r = httpx.post(f"{API}/ingest/pdf-path", params={"path": str(sample)}, timeout=60)
            r.raise_for_status()
            consume(r.json()["job_id"], "Pages")

with tab_url:
    st.markdown("#### Ingest from the web")
    st.caption("One URL per line. A category or index page is expanded into its "
               "recipe links and traversed.")
    urls_raw = st.text_area("URLs", height=120, label_visibility="collapsed",
                            placeholder="https://www.example.com/recipes/curry")
    if st.button("Start crawl", key="go_url"):
        urls = [u.strip() for u in urls_raw.splitlines() if u.strip()]
        if not urls:
            st.warning("Add at least one URL.")
        else:
            # A recipe page yields a recipe and stops; an index page yields none
            # and is expanded. No toggle needed - the page itself decides.
            r = httpx.post(f"{API}/ingest/url",
                           json={"urls": urls, "crawl_category": True,
                                 "max_pages": 12}, timeout=60)
            r.raise_for_status()
            consume(r.json()["job_id"], "Pages")

with tab_data:
    st.markdown("#### Ingested datasets")
    st.button("Refresh", key="refresh")
    try:
        rows = httpx.get(f"{API}/datasets", timeout=10).json()
    except Exception as exc:
        rows = []
        st.error(f"Cannot reach API: {exc}")
    if rows:
        metric_row(st.columns(3), [
            ("Runs saved", len(rows)),
            ("Recipes total", sum(r["n_recipes"] or 0 for r in rows)),
            ("Units read", sum(r["n_units_read"] or 0 for r in rows)),
        ])
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        pick = st.selectbox("Inspect a run", [r["file"] for r in rows])
        if pick:
            d = json.loads((settings.ingested_dir / pick).read_text(encoding="utf-8"))
            st.json({k: v for k, v in d.items() if k != "recipes"}, expanded=False)
            if d.get("recipes"):
                st.caption(f"{len(d['recipes'])} recipes")
                st.dataframe(pd.DataFrame([
                    {"id": x["recipe_id"], "title": x["title"],
                     "ingredients": len(x["ingredient_lines"]),
                     "steps": len(x["step_lines"]),
                     "pages": ",".join(map(str, x["pages"])) or None,
                     "detected_by": x["detected_by"]} for x in d["recipes"]]),
                    use_container_width=True, hide_index=True, height=340)
    else:
        st.info("Nothing ingested yet. Start with the sample PDF.")


# --------------------------------------------------------------- warm-up panel
with st.sidebar:
    st.divider()
    st.subheader("Warm-up")
    st.caption(warmup.summary())
    if st.toggle("Also preload the answering model",
                 value=st.session_state.get("warm_generator", False),
                 key="warm_generator",
                 help="Loads Qwen2.5-1.5B up front, about 6 GB resident. It is "
                      "otherwise loaded on your first question rather than on "
                      "page open, so this helps the first answer and not the "
                      "first click."):
        warmup.start(True)

# ------------------------------------------------------------ fill the pills
# Last statement in the script: everything above is already on screen.
_pills = []
if S["recipes"]:
    _pills.append(f'<span class="statpill">{S["recipes"]} <span>recipes</span></span>')
if S["ingredients"]:
    _pills.append(f'<span class="statpill">{S["ingredients"]} <span>ingredients</span></span>')
_nodes = graph_nodes()
if _nodes:
    _pills.append(f'<span class="statpill">{_nodes:,} <span>graph nodes</span></span>')
_pills.append('<span class="statpill">9 <span>retrievers scored</span></span>')
pills_slot.markdown('<div style="margin:-4px 0 18px">' + "\n".join(_pills) + "</div>",
                    unsafe_allow_html=True)

