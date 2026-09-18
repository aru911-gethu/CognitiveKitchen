"""Cognitive Kitchen - ingestion console."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

from cognitive_kitchen.config import settings

API = settings.api_base
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

st.set_page_config(page_title="Cognitive Kitchen", page_icon="C",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown((HERE / "theme.css").read_text(encoding="utf-8"), unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><h1>Cognitive Kitchen</h1>'
    '<p>Decide what to cook, grounded in recipes you actually own.</p>'
    '<div style="margin-top:14px">'
    '<span class="pill">Phase 1 &middot; Ingestion</span>'
    '<span class="pill">PDF to structured</span>'
    '<span class="pill">Playwright crawl</span>'
    '<span class="pill">Live page stream</span>'
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
# This page used to open on a file uploader, which asks for work before saying
# what the work is for. The finding goes first now: a reader who leaves after
# one screen should still leave knowing what this measured.
st.markdown("#### The best-ranking retriever satisfied the user's stated "
            "constraint 40% of the time.")
st.markdown("Four standard retrieval metrics all called it good. "
            "A fifth one, which had to be written, is the only one that saw it.")

st.code("""configuration          hit@5   constraint respected   secs
dense                  0.500                  33.8%    0.3
bm25                   0.350                  58.8%    0.1
fusion:rrf             0.600                  38.3%    0.1
rrf + rerank           0.700                  40.0%   25.8
rrf + graph            0.700                 100.0%    1.1
rrf + graph + rerank   0.800                 100.0%   62.5""", language="text")

st.caption("Asked *\"I am avoiding nuts, what can I make?\"* the strongest "
           "ranker returned Nut Milk, Cashew Nut Chutney and Almond Honey Milk. "
           "Similarity search has no direction for *without* -- so the better "
           "the ranker, the more confidently wrong. The graph route fixes it by "
           "deciding membership before anything is ranked.")

hero_left, hero_right = st.columns(2)
with hero_left:
    st.page_link("pages/2_RAG_Lab.py",
                 label="**RAG Lab** - score every option, stage by stage",
                 icon=":material/science:")
with hero_right:
    st.page_link("pages/3_Kitchen.py",
                 label="**Kitchen** - chat on the pipeline you locked",
                 icon=":material/restaurant:")

st.divider()
st.markdown("##### Bring your own corpus")
st.caption("Ingest a cookbook PDF or a set of recipe URLs. Everything above is "
           "measured on whatever you ingest, not on a shipped demo set.")

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