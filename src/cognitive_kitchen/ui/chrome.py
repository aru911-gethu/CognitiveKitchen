"""Shared page chrome, so every page looks like the same product.

theme.css was loaded by the console and by nothing else, which meant the two
pages people actually work in -- the Lab and the Kitchen -- rendered as unstyled
Streamlit while the landing page looked designed. Streamlit runs each page as its
own script, so stylesheets do not carry across; every page has to ask.

set_page_config must be the first Streamlit call on a page, so this wraps both
steps in one call that pages make before anything else.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent


def page(title: str, icon: str = "C", layout: str = "wide",
         sidebar: str = "expanded") -> None:
    """Set the page config and load the shared stylesheet. Call this first."""
    st.set_page_config(page_title=title, page_icon=icon, layout=layout,
                       initial_sidebar_state=sidebar)
    st.markdown((HERE / "theme.css").read_text(encoding="utf-8"),
                unsafe_allow_html=True)


def header(title: str, subtitle: str = "", pills: list[str] | None = None) -> None:
    """A compact version of the console hero, for interior pages."""
    parts = [f'<div class="hero hero-slim"><h1>{title}</h1>']
    if subtitle:
        parts.append(f"<p>{subtitle}</p>")
    if pills:
        parts.append('<div style="margin-top:12px">')
        parts.extend(f'<span class="pill">{p}</span>' for p in pills)
        parts.append("</div>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def stepper(steps: list[tuple[str, str]]) -> None:
    """Show where the pipeline stands: done, current, or not reached.

    The Lab showed this as a run of bold text, which gave no sense of progress
    through a chain. Each step carries its locked value so the whole decision is
    readable at a glance.
    """
    out = ['<div class="stepper">']
    seen_pending = False
    for label, value in steps:
        if value:
            state, shown = "done", value
        elif not seen_pending:
            state, shown = "current", "choose"
            seen_pending = True
        else:
            state, shown = "todo", "-"
        out.append(f'<span class="stepitem {state}">'
                   f'<b>{label}</b><i>{shown}</i></span>')
    out.append("</div>")
    st.markdown("".join(out), unsafe_allow_html=True)