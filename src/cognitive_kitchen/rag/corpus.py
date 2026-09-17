"""Render Documents into one corpus and record where each recipe sits.

The offsets recorded here are what make chunk quality measurable: a chunk knows
its own span, every recipe knows its span, so purity and coverage are
arithmetic. Nothing external is consulted.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from ..config import settings
from .types import Corpus, RecipeSpan

SEPARATOR = "\n\n"


def build_corpus(docs: list[Document], source: str = "") -> Corpus:
    chunks: list[str] = []
    spans: list[RecipeSpan] = []
    cursor = 0
    for doc in docs:
        text = doc.page_content
        if not text.strip():
            continue
        spans.append(RecipeSpan(
            recipe_id=doc.metadata.get("recipe_id", f"d_{len(spans):04d}"),
            title=doc.metadata.get("title", "")[:180],
            start=cursor,
            end=cursor + len(text),
            pages=list(doc.metadata.get("pages") or []),
        ))
        chunks.append(text)
        cursor += len(text) + len(SEPARATOR)
    return Corpus(text=SEPARATOR.join(chunks), spans=tuple(spans), source=source)


def cache_path(corpus: Corpus) -> Path:
    return settings.data_dir / "eval" / f"corpus-{corpus.key}.json"


def save_corpus(corpus: Corpus) -> Path:
    path = cache_path(corpus)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "sha256": corpus.sha256,
        "source": corpus.source,
        "characters": len(corpus.text),
        "n_recipes": len(corpus.spans),
        "separator": SEPARATOR,
        "spans": [{"recipe_id": s.recipe_id, "title": s.title, "start": s.start,
                   "end": s.end, "pages": s.pages} for s in corpus.spans],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def summarise(corpus: Corpus) -> dict:
    lengths = [s.length for s in corpus.spans] or [0]
    return {
        "sha256": corpus.sha256[:16],
        "characters": len(corpus.text),
        "recipes": len(corpus.spans),
        "mean_recipe_chars": round(sum(lengths) / len(lengths), 1),
        "min_recipe_chars": min(lengths),
        "max_recipe_chars": max(lengths),
        "source": corpus.source,
    }