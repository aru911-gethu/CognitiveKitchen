"""Load ingestion output (data/ingested/*.json) into LangChain Documents.

One Document per recipe. This is the normal path: the pipeline reads what the
Phase 1 ingester produced and knows nothing about any curated dataset.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from ...config import settings


def _render(recipe: dict) -> str:
    """Flatten a recipe into the text the retriever will see."""
    parts: list[str] = [recipe.get("title") or "(untitled)"]
    if recipe.get("servings_hint"):
        parts.append(str(recipe["servings_hint"]))
    if recipe.get("time_hint"):
        parts.append(str(recipe["time_hint"]))
    ings = recipe.get("ingredient_lines") or []
    if ings:
        parts.append("Ingredients:")
        parts.extend(f"- {line}" for line in ings)
    steps = recipe.get("step_lines") or []
    if steps:
        parts.append("Method:")
        parts.extend(f"{i}. {line}" for i, line in enumerate(steps, 1))
    return "\n".join(parts)


def load_ingested_json(path: str | Path) -> list[Document]:
    path = Path(path)
    run = json.loads(path.read_text(encoding="utf-8"))
    docs: list[Document] = []
    for recipe in run.get("recipes", []):
        docs.append(Document(
            page_content=_render(recipe),
            metadata={
                "recipe_id": recipe["recipe_id"],
                "title": recipe.get("title") or "",
                "source_type": recipe.get("source_type") or run.get("source_type"),
                "origin": recipe.get("origin") or run.get("origin"),
                "pages": recipe.get("pages") or [],
                "url": recipe.get("url"),
                "detected_by": recipe.get("detected_by") or "",
                "run_id": run.get("run_id"),
                "source_file": path.name,
            },
        ))
    return docs


def load_latest_ingested(source_type: str | None = None) -> list[Document]:
    """Newest run in data/ingested, optionally filtered to pdf or url."""
    pattern = f"{source_type}-*.json" if source_type else "*.json"
    files = sorted(settings.ingested_dir.glob(pattern),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(
            f"no ingestion output matching {pattern} in {settings.ingested_dir}")
    return load_ingested_json(files[0])


def list_runs() -> list[dict]:
    out = []
    for p in sorted(settings.ingested_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({"file": p.name, "source_type": d.get("source_type"),
                    "n_recipes": d.get("n_recipes"), "origin": d.get("origin")})
    return out