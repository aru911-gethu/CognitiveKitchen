"""Schemas for ingestion output.

Mirrors the shape of data/recipes.json so downstream stages read one format
regardless of whether the source was a PDF or the web.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SourceType(str, Enum):
    pdf = "pdf"
    url = "url"


class JobState(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class RawRecipe(BaseModel):
    recipe_id: str
    title: str
    source_type: SourceType
    origin: str
    pages: list[int] = Field(default_factory=list)
    url: str | None = None
    ingredient_lines: list[str] = Field(default_factory=list)
    step_lines: list[str] = Field(default_factory=list)
    servings_hint: str | None = None
    time_hint: str | None = None
    detected_by: str = "headings"
    raw_text: str = ""


class IngestionRun(BaseModel):
    run_id: str
    source_type: SourceType
    origin: str
    started_at: str = Field(default_factory=utcnow)
    finished_at: str | None = None
    elapsed_seconds: float = 0.0
    n_units_total: int = 0
    n_units_read: int = 0
    n_units_empty: int = 0
    empty_units: list[int] = Field(default_factory=list)
    n_recipes: int = 0
    recipes: list[RawRecipe] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ProgressEvent(BaseModel):
    """One frame of the live ingestion stream."""

    job_id: str
    seq: int = 0         # monotonic per job, lets a late subscriber de-duplicate
    kind: str            # started | unit | recipe | warning | done | error
    state: JobState
    message: str = ""
    unit_index: int = 0          # page number, or nth page crawled
    unit_total: int = 0
    unit_label: str = ""         # "page 12" or the URL
    units_read: int = 0
    recipes_found: int = 0
    chars_read: int = 0
    elapsed_seconds: float = 0.0
    payload: dict[str, Any] = Field(default_factory=dict)
    at: str = Field(default_factory=utcnow)
