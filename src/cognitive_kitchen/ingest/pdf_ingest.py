"""PDF ingestion: one progress frame per page, emitted while reading."""
from __future__ import annotations

import re
import time
from pathlib import Path

from pypdf import PdfReader

from ..jobs import Job
from ..models import IngestionRun, JobState, ProgressEvent, RawRecipe, SourceType, utcnow

ING_HDR = re.compile(r"^\s*ingredients?\b", re.I)
METHOD_HDR = re.compile(
    r"^\s*((.*\s)?recipe\s+directions?|method|preparation|procedure|directions)\b\s*:?\s*$",
    re.I)
META = re.compile(r"^\s*(serves?\b|servings?\b|prep\s*time|cook\s*time|total\s*time|makes\b)", re.I)
VERB = re.compile(
    r"\b(heat|fry|add|stir|cook|serve|boil|mix|remove|combine|knead|roll|grind|wash|"
    r"soak|drain|simmer|blend|pour|cover|melt|bake|garnish|sprinkle|peel|chop|slice|"
    r"cut|put|keep|divide|shape|dip|toss|season|thicken|bring|sift|rub)\b", re.I)


def _lines(text: str) -> list[str]:
    out = [re.sub(r"[ ]{2,}", " ", l.strip())
           for l in text.replace("\t", " ").splitlines()]
    return [l for l in out if l]


def _looks_like_recipe(lines: list[str]) -> bool:
    if len(lines) < 3:
        return False
    joined = " ".join(lines)
    return bool(ING_HDR.search(joined) or METHOD_HDR.search(joined)
                or (VERB.search(joined) and len(joined) > 120))


def _parse_page(page_no: int, lines: list[str], origin: str, idx: int) -> RawRecipe | None:
    if not _looks_like_recipe(lines):
        return None
    title = lines[0][:180]
    ing_hdr = next((i for i, l in enumerate(lines) if ING_HDR.match(l)), None)
    met_hdr = next((i for i, l in enumerate(lines) if METHOD_HDR.match(l)), None)
    start = (ing_hdr + 1) if ing_hdr is not None else 1
    if met_hdr is not None and met_hdr > start:
        ing_end, step_start = met_hdr, met_hdr + 1
    else:
        prose = next((i for i, l in enumerate(lines)
                      if i >= start and len(l) > 55 and VERB.search(l)), None)
        ing_end = step_start = prose if prose is not None else len(lines)

    ing = [l for l in lines[start:ing_end] if not META.match(l)]
    steps = lines[step_start:]
    meta = [l for l in lines if META.match(l)]
    servings = next((m for m in meta if re.match(r"^\s*(serves?|servings?|makes)", m, re.I)), None)
    thint = next((m for m in meta if re.search(r"time", m, re.I)), None)

    return RawRecipe(
        recipe_id=f"r_{idx:04d}",
        title=title,
        source_type=SourceType.pdf,
        origin=origin,
        pages=[page_no],
        ingredient_lines=ing,
        step_lines=steps,
        servings_hint=servings,
        time_hint=thint,
        detected_by="headings" if ing_hdr is not None else "prose-boundary",
        raw_text="\n".join(lines),
    )


def ingest_pdf(job: Job, pdf_path: Path) -> IngestionRun:
    """Read a PDF page by page, emitting a frame for every page."""
    t0 = time.perf_counter()
    origin = pdf_path.name
    run = IngestionRun(run_id=job.job_id, source_type=SourceType.pdf, origin=origin)

    job.emit(ProgressEvent(job_id=job.job_id, kind="started", state=JobState.running,
                           message=f"Opening {origin}", unit_label=origin))

    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    run.n_units_total = total
    job.emit(ProgressEvent(job_id=job.job_id, kind="started", state=JobState.running,
                           message=f"{total} pages detected", unit_total=total,
                           unit_label=origin))

    chars = 0
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:                      # a damaged page must not stop the run
            text = ""
            run.warnings.append(f"page {i}: {type(exc).__name__}")

        lines = _lines(text)
        if not lines:
            run.n_units_empty += 1
            run.empty_units.append(i)
        else:
            run.n_units_read += 1
            chars += len(text)
            rec = _parse_page(i, lines, origin, len(run.recipes) + 1)
            if rec is not None:
                run.recipes.append(rec)
                job.emit(ProgressEvent(
                    job_id=job.job_id, kind="recipe", state=JobState.running,
                    message=f"Recipe: {rec.title[:60]}", unit_index=i, unit_total=total,
                    unit_label=f"page {i}", units_read=run.n_units_read,
                    recipes_found=len(run.recipes), chars_read=chars,
                    elapsed_seconds=round(time.perf_counter() - t0, 3),
                    payload={"title": rec.title, "page": i}))

        job.emit(ProgressEvent(
            job_id=job.job_id, kind="unit", state=JobState.running,
            message=f"Read page {i} of {total}", unit_index=i, unit_total=total,
            unit_label=f"page {i}", units_read=run.n_units_read,
            recipes_found=len(run.recipes), chars_read=chars,
            elapsed_seconds=round(time.perf_counter() - t0, 3),
            payload={"empty": not lines, "chars": len(text)}))

    run.n_recipes = len(run.recipes)
    run.elapsed_seconds = round(time.perf_counter() - t0, 3)
    run.finished_at = utcnow()
    return run
