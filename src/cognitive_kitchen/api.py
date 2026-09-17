"""FastAPI ingestion service.

Ingestion runs in a worker thread so the event loop stays free to stream
progress frames over Server-Sent Events while pages are still being read.

SECURITY: this service has no authentication. It accepts file uploads and
fetches user-supplied URLs, so it is bound to localhost by default. Do not
expose it to a network without adding auth and hardening the crawler.
"""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .config import settings
from .ingest.pdf_ingest import ingest_pdf
from .ingest.web_ingest import UnsafeUrl, ingest_urls
from .jobs import SENTINEL, Job, registry
from .models import IngestionRun, JobState, ProgressEvent

app = FastAPI(title="Cognitive Kitchen - Ingestion", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"], allow_headers=["*"])

MAX_UPLOAD_BYTES = 80 * 1024 * 1024


class UrlRequest(BaseModel):
    urls: list[str] = Field(min_length=1)
    crawl_category: bool = True
    max_pages: int | None = None


def _persist(run: IngestionRun) -> Path:
    out = settings.ingested_dir / f"{run.source_type.value}-{run.run_id}.json"
    out.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return out


def _finish(job: Job, run: IngestionRun | None, error: str | None) -> None:
    if error is not None:
        job.error = error
        job.emit(ProgressEvent(job_id=job.job_id, kind="error", state=JobState.failed,
                               message=error))
    else:
        assert run is not None
        path = _persist(run)
        job.result = run
        job.emit(ProgressEvent(
            job_id=job.job_id, kind="done", state=JobState.done,
            message=f"Saved {run.n_recipes} recipes to {path.name}",
            unit_index=run.n_units_read, unit_total=run.n_units_total,
            units_read=run.n_units_read, recipes_found=run.n_recipes,
            elapsed_seconds=run.elapsed_seconds,
            payload={"saved_to": str(path), "run": json.loads(run.model_dump_json())}))
    job.close()


def _run_pdf(job: Job, path: Path) -> None:
    try:
        _finish(job, ingest_pdf(job, path), None)
    except Exception as exc:
        _finish(job, None, f"{type(exc).__name__}: {exc}")


def _run_urls(job: Job, req: UrlRequest) -> None:
    try:
        run = ingest_urls(job, req.urls, req.crawl_category, req.max_pages)
        _finish(job, run, None)
    except UnsafeUrl as exc:
        _finish(job, None, f"Unsafe URL: {exc}")
    except Exception as exc:
        _finish(job, None, f"{type(exc).__name__}: {exc}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "env": settings.app_env,
            "ingested_dir": str(settings.ingested_dir)}


@app.post("/ingest/pdf")
async def ingest_pdf_endpoint(background: BackgroundTasks,
                             file: UploadFile = File(...)) -> JSONResponse:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only .pdf files are accepted")
    dest = settings.upload_dir / Path(file.filename).name
    size = 0
    with dest.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "file exceeds 80 MB limit")
            fh.write(chunk)

    job = registry.create("pdf", dest.name)
    background.add_task(asyncio.to_thread, _run_pdf, job, dest)
    return JSONResponse({"job_id": job.job_id, "saved_pdf": str(dest),
                         "bytes": size}, status_code=202)


@app.post("/ingest/pdf-path")
async def ingest_pdf_path(background: BackgroundTasks, path: str) -> JSONResponse:
    """Ingest a PDF already on disk, e.g. the sample in data/."""
    src = Path(path)
    if not src.is_file() or src.suffix.lower() != ".pdf":
        raise HTTPException(400, f"not a readable pdf: {path}")
    job = registry.create("pdf", src.name)
    background.add_task(asyncio.to_thread, _run_pdf, job, src)
    return JSONResponse({"job_id": job.job_id, "saved_pdf": str(src)}, status_code=202)


@app.post("/ingest/url")
async def ingest_url_endpoint(req: UrlRequest, background: BackgroundTasks) -> JSONResponse:
    job = registry.create("url", req.urls[0])
    background.add_task(asyncio.to_thread, _run_urls, job, req)
    return JSONResponse({"job_id": job.job_id, "seeds": req.urls}, status_code=202)


@app.get("/jobs")
def list_jobs() -> list[dict]:
    return registry.list()


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return {"job_id": job.job_id, "state": job.state, "kind": job.kind,
            "origin": job.origin, "error": job.error,
            "frames": [json.loads(f.model_dump_json()) for f in job.history[-200:]],
            "result": json.loads(job.result.model_dump_json()) if job.result else None}


@app.get("/ingest/{job_id}/stream")
async def stream(job_id: str):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")

    async def gen():
        replayed = 0
        for frame in list(job.history):          # catch a late subscriber up
            replayed = max(replayed, frame.seq)
            yield {"event": "progress", "data": frame.model_dump_json()}
        while True:
            item = await job.queue.get()
            if item is SENTINEL:
                yield {"event": "end", "data": "{}"}
                return
            if item.seq <= replayed:             # already sent during replay
                continue
            yield {"event": "progress", "data": item.model_dump_json()}

    return EventSourceResponse(gen())


@app.get("/datasets")
def datasets() -> list[dict]:
    out = []
    for p in sorted(settings.ingested_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({"file": p.name, "run_id": d.get("run_id"),
                    "source_type": d.get("source_type"), "origin": d.get("origin"),
                    "n_recipes": d.get("n_recipes"),
                    "n_units_read": d.get("n_units_read"),
                    "n_units_total": d.get("n_units_total"),
                    "started_at": d.get("started_at"),
                    "elapsed_seconds": d.get("elapsed_seconds"),
                    "size_kb": round(p.stat().st_size / 1024, 1)})
    return out