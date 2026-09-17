"""In-process job registry with an async event queue per job.

Ingestion runs in a worker thread; progress frames are pushed onto an asyncio
queue that the SSE endpoint drains, so the UI sees pages as they are read.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from .models import IngestionRun, JobState, ProgressEvent

SENTINEL = object()


@dataclass
class Job:
    job_id: str
    kind: str
    origin: str
    state: JobState = JobState.pending
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    history: list[ProgressEvent] = field(default_factory=list)
    result: IngestionRun | None = None
    error: str | None = None
    loop: asyncio.AbstractEventLoop | None = None
    _seq: int = 0

    def emit(self, event: ProgressEvent) -> None:
        """Thread-safe publish from the worker thread."""
        self._seq += 1
        event.seq = self._seq
        self.history.append(event)
        self.state = event.state
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, event)

    def close(self) -> None:
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, SENTINEL)


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self, kind: str, origin: str) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id=job_id, kind=kind, origin=origin)
        try:
            job.loop = asyncio.get_running_loop()
        except RuntimeError:
            job.loop = None
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        return [
            {"job_id": j.job_id, "kind": j.kind, "origin": j.origin,
             "state": j.state, "frames": len(j.history),
             "recipes": (j.result.n_recipes if j.result else 0)}
            for j in self._jobs.values()
        ]


registry = JobRegistry()
