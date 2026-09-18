"""Load the heavy things while the reader is still on the landing page.

Nothing here is required for correctness. The first visit to the Lab or the
Kitchen otherwise pays for the embedding model, the ingredient map and the first
Neo4j round trip all at once, which reads as the app being slow when it is
really the app being cold.

Everything runs on a daemon thread, so a failure or a slow network delays
nothing and the page never waits. Results land in Streamlit's own resource cache
where the pages will find them, because that cache is process-global rather than
per-script-run.

The answering model is deliberately NOT warmed by default. It is the one item
here that costs real memory -- Qwen2.5-1.5B in float32 is roughly 6 GB resident
-- and it is loaded lazily on the first answer rather than on page open, so
warming it improves the first question and not the first click. On a 16 GB
machine also running the Lab that trade is not obviously worth it, so it is a
choice rather than a default.
"""
from __future__ import annotations

import threading
import time

STATUS: dict[str, str] = {}
_lock = threading.Lock()
_started = False


def _mark(name: str, state: str) -> None:
    with _lock:
        STATUS[name] = state


def _run(warm_generator: bool) -> None:
    started = time.perf_counter()

    _mark("graph", "loading")
    try:
        from ..rag.graph import client

        health = client.health()
        _mark("graph", "ready" if health.get("ok") else "offline")
    except Exception:
        _mark("graph", "offline")

    _mark("vocabulary", "loading")
    try:
        from ..rag.vocab import ingredients as vocab

        _mark("vocabulary", f"{len(vocab.all_canonical())} ingredients")
    except Exception:
        _mark("vocabulary", "not built")

    _mark("embeddings", "loading")
    try:
        from ..rag.embedding.sentence_transformer import TextEmbedder

        TextEmbedder().encode(["warm up"])
        _mark("embeddings", "ready")
    except Exception as exc:
        _mark("embeddings", f"failed: {type(exc).__name__}")

    if warm_generator:
        _mark("answering model", "loading")
        try:
            from ..rag.generate.qwen_chat import QwenChat

            QwenChat()._ensure()
            _mark("answering model", "ready")
        except Exception as exc:
            _mark("answering model", f"failed: {type(exc).__name__}")

    _mark("_elapsed", f"{time.perf_counter() - started:.1f}s")


def start(warm_generator: bool = False) -> None:
    """Kick off the background warm-up once per process. Returns immediately."""
    global _started
    with _lock:
        if _started:
            return
        _started = True
    threading.Thread(target=_run, args=(warm_generator,),
                     name="ck-warmup", daemon=True).start()


def summary() -> str:
    """One line for the sidebar, safe to call at any point."""
    with _lock:
        items = {k: v for k, v in STATUS.items() if not k.startswith("_")}
    if not items:
        return "starting..."
    return "  ·  ".join(f"{k} {v}" for k, v in items.items())