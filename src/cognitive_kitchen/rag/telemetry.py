"""Latency, token and cost accounting, plus optional LangSmith tracing.

Every stage records where its wall clock went and what it spent, so a strategy
that gains 0.02 for 65x the latency is visibly a bad trade rather than a
footnote. LangSmith is used when configured and silently skipped when not.
"""
from __future__ import annotations

import functools
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from ..config import settings

# Cost per million tokens, USD. Only the judge is paid; local models are free
# but their latency is the real budget.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def price_of(model: str, tokens_in: int, tokens_out: int) -> float:
    key = next((k for k in PRICES if k in model), None)
    if key is None:
        return 0.0
    rate_in, rate_out = PRICES[key]
    return round(tokens_in / 1e6 * rate_in + tokens_out / 1e6 * rate_out, 6)


@dataclass
class Span:
    name: str
    seconds: float = 0.0
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Ledger:
    """Collects spans for one run so the UI can show a breakdown."""

    spans: dict[str, Span] = field(default_factory=dict)
    started: float = field(default_factory=time.perf_counter)

    def span(self, name: str) -> Span:
        return self.spans.setdefault(name, Span(name))

    def record(self, name: str, seconds: float, *, tokens_in: int = 0,
               tokens_out: int = 0, model: str = "", **meta: Any) -> None:
        s = self.span(name)
        s.seconds += seconds
        s.calls += 1
        s.tokens_in += tokens_in
        s.tokens_out += tokens_out
        if model:
            s.usd += price_of(model, tokens_in, tokens_out)
            s.meta["model"] = model
        s.meta.update(meta)

    @contextmanager
    def timed(self, name: str, **meta: Any) -> Iterator[Span]:
        start = time.perf_counter()
        try:
            yield self.span(name)
        finally:
            self.record(name, time.perf_counter() - start, **meta)

    @property
    def total_seconds(self) -> float:
        return round(time.perf_counter() - self.started, 3)

    @property
    def total_usd(self) -> float:
        return round(sum(s.usd for s in self.spans.values()), 6)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_seconds": self.total_seconds,
            "total_usd": self.total_usd,
            "tokens_in": sum(s.tokens_in for s in self.spans.values()),
            "tokens_out": sum(s.tokens_out for s in self.spans.values()),
            "spans": [
                {"name": s.name, "seconds": round(s.seconds, 3), "calls": s.calls,
                 "tokens_in": s.tokens_in, "tokens_out": s.tokens_out,
                 "usd": round(s.usd, 6),
                 "per_call_s": round(s.seconds / s.calls, 3) if s.calls else 0.0,
                 **({"model": s.meta["model"]} if "model" in s.meta else {})}
                for s in sorted(self.spans.values(), key=lambda x: -x.seconds)
            ],
        }


# ------------------------------------------------------------- LangSmith
def langsmith_enabled() -> bool:
    return bool(settings.langsmith_tracing and settings.langsmith_api_key)


def configure_langsmith() -> bool:
    """Export the env vars the LangSmith SDK reads. Returns whether it is on."""
    if not langsmith_enabled():
        os.environ.pop("LANGSMITH_TRACING", None)
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key or ""
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    return True


def traced(name: str) -> Callable:
    """Wrap a function as a LangSmith run when tracing is configured.

    A no-op otherwise, so the pipeline has no hard dependency on the service.
    """
    def decorator(fn: Callable) -> Callable:
        if not configure_langsmith():
            return fn
        try:
            from langsmith import traceable
        except Exception:
            return fn

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return traceable(name=name)(fn)(*args, **kwargs)
        return wrapper
    return decorator