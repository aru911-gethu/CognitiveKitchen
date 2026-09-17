"""Live commentary for a stage run.

A progress bar says how far along a run is; it does not say what the run is
doing. On this machine a Stage 6 comparison with the local model is several
minutes of an unmoving bar, which is indistinguishable from a hang. So each
stage narrates: what it is working on, what it just finished, and what that
cost.

Wraps st.status so the whole transcript stays available after the run and
collapses out of the way once it is done.
"""
from __future__ import annotations

import time
from typing import Any

import streamlit as st

MAX_LINES = 400


class Commentary:
    def __init__(self, title: str, total: int = 0) -> None:
        self.total = max(total, 0)
        self.done = 0
        self.started = time.perf_counter()
        self.lines: list[str] = []
        self._status = st.status(title, expanded=True)
        self._bar = self._status.progress(0.0) if total else None
        self._body = self._status.empty()

    # -- narration ---------------------------------------------------------
    def say(self, line: str, indent: int = 0) -> None:
        prefix = "    " * indent
        self.lines.append(f"{prefix}{line}")
        del self.lines[:-MAX_LINES]
        self._body.code("\n".join(self.lines), language="text")

    def step(self, label: str, detail: str = "") -> None:
        """Announce the thing about to happen and move the bar."""
        self.done += 1
        self._status.update(label=f"{label}  ({self.done}/{self.total})"
                                 if self.total else label)
        if self._bar is not None and self.total:
            self._bar.progress(min(self.done / self.total, 1.0))
        self.say(f"[{self.elapsed:6.1f}s] {label}" + (f"  {detail}" if detail else ""))

    def result(self, line: str) -> None:
        self.say(line, indent=1)

    def warn(self, line: str) -> None:
        self.say(f"!! {line}", indent=1)

    # -- state -------------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def finish(self, summary: str = "") -> None:
        label = summary or "Done"
        self._status.update(label=f"{label}  ·  {self.elapsed:.1f}s",
                            state="complete", expanded=False)

    def fail(self, message: str) -> None:
        self.warn(message)
        self._status.update(label=f"Failed after {self.elapsed:.1f}s",
                            state="error", expanded=True)


def per_question(commentary: Commentary, every: int = 1):
    """Callback for the stage evaluators: reports each question as it lands."""
    def report(index: int, total: int, question: str, **facts: Any) -> None:
        if index % every and index != total:
            return
        bits = " ".join(f"{k}={v}" for k, v in facts.items() if v is not None)
        commentary.result(f"{index:>3}/{total} {question[:52]:54s} {bits}")
    return report