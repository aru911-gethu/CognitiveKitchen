"""The honest baseline: search exactly what was asked.

Any transform has to beat this to justify its latency.
"""
from __future__ import annotations

from ..registry import register


class Passthrough:
    name = "passthrough"

    def expand(self, question: str) -> list[str]:
        return [question]


@register("query", "passthrough")
def make() -> Passthrough:
    return Passthrough()