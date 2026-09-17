"""Interfaces a plugin must satisfy to appear in the UI dropdowns."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from .types import Answer, Corpus, Passage, Scored


@runtime_checkable
class Chunker(Protocol):
    name: str
    params: dict[str, Any]

    def split(self, corpus: Corpus) -> list[Passage]: ...


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) L2-normalised float32 array."""
        ...


@runtime_checkable
class Retriever(Protocol):
    name: str

    def index(self, passages: list[Passage]) -> None: ...
    def search(self, query: str, k: int) -> list[Scored]: ...


@runtime_checkable
class QueryTransform(Protocol):
    name: str

    def expand(self, question: str) -> list[str]: ...


@runtime_checkable
class Generator(Protocol):
    name: str

    def answer(self, question: str, contexts: list[Passage]) -> Answer: ...