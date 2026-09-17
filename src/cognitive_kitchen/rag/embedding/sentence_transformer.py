"""Sentence embeddings via sentence-transformers, model set by EMBEDDING_MODEL.

The model is loaded on first encode, not on import, so naive and recursive
chunking run without pulling weights. Vectors are L2-normalised, which makes
cosine similarity a plain dot product everywhere downstream.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ...config import settings
from ..registry import register


class TextEmbedder:
    def __init__(self, model_name: str | None = None, batch_size: int = 16,
                 cache: bool = True) -> None:
        self.model_name = model_name or settings.embedding_model
        self.name = self.model_name.split("/")[-1]
        self.batch_size = batch_size
        self.cache = cache
        self._model = None
        self._dim: int | None = None

    # -- lazy load ---------------------------------------------------------
    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
            self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._ensure()
        return int(self._dim or 0)

    # -- disk cache --------------------------------------------------------
    def _cache_file(self, texts: list[str]) -> Path:
        digest = hashlib.sha256(
            (self.model_name + "\x00" + "\x00".join(texts)).encode("utf-8")).hexdigest()
        return settings.data_dir / "eval" / f"emb-{digest[:16]}.npy"

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        path = self._cache_file(texts)
        if self.cache and path.exists():
            return np.load(path)

        model = self._ensure()
        vectors = model.encode(texts, batch_size=self.batch_size,
                               convert_to_numpy=True, normalize_embeddings=True,
                               show_progress_bar=False)
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vectors)
        return vectors


@register("embedder", "st")
def make(model_name: str | None = None, batch_size: int = 16,
         cache: bool = True) -> TextEmbedder:
    return TextEmbedder(model_name=model_name, batch_size=batch_size, cache=cache)