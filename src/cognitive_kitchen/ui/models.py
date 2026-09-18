"""Model choice at runtime, so a stage can be re-measured on a bigger model.

Every component already accepts `model_name` and falls back to settings, so the
only thing missing was a way to say so from the UI. This writes the choice onto
settings and the caller clears Streamlit's resource cache, because an embedder
cached under the old model would otherwise survive the change and produce a
table labelled with one model and computed with another.

Verification is separate and explicit. A model id is only a string until
something tries to load it, and finding out mid-run -- forty seconds into a
nine-retriever sweep -- is the worst time to find out. `verify` pays that cost up
front on a single load and reports instead of raising.
"""
from __future__ import annotations

from ..config import settings

KNOWN: dict[str, list[str]] = {
    "embedding_model": [
        "Qwen/Qwen3-Embedding-0.6B",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
        "intfloat/e5-base-v2",
    ],
    "generation_model": [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "microsoft/Phi-3.5-mini-instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ],
    "reranker_model": [
        "BAAI/bge-reranker-base",
        "BAAI/bge-reranker-v2-m3",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ],
}

LABELS = {"embedding_model": "Embeddings", "generation_model": "Generation",
          "reranker_model": "Reranker"}

OTHER = "(other - type it)"


def current(role: str) -> str:
    return str(getattr(settings, role, "") or "")


def apply(role: str, model_id: str) -> bool:
    """Point a role at a different model. True when something actually changed."""
    model_id = (model_id or "").strip()
    if not model_id or model_id == current(role):
        return False
    setattr(settings, role, model_id)
    return True


def verify(role: str, model_id: str) -> tuple[bool, str]:
    """Try to load it now and say what happened. Never raises."""
    model_id = (model_id or "").strip()
    if not model_id:
        return False, "no model id given"
    try:
        if role == "embedding_model":
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_id, device="cpu")
            return True, f"loaded, {int(model.get_sentence_embedding_dimension())} dimensions"
        if role == "reranker_model":
            from sentence_transformers import CrossEncoder

            CrossEncoder(model_id, device="cpu")
            return True, "loaded"
        if role == "generation_model":
            from transformers import AutoTokenizer

            AutoTokenizer.from_pretrained(model_id)
            return True, "tokenizer loaded; weights load on the first answer"
        return False, f"unknown role {role}"
    except Exception as exc:
        first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        return False, first[:160]