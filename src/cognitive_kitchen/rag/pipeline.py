"""The pipeline the Lab builds and the Kitchen runs.

One file is the contract between the two pages: the Lab writes it after you lock
a choice at each stage, the chat page reads it and runs nothing else. Keeping it
on disk rather than in session state means the chat survives a restart and the
configuration can be inspected, diffed and committed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import settings


@dataclass
class Pipeline:
    # stage 2
    chunker: str = "recursive"
    chunker_params: dict[str, Any] = field(default_factory=dict)
    # stage 3
    source: str = "rrf"              # dense | bm25 | tfidf | hybrid | rrf
    sparse: str = "bm25"             # which sparse half a fuser uses
    use_mmr: bool = False
    use_rerank: bool = False
    use_graph: bool = False
    k: int = 5
    # stage 4
    query_transform: str = "passthrough"
    # stage 6
    strategy: str = "stuff_strict"
    generator: str = "qwen"
    max_new_tokens: int = 350   # 120 truncated recipes mid-step
    # provenance
    corpus_source: str = "pdf"
    scores: dict[str, Any] = field(default_factory=dict)

    def retriever_spec(self) -> tuple[str, dict[str, Any]]:
        """Compose the four Stage 3 controls into one registry call.

        Order matters and is fixed: the graph decides membership, then a source
        ranks, then diversity or reranking reshuffles. Reranking wraps the
        source, and the graph wraps whatever that produced, so the graph is
        outermost -- a gate applied after reranking is still a gate.
        """
        inner, params = self.source, {}
        if self.source in ("hybrid", "rrf"):
            params["sparse"] = self.sparse
        if self.use_mmr:
            inner, params = "mmr", {"base": inner}
        if self.use_rerank:
            inner, params = "cross_encoder", {"first_stage": inner}
        if self.use_graph:
            inner, params = "graph_hybrid", {"base": inner}
        return inner, params

    def label(self) -> str:
        bits = [self.chunker, self.source]
        if self.use_mmr:
            bits.append("mmr")
        if self.use_rerank:
            bits.append("rerank")
        if self.use_graph:
            bits.append("graph")
        bits += [self.query_transform, self.strategy]
        return " -> ".join(bits)


def path() -> Path:
    return settings.data_dir / "eval" / "pipeline.json"


def save(pipeline: Pipeline) -> Path:
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(pipeline), indent=2, ensure_ascii=False),
                      encoding="utf-8")
    return target


def load() -> Pipeline | None:
    target = path()
    if not target.exists():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    known = {f for f in Pipeline.__dataclass_fields__}
    return Pipeline(**{k: v for k, v in data.items() if k in known})


def build_runtime(pipeline: Pipeline, dataset_file: str | Path | None = None):
    """Turn a saved pipeline into live objects. Used by the chat page."""
    from .corpus import build_corpus
    from .loaders import load_ingested_json, load_latest_ingested
    from .registry import build, discover

    discover("cognitive_kitchen.rag.chunking", "cognitive_kitchen.rag.embedding",
             "cognitive_kitchen.rag.retrieval", "cognitive_kitchen.rag.query",
             "cognitive_kitchen.rag.generate", "cognitive_kitchen.rag.strategies")

    if dataset_file:
        file_path = settings.ingested_dir / dataset_file if isinstance(dataset_file, str) else dataset_file
        if file_path.exists():
            docs = load_ingested_json(file_path)
            source_name = file_path.name
        else:
            docs = load_latest_ingested(source_type=pipeline.corpus_source or None)
            source_name = pipeline.corpus_source
    else:
        docs = load_latest_ingested(source_type=pipeline.corpus_source or None)
        source_name = pipeline.corpus_source

    corpus = build_corpus(docs, source=source_name)
    embedder = build("embedder", "st")
    passages = build("chunker", pipeline.chunker,
                     **(pipeline.chunker_params or {})).split(corpus)

    name, params = pipeline.retriever_spec()
    if name in ("dense", "hybrid", "rrf", "mmr", "cross_encoder", "graph_hybrid"):
        params["embedder"] = embedder
    retriever = build("retriever", name, **params)
    retriever.index(passages)

    transform = (None if pipeline.query_transform in ("", "passthrough")
                 else build("query", pipeline.query_transform))
    generator = build("generator", pipeline.generator,
                      max_new_tokens=pipeline.max_new_tokens)
    strategy = build("strategy", pipeline.strategy, generator=generator)
    return {"corpus": corpus, "passages": passages, "retriever": retriever,
            "transform": transform, "generator": generator,
            "strategy": strategy, "embedder": embedder}