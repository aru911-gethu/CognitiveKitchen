"""Cognitive Kitchen - RAG over recipes you own."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ directory is prioritized over stale site-packages
_src_dir = str(Path(__file__).resolve().parents[1])
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

__version__ = "0.1.0"


def serve_api() -> None:
    """Run the ingestion API. Bound to localhost: it has no authentication."""
    import uvicorn

    from .config import settings

    uvicorn.run("cognitive_kitchen.api:app", host=settings.api_host,
                port=settings.api_port, reload=False)


def serve_ui() -> None:
    """Run the Streamlit console."""
    import sys
    from pathlib import Path

    from streamlit.web import cli

    app = Path(__file__).resolve().parent / "ui" / "app.py"
    sys.argv = ["streamlit", "run", str(app),
                "--server.address", "127.0.0.1", "--server.port", "8501"]
    cli.main()


def build_vocabulary() -> None:
    """ck-vocab: build data/eval/ingredient_map.json. Run before the graph."""
    from .rag.vocab.build import main

    main()


def build_graph() -> None:
    """ck-graph: rebuild the Neo4j knowledge graph from data/ingested/."""
    from .rag.graph.build import main

    main()
