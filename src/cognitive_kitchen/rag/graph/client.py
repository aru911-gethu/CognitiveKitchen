"""Neo4j Aura connection.

Aura quirk worth remembering: the username and the database name are both the
instance id, something like "09bfbafe". They are not the literal "neo4j", and
using "neo4j" fails with an unhelpful authentication error.

The driver is created lazily and shared, so importing this module costs nothing
and the rest of the pipeline runs with no graph configured at all.
"""
from __future__ import annotations

import contextlib
from typing import Any, Iterator

from ...config import settings


class GraphUnavailable(RuntimeError):
    """Raised when the graph is asked for but not configured or not reachable."""


_driver = None


def driver():
    global _driver
    if _driver is None:
        if not settings.graph_configured:
            raise GraphUnavailable(
                "NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD must be set in .env")
        import traceback
        from neo4j import GraphDatabase

        uri = settings.neo4j_uri
        auth = (settings.neo4j_username, settings.neo4j_password)
        try:
            d = GraphDatabase.driver(uri, auth=auth)
            d.verify_connectivity()
            _driver = d
        except Exception:
            tb = traceback.format_exc()
            if "certificate" in tb.lower() or "ssl" in tb.lower():
                fallback_uri = uri.replace("neo4j+s://", "neo4j+ssc://").replace(
                    "bolt+s://", "bolt+ssc://")
                d = GraphDatabase.driver(fallback_uri, auth=auth)
                d.verify_connectivity()
                _driver = d
            else:
                raise
    return _driver


def database() -> str:
    return settings.neo4j_database or settings.neo4j_username or "neo4j"


def close() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


@contextlib.contextmanager
def session() -> Iterator[Any]:
    with driver().session(database=database()) as s:
        yield s


def run(cypher: str, **params: Any) -> list[dict[str, Any]]:
    """Run one statement and return the rows as plain dicts."""
    with session() as s:
        return [record.data() for record in s.run(cypher, **params)]


def write(cypher: str, **params: Any) -> dict[str, int]:
    """Run one write statement and return the counters that changed."""
    with session() as s:
        summary = s.run(cypher, **params).consume()
        c = summary.counters
        return {"nodes": c.nodes_created, "relationships": c.relationships_created,
                "properties": c.properties_set}


def health() -> dict[str, Any]:
    """Cheap probe for the UI: is the graph reachable, and how big is it?"""
    try:
        rows = run("""
            CALL () {MATCH (n) RETURN count(n) AS nodes}
            CALL () {MATCH ()-[r]->() RETURN count(r) AS rels}
            RETURN nodes, rels""")
        labels = run("""
            MATCH (n) UNWIND labels(n) AS label
            RETURN label, count(*) AS n ORDER BY n DESC""")
        return {"ok": True, "database": database(),
                "nodes": rows[0]["nodes"], "relationships": rows[0]["rels"],
                "labels": {r["label"]: r["n"] for r in labels}}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}