"""Load a PDF directly with PyMuPDF, one Document per page.

Phase 1 already produces structured JSON; this loader exists for the case where
you want the raw page text rather than the extractor's interpretation of it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf
from langchain_core.documents import Document


def load_pdf(path: str | Path, skip_empty: bool = True) -> list[Document]:
    path = Path(path)
    docs: list[Document] = []
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text() or ""
            text = re.sub(r"[ \t]{2,}", " ", text).strip()
            if skip_empty and not text:
                continue
            first = next((l.strip() for l in text.splitlines() if l.strip()), "")
            docs.append(Document(
                page_content=text,
                metadata={"recipe_id": f"p_{index:04d}", "title": first[:180],
                          "source_type": "pdf", "origin": path.name,
                          "pages": [index], "detected_by": "pdf-page",
                          "source_file": path.name},
            ))
    return docs