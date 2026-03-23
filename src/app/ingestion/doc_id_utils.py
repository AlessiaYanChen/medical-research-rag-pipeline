from __future__ import annotations

from pathlib import Path


def doc_id_from_path(path: str | Path) -> str:
    pdf_path = Path(path)
    return normalize_doc_id(pdf_path.stem)


def normalize_doc_id(value: str) -> str:
    normalized = " ".join(str(value).strip().split())
    if not normalized:
        raise ValueError("Document ID must not be empty.")
    return normalized
