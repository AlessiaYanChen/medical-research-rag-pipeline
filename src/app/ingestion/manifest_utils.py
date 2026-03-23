from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.app.ingestion.dedup_utils import validate_unique_doc_identities
from src.domain.models.chunk import Chunk


def build_manifest_doc_entry(
    *,
    doc_id: str,
    source_file: str,
    local_file: str,
    chunks: list[Chunk],
    ingestion_version: str,
    chunking_version: str,
) -> dict[str, Any]:
    text_chunks = sum(chunk.metadata.chunk_type == "text" for chunk in chunks)
    table_chunks = sum(chunk.metadata.chunk_type == "table" for chunk in chunks)
    return {
        "doc_id": doc_id,
        "source_file": source_file,
        "local_file": local_file,
        "chunk_count": len(chunks),
        "text_chunk_count": text_chunks,
        "table_chunk_count": table_chunks,
        "ingestion_version": ingestion_version,
        "chunking_version": chunking_version,
    }


def write_rebuild_manifest(
    *,
    manifest_path: str | Path,
    collection: str,
    pdf_dir: str,
    glob_pattern: str,
    docs: list[dict[str, Any]],
    ingestion_version: str,
    chunking_version: str,
) -> None:
    validate_unique_doc_identities(docs, context="Rebuild manifest")
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "collection": collection,
        "pdf_dir": pdf_dir,
        "glob": glob_pattern,
        "doc_count": len(docs),
        "chunk_count": sum(int(doc["chunk_count"]) for doc in docs),
        "ingestion_version": ingestion_version,
        "chunking_version": chunking_version,
        "docs": docs,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def upsert_manifest_doc_entry(
    *,
    manifest_path: str | Path,
    collection: str,
    doc_entry: dict[str, Any],
    ingestion_version: str,
    chunking_version: str,
) -> None:
    path = Path(manifest_path)
    payload: dict[str, Any]
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        payload = loaded if isinstance(loaded, dict) else {}
    else:
        payload = {}

    payload["collection"] = collection
    payload["ingestion_version"] = ingestion_version
    payload["chunking_version"] = chunking_version

    docs = payload.get("docs")
    if not isinstance(docs, list):
        docs = []

    updated = False
    for index, item in enumerate(docs):
        if not isinstance(item, dict):
            continue
        if str(item.get("doc_id", "")).strip() == doc_entry["doc_id"]:
            docs[index] = doc_entry
            updated = True
            break

    if not updated:
        docs.append(doc_entry)

    docs.sort(key=lambda item: str(item.get("doc_id", "")).lower())
    validate_unique_doc_identities(docs, context="Rebuild manifest")
    payload["docs"] = docs
    payload["doc_count"] = len(docs)
    payload["chunk_count"] = sum(int(item.get("chunk_count", 0)) for item in docs if isinstance(item, dict))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
