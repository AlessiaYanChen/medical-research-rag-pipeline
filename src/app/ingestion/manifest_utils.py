from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.app.ingestion.dedup_utils import validate_unique_doc_identities
from src.app.ingestion.versioning_utils import build_version_metadata
from src.domain.models.chunk import Chunk


def build_manifest_doc_entry(
    *,
    doc_id: str,
    source_file: str,
    local_file: str,
    chunks: list[Chunk],
    ingestion_version: str,
    chunking_version: str,
    parser_name: str = "",
    source_sha256: str = "",
    file_size_bytes: int | None = None,
) -> dict[str, Any]:
    text_chunks = sum(chunk.metadata.chunk_type == "text" for chunk in chunks)
    table_chunks = sum(chunk.metadata.chunk_type == "table" for chunk in chunks)
    payload = {
        "doc_id": doc_id,
        "source_file": source_file,
        "local_file": local_file,
        "chunk_count": len(chunks),
        "text_chunk_count": text_chunks,
        "table_chunk_count": table_chunks,
        "parser": str(parser_name).strip(),
    }
    payload.update(
        build_version_metadata(
            ingestion_version=ingestion_version,
            chunker_version=chunking_version,
        )
    )
    if str(source_sha256).strip():
        payload["source_sha256"] = str(source_sha256).strip()
    if file_size_bytes is not None:
        payload["file_size_bytes"] = int(file_size_bytes)
    return payload


def write_rebuild_manifest(
    *,
    manifest_path: str | Path,
    collection: str,
    pdf_dir: str,
    glob_pattern: str,
    docs: list[dict[str, Any]],
    ingestion_version: str,
    chunking_version: str,
    parser_name: str = "",
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
        "parser": str(parser_name).strip(),
        "docs": docs,
    }
    payload.update(
        build_version_metadata(
            ingestion_version=ingestion_version,
            chunker_version=chunking_version,
        )
    )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def upsert_manifest_doc_entry(
    *,
    manifest_path: str | Path,
    collection: str,
    doc_entry: dict[str, Any],
    ingestion_version: str,
    chunking_version: str,
    parser_name: str = "",
) -> None:
    path = Path(manifest_path)
    payload: dict[str, Any]
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        payload = loaded if isinstance(loaded, dict) else {}
    else:
        payload = {}

    payload["collection"] = collection
    payload["parser"] = str(parser_name).strip()
    payload.update(
        build_version_metadata(
            ingestion_version=ingestion_version,
            chunker_version=chunking_version,
        )
    )

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
