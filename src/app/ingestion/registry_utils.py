from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.app.ingestion.dedup_utils import ensure_doc_identity_is_available, validate_unique_doc_identities


def default_manifest_path_for_collection(collection_name: str) -> Path:
    return Path("data/ingestion_manifests") / f"{collection_name}_rebuild_manifest.json"


def load_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.exists():
        return {"collections": {}}

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"collections": {}}
    collections = payload.get("collections")
    if not isinstance(collections, dict):
        payload["collections"] = {}
        return payload
    for item in collections.values():
        if isinstance(item, dict):
            _normalize_collection_entry(item)
    return payload


def save_registry(path: str | Path, registry: dict[str, Any]) -> None:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def sync_collection_from_manifest(
    registry: dict[str, Any],
    *,
    collection_name: str,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    collection_entry = _ensure_collection_entry(registry, collection_name)
    _normalize_collection_entry(collection_entry)
    effective_manifest_path = Path(manifest_path) if manifest_path is not None else default_manifest_path_for_collection(collection_name)
    if not effective_manifest_path.exists():
        return collection_entry

    manifest_payload = json.loads(effective_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        return collection_entry

    docs_payload = manifest_payload.get("docs")
    docs: dict[str, dict[str, Any]] = {}
    if isinstance(docs_payload, list):
        validate_unique_doc_identities(docs_payload, context="Rebuild manifest")
        for item in docs_payload:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("doc_id", "")).strip()
            if not doc_id:
                continue
            docs[doc_id] = {
                "doc_id": doc_id,
                "pdf_path": str(item.get("local_file", "")).strip(),
                "source_file": str(item.get("source_file", "")).strip(),
                "chunks": int(item.get("chunk_count", 0)),
                "text_chunks": int(item.get("text_chunk_count", 0)),
                "table_chunks": int(item.get("table_chunk_count", 0)),
                "ingestion_version": str(
                    item.get("ingestion_version", manifest_payload.get("ingestion_version", ""))
                ).strip(),
                "chunking_version": str(
                    item.get("chunking_version", manifest_payload.get("chunking_version", ""))
                ).strip(),
            }

    collection_entry["docs"] = docs
    collection_entry["manifest_path"] = str(effective_manifest_path)
    collection_entry["doc_count"] = len(docs)
    collection_entry["chunk_count"] = sum(item["chunks"] for item in docs.values())
    collection_entry["ingestion_version"] = str(manifest_payload.get("ingestion_version", "")).strip()
    collection_entry["chunking_version"] = str(manifest_payload.get("chunking_version", "")).strip()
    return collection_entry


def upsert_collection_doc(
    registry: dict[str, Any],
    *,
    collection_name: str,
    doc_id: str,
    summary: dict[str, Any],
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    collection_entry = _ensure_collection_entry(registry, collection_name)
    _normalize_collection_entry(collection_entry)
    docs = collection_entry.setdefault("docs", {})
    if not isinstance(docs, dict):
        docs = {}
        collection_entry["docs"] = docs

    ensure_doc_identity_is_available(
        doc_id=doc_id,
        source_file=str(summary.get("source_file", "")).strip(),
        local_file=str(summary.get("pdf_path", "")).strip(),
        existing_entries=list(docs.values()),
        context=f"Registry collection '{collection_name}'",
        allowed_doc_ids={doc_id} if doc_id in docs else set(),
    )

    docs[doc_id] = {
        "doc_id": doc_id,
        "pdf_path": str(summary.get("pdf_path", "")).strip(),
        "source_file": str(summary.get("source_file", "")).strip(),
        "chunks": int(summary.get("chunks", 0)),
        "text_chunks": int(summary.get("text_chunks", 0)),
        "table_chunks": int(summary.get("table_chunks", 0)),
        "ingestion_version": str(summary.get("ingestion_version", "")).strip(),
        "chunking_version": str(summary.get("chunking_version", "")).strip(),
    }

    effective_manifest_path = Path(manifest_path) if manifest_path is not None else default_manifest_path_for_collection(collection_name)
    if effective_manifest_path.exists():
        collection_entry["manifest_path"] = str(effective_manifest_path)

    collection_entry["doc_count"] = len(docs)
    collection_entry["chunk_count"] = sum(int(item.get("chunks", 0)) for item in docs.values())
    collection_entry["ingestion_version"] = str(summary.get("ingestion_version", collection_entry.get("ingestion_version", ""))).strip()
    collection_entry["chunking_version"] = str(summary.get("chunking_version", collection_entry.get("chunking_version", ""))).strip()
    return collection_entry


def get_collection_docs(registry: dict[str, Any], collection_name: str) -> dict[str, dict[str, Any]]:
    collections = registry.get("collections", {})
    if not isinstance(collections, dict):
        return {}
    collection_entry = collections.get(collection_name, {})
    if not isinstance(collection_entry, dict):
        return {}
    docs = collection_entry.get("docs", {})
    if not isinstance(docs, dict):
        return {}
    return docs


def _ensure_collection_entry(registry: dict[str, Any], collection_name: str) -> dict[str, Any]:
    collections = registry.setdefault("collections", {})
    if not isinstance(collections, dict):
        collections = {}
        registry["collections"] = collections

    collection_entry = collections.setdefault(collection_name, {})
    if not isinstance(collection_entry, dict):
        collection_entry = {}
        collections[collection_name] = collection_entry
    return collection_entry


def _normalize_collection_entry(collection_entry: dict[str, Any]) -> None:
    metadata_keys = {
        "docs",
        "manifest_path",
        "doc_count",
        "chunk_count",
        "ingestion_version",
        "chunking_version",
    }
    docs = collection_entry.get("docs")
    if not isinstance(docs, dict):
        docs = {}
    for key, value in list(collection_entry.items()):
        if key in metadata_keys:
            continue
        if isinstance(value, dict):
            docs.setdefault(key, value)
            collection_entry.pop(key, None)
    collection_entry["docs"] = docs
