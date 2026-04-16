from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.app.ingestion.registry_utils import (
    get_collection_docs,
    load_registry,
    sync_collection_from_manifest,
    upsert_collection_doc,
)


def test_load_registry_normalizes_missing_file(tmp_path: Path) -> None:
    registry = load_registry(tmp_path / "kb_registry.json")

    assert registry == {"collections": {}}


def test_sync_collection_from_manifest_hydrates_registry_docs(tmp_path: Path) -> None:
    manifest_path = tmp_path / "medical_research_chunks_v1_rebuild_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "collection": "medical_research_chunks_v1",
                "ingestion_version": "ingestion_v2",
                "chunker_version": "chunking_v2",
                "chunking_version": "chunking_v2",
                "parser": "docling",
                "docs": [
                    {
                        "doc_id": "DOC-1",
                        "source_file": "doc1.pdf",
                        "local_file": "C:/docs/doc1.pdf",
                        "source_sha256": "abc123",
                        "file_size_bytes": 2048,
                        "chunk_count": 4,
                        "text_chunk_count": 3,
                        "table_chunk_count": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    registry = {"collections": {}}
    sync_collection_from_manifest(
        registry,
        collection_name="medical_research_chunks_v1",
        manifest_path=manifest_path,
    )

    docs = get_collection_docs(registry, "medical_research_chunks_v1")
    assert docs["DOC-1"]["pdf_path"] == "C:/docs/doc1.pdf"
    assert docs["DOC-1"]["chunks"] == 4
    assert docs["DOC-1"]["parser"] == "docling"
    assert docs["DOC-1"]["source_sha256"] == "abc123"
    assert docs["DOC-1"]["file_size_bytes"] == 2048
    assert docs["DOC-1"]["chunker_version"] == "chunking_v2"
    assert registry["collections"]["medical_research_chunks_v1"]["doc_count"] == 1
    assert registry["collections"]["medical_research_chunks_v1"]["chunk_count"] == 4
    assert registry["collections"]["medical_research_chunks_v1"]["parser"] == "docling"
    assert registry["collections"]["medical_research_chunks_v1"]["chunker_version"] == "chunking_v2"


def test_upsert_collection_doc_recalculates_collection_totals() -> None:
    registry = {"collections": {}}

    upsert_collection_doc(
        registry,
        collection_name="medical_research_chunks_v1",
        doc_id="DOC-1",
        summary={
            "pdf_path": "C:/docs/doc1.pdf",
            "chunks": 4,
            "text_chunks": 3,
            "table_chunks": 1,
            "ingestion_version": "ingestion_v2",
            "chunker_version": "chunking_v2",
            "chunking_version": "chunking_v2",
            "parser": "docling",
            "source_sha256": "abc123",
            "file_size_bytes": 2048,
        },
    )
    upsert_collection_doc(
        registry,
        collection_name="medical_research_chunks_v1",
        doc_id="DOC-2",
        summary={
            "pdf_path": "C:/docs/doc2.pdf",
            "chunks": 6,
            "text_chunks": 6,
            "table_chunks": 0,
            "ingestion_version": "ingestion_v2",
            "chunker_version": "chunking_v2",
            "chunking_version": "chunking_v2",
            "parser": "docling",
            "source_sha256": "def456",
            "file_size_bytes": 4096,
        },
    )

    collection_entry = registry["collections"]["medical_research_chunks_v1"]
    assert collection_entry["doc_count"] == 2
    assert collection_entry["chunk_count"] == 10
    assert collection_entry["parser"] == "docling"
    assert collection_entry["chunker_version"] == "chunking_v2"
    assert collection_entry["docs"]["DOC-2"]["pdf_path"] == "C:/docs/doc2.pdf"
    assert collection_entry["docs"]["DOC-2"]["source_sha256"] == "def456"
    assert collection_entry["docs"]["DOC-2"]["file_size_bytes"] == 4096


def test_sync_collection_from_manifest_replaces_legacy_flat_doc_entries(tmp_path: Path) -> None:
    manifest_path = tmp_path / "medical_research_chunks_v1_rebuild_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "collection": "medical_research_chunks_v1",
                "ingestion_version": "ingestion_v2",
                "chunker_version": "chunking_v2",
                "chunking_version": "chunking_v2",
                "parser": "docling",
                "docs": [
                    {
                        "doc_id": "DOC-1",
                        "source_file": "doc1.pdf",
                        "local_file": "C:/docs/doc1.pdf",
                        "chunk_count": 4,
                        "text_chunk_count": 3,
                        "table_chunk_count": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    registry = {
        "collections": {
            "medical_research_chunks_v1": {
                "DOC-1": {
                    "doc_id": "DOC-1",
                    "pdf_path": "stale.pdf",
                    "chunks": 2,
                    "text_chunks": 2,
                    "table_chunks": 0,
                }
            }
        }
    }

    sync_collection_from_manifest(
        registry,
        collection_name="medical_research_chunks_v1",
        manifest_path=manifest_path,
    )

    collection_entry = registry["collections"]["medical_research_chunks_v1"]
    assert "DOC-1" not in collection_entry
    assert collection_entry["docs"]["DOC-1"]["chunks"] == 4


def test_upsert_collection_doc_rejects_duplicate_pdf_path_for_other_doc() -> None:
    registry = {"collections": {}}

    upsert_collection_doc(
        registry,
        collection_name="medical_research_chunks_v1",
        doc_id="DOC-1",
        summary={
            "pdf_path": "C:/docs/doc.pdf",
            "source_file": "doc.pdf",
            "chunks": 4,
            "text_chunks": 3,
            "table_chunks": 1,
            "ingestion_version": "ingestion_v2",
            "chunker_version": "chunking_v2",
            "chunking_version": "chunking_v2",
            "parser": "docling",
        },
    )

    with pytest.raises(ValueError, match="local_file 'C:/docs/doc.pdf' is already registered"):
        upsert_collection_doc(
            registry,
            collection_name="medical_research_chunks_v1",
            doc_id="DOC-2",
            summary={
                "pdf_path": "C:/docs/doc.pdf",
                "source_file": "doc-renamed.pdf",
                "chunks": 4,
                "text_chunks": 3,
                "table_chunks": 1,
                "ingestion_version": "ingestion_v2",
                "chunker_version": "chunking_v2",
                "chunking_version": "chunking_v2",
                "parser": "docling",
            },
        )
