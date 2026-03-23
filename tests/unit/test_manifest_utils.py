from __future__ import annotations

import json
from pathlib import Path

from src.app.ingestion.manifest_utils import (
    build_manifest_doc_entry,
    upsert_manifest_doc_entry,
    write_rebuild_manifest,
)
from src.domain.models.chunk import Chunk, ChunkMetadata


def _build_chunks() -> list[Chunk]:
    return [
        Chunk(
            id="DOC-1:P00001:C01",
            content="Results text",
            metadata=ChunkMetadata(
                doc_id="DOC-1",
                chunk_type="text",
                parent_header="Results",
            ),
        ),
        Chunk(
            id="DOC-1:T00001",
            content="Table text",
            metadata=ChunkMetadata(
                doc_id="DOC-1",
                chunk_type="table",
                parent_header="Results",
            ),
        ),
    ]


def test_build_manifest_doc_entry_counts_text_and_table_chunks() -> None:
    entry = build_manifest_doc_entry(
        doc_id="DOC-1",
        source_file="doc.pdf",
        local_file="C:/docs/doc.pdf",
        chunks=_build_chunks(),
        ingestion_version="ingestion_v2",
        chunking_version="chunking_v2",
    )

    assert entry["doc_id"] == "DOC-1"
    assert entry["chunk_count"] == 2
    assert entry["text_chunk_count"] == 1
    assert entry["table_chunk_count"] == 1
    assert entry["ingestion_version"] == "ingestion_v2"
    assert entry["chunking_version"] == "chunking_v2"


def test_write_rebuild_manifest_writes_consistent_totals(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    docs = [
        build_manifest_doc_entry(
            doc_id="DOC-1",
            source_file="doc1.pdf",
            local_file="C:/docs/doc1.pdf",
            chunks=_build_chunks(),
            ingestion_version="ingestion_v2",
            chunking_version="chunking_v2",
        )
    ]

    write_rebuild_manifest(
        manifest_path=manifest_path,
        collection="medical_research_chunks_v1",
        pdf_dir="data/raw_pdfs/uploaded",
        glob_pattern="*.pdf",
        docs=docs,
        ingestion_version="ingestion_v2",
        chunking_version="chunking_v2",
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["collection"] == "medical_research_chunks_v1"
    assert payload["doc_count"] == 1
    assert payload["chunk_count"] == 2
    assert payload["docs"][0]["doc_id"] == "DOC-1"


def test_upsert_manifest_doc_entry_replaces_existing_doc_and_recalculates_totals(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "collection": "medical_research_chunks_v1",
                "doc_count": 1,
                "chunk_count": 5,
                "docs": [
                    {
                        "doc_id": "DOC-1",
                        "chunk_count": 5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    upsert_manifest_doc_entry(
        manifest_path=manifest_path,
        collection="medical_research_chunks_v1",
        doc_entry=build_manifest_doc_entry(
            doc_id="DOC-1",
            source_file="doc1.pdf",
            local_file="C:/docs/doc1.pdf",
            chunks=_build_chunks(),
            ingestion_version="ingestion_v2",
            chunking_version="chunking_v2",
        ),
        ingestion_version="ingestion_v2",
        chunking_version="chunking_v2",
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["doc_count"] == 1
    assert payload["chunk_count"] == 2
    assert payload["docs"][0]["doc_id"] == "DOC-1"
    assert payload["docs"][0]["text_chunk_count"] == 1


def test_upsert_manifest_doc_entry_appends_new_doc_and_sorts(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "collection": "medical_research_chunks_v1",
                "docs": [
                    {"doc_id": "DOC-B", "chunk_count": 3},
                ],
            }
        ),
        encoding="utf-8",
    )

    upsert_manifest_doc_entry(
        manifest_path=manifest_path,
        collection="medical_research_chunks_v1",
        doc_entry=build_manifest_doc_entry(
            doc_id="DOC-A",
            source_file="docA.pdf",
            local_file="C:/docs/docA.pdf",
            chunks=_build_chunks(),
            ingestion_version="ingestion_v2",
            chunking_version="chunking_v2",
        ),
        ingestion_version="ingestion_v2",
        chunking_version="chunking_v2",
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["doc_count"] == 2
    assert payload["chunk_count"] == 5
    assert [item["doc_id"] for item in payload["docs"]] == ["DOC-A", "DOC-B"]
