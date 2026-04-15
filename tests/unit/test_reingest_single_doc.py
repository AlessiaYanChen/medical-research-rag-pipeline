from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.domain.models.chunk import Chunk, ChunkMetadata

import scripts.reingest_single_doc as reingest_single_doc
from scripts.reingest_single_doc import (
    build_failure_record,
    load_manifest_json_object,
    resolve_failure_report_path,
    write_failure_report,
)


def test_reingest_build_failure_record_includes_stage_and_collection(tmp_path: Path) -> None:
    pdf_path = tmp_path / "study.pdf"
    pdf_path.write_text("placeholder", encoding="utf-8")

    failure = build_failure_record(
        pdf_path=pdf_path,
        doc_id="DOC-7",
        collection="medical_research_chunks_v1",
        stage="parse",
        error=RuntimeError("parse failed"),
    )

    assert failure == {
        "pdf_path": str(pdf_path),
        "doc_id": "DOC-7",
        "collection": "medical_research_chunks_v1",
        "stage": "parse",
        "error": "parse failed",
    }


def test_reingest_write_failure_report_writes_single_failure_payload(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "reingest_failure.json"
    failure = {
        "pdf_path": "C:/docs/study.pdf",
        "doc_id": "DOC-7",
        "collection": "medical_research_chunks_v1",
        "stage": "manifest_update",
        "error": "manifest write failed",
    }

    written_path = write_failure_report(
        output_path=output_path,
        failure=failure,
        manifest_path="data/ingestion_manifests/medical_research_chunks_v1_rebuild_manifest.json",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert written_path == output_path
    assert payload["doc_id"] == "DOC-7"
    assert payload["collection"] == "medical_research_chunks_v1"
    assert payload["pdf_path"] == "C:/docs/study.pdf"
    assert payload["manifest_path"] == "data/ingestion_manifests/medical_research_chunks_v1_rebuild_manifest.json"
    assert payload["failure"] == failure


def test_reingest_resolve_failure_report_path_uses_default_collection_and_doc_id() -> None:
    resolved = resolve_failure_report_path(
        output_path="",
        collection="medical_research_chunks_v1",
        doc_id="DOC-7",
    )

    assert resolved == Path("data/eval/results/reingest_failure_medical_research_chunks_v1_DOC-7.json")


def test_reingest_resolve_failure_report_path_sanitizes_doc_id() -> None:
    resolved = resolve_failure_report_path(
        output_path="",
        collection="medical_research_chunks_v1",
        doc_id="DOC:7/alpha",
    )

    assert resolved == Path("data/eval/results/reingest_failure_medical_research_chunks_v1_DOC_7_alpha.json")


def test_load_manifest_json_object_reads_object_payload(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"collection": "medical_research_chunks_v1"}), encoding="utf-8")

    payload = load_manifest_json_object(manifest_path)

    assert payload == {"collection": "medical_research_chunks_v1"}


def test_load_manifest_json_object_rejects_invalid_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match=r"manifest is not valid JSON: .*line 1, column 2"):
        load_manifest_json_object(manifest_path)


def test_load_manifest_json_object_rejects_non_object_payload(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest is not a JSON object"):
        load_manifest_json_object(manifest_path)


def test_main_returns_failure_when_upsert_new_doc_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "study.pdf"
    pdf_path.write_text("placeholder", encoding="utf-8")
    failure_report_path = tmp_path / "failure.json"

    class FakeEmbeddingAdapter:
        def __init__(self, **_: object) -> None:
            pass

        def __call__(self, texts: list[str]) -> list[list[float]]:
            return [[0.0, 0.0, 0.0] for _ in texts]

    class FakeParser:
        def parse(self, path: Path) -> object:
            assert path == pdf_path
            return type("ParsedDocument", (), {"markdown_text": "content", "tables": []})()

    class FakeChunker:
        INGESTION_VERSION = "ingestion_v2"
        CHUNKING_VERSION = "chunking_v2"

        def __init__(self, **_: object) -> None:
            pass

        def chunk_document(self, **_: object) -> list[Chunk]:
            return [
                Chunk(
                    id="DOC-7:00001",
                    content="content",
                    metadata=ChunkMetadata(
                        doc_id="DOC-7",
                        chunk_type="text",
                        parent_header="Results",
                    ),
                )
            ]

    class FakeClient:
        def __init__(self, **_: object) -> None:
            self.deleted = False
            self.restored_points: list[object] = []

        def scroll(self, **_: object) -> tuple[list[object], None]:
            return (
                [
                    type(
                        "Point",
                        (),
                        {
                            "id": "old-point-id",
                            "payload": {
                                "chunk_id": "DOC-7:00001",
                                "doc_id": "DOC-7",
                                "chunk_type": "text",
                                "parent_header": "Results",
                                "content": "old content",
                            },
                            "vector": [0.1, 0.2, 0.3],
                        },
                    )()
                ],
                None,
            )

        def delete(self, **_: object) -> None:
            self.deleted = True

        def upsert(self, *, points: list[object], **_: object) -> None:
            self.restored_points = points

    class FailingRepository:
        def __init__(self, **_: object) -> None:
            pass

        def upsert_chunks(self, chunks: list[Chunk]) -> None:
            assert len(chunks) == 1
            raise RuntimeError("simulated write failure")

    monkeypatch.setattr(
        "sys.argv",
        [
            "reingest_single_doc.py",
            "--doc-id",
            "DOC-7",
            "--pdf",
            str(pdf_path),
            "--collection",
            "medical_research_chunks_v1",
            "--failure-report-out",
            str(failure_report_path),
            "--embedding-api-key",
            "test-key",
        ],
    )
    monkeypatch.setattr(reingest_single_doc, "OpenAIEmbeddingAdapter", FakeEmbeddingAdapter)
    monkeypatch.setattr(reingest_single_doc, "build_parser", lambda _: FakeParser())
    monkeypatch.setattr(reingest_single_doc, "normalize_tables", lambda tables, file_name: [])
    monkeypatch.setattr(reingest_single_doc, "UnifiedChunker", FakeChunker)
    monkeypatch.setattr(reingest_single_doc, "QdrantClient", FakeClient)
    monkeypatch.setattr(reingest_single_doc, "QdrantRepository", FailingRepository)
    monkeypatch.setattr(reingest_single_doc, "compute_file_identity", lambda _: {"source_sha256": "abc", "file_size_bytes": 12})
    monkeypatch.setattr(reingest_single_doc, "ensure_doc_identity_is_available", lambda **_: None)
    monkeypatch.setattr(reingest_single_doc, "fetch_collection_doc_identities", lambda *_, **__: [])

    exit_code = reingest_single_doc.main()

    assert exit_code == 1
    payload = json.loads(failure_report_path.read_text(encoding="utf-8"))
    assert payload["failure"]["stage"] == "upsert_new_doc"
    assert payload["failure"]["error"] == "simulated write failure"


def test_main_reports_rollback_failure_when_restore_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "study.pdf"
    pdf_path.write_text("placeholder", encoding="utf-8")
    failure_report_path = tmp_path / "failure.json"

    class FakeEmbeddingAdapter:
        def __init__(self, **_: object) -> None:
            pass

        def __call__(self, texts: list[str]) -> list[list[float]]:
            return [[0.0, 0.0, 0.0] for _ in texts]

    class FakeParser:
        def parse(self, path: Path) -> object:
            assert path == pdf_path
            return type("ParsedDocument", (), {"markdown_text": "content", "tables": []})()

    class FakeChunker:
        INGESTION_VERSION = "ingestion_v2"
        CHUNKING_VERSION = "chunking_v2"

        def __init__(self, **_: object) -> None:
            pass

        def chunk_document(self, **_: object) -> list[Chunk]:
            return [
                Chunk(
                    id="DOC-7:00001",
                    content="content",
                    metadata=ChunkMetadata(
                        doc_id="DOC-7",
                        chunk_type="text",
                        parent_header="Results",
                    ),
                )
            ]

    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def scroll(self, **_: object) -> tuple[list[object], None]:
            return (
                [
                    type(
                        "Point",
                        (),
                        {
                            "id": "old-point-id",
                            "payload": {
                                "chunk_id": "DOC-7:00001",
                                "doc_id": "DOC-7",
                                "chunk_type": "text",
                                "parent_header": "Results",
                                "content": "old content",
                            },
                            "vector": [0.1, 0.2, 0.3],
                        },
                    )()
                ],
                None,
            )

        def delete(self, **_: object) -> None:
            pass

        def upsert(self, **_: object) -> None:
            raise RuntimeError("simulated rollback failure")

    class FailingRepository:
        def __init__(self, **_: object) -> None:
            pass

        def upsert_chunks(self, chunks: list[Chunk]) -> None:
            assert len(chunks) == 1
            raise RuntimeError("simulated write failure")

    monkeypatch.setattr(
        "sys.argv",
        [
            "reingest_single_doc.py",
            "--doc-id",
            "DOC-7",
            "--pdf",
            str(pdf_path),
            "--collection",
            "medical_research_chunks_v1",
            "--failure-report-out",
            str(failure_report_path),
            "--embedding-api-key",
            "test-key",
        ],
    )
    monkeypatch.setattr(reingest_single_doc, "OpenAIEmbeddingAdapter", FakeEmbeddingAdapter)
    monkeypatch.setattr(reingest_single_doc, "build_parser", lambda _: FakeParser())
    monkeypatch.setattr(reingest_single_doc, "normalize_tables", lambda tables, file_name: [])
    monkeypatch.setattr(reingest_single_doc, "UnifiedChunker", FakeChunker)
    monkeypatch.setattr(reingest_single_doc, "QdrantClient", FakeClient)
    monkeypatch.setattr(reingest_single_doc, "QdrantRepository", FailingRepository)
    monkeypatch.setattr(reingest_single_doc, "compute_file_identity", lambda _: {"source_sha256": "abc", "file_size_bytes": 12})
    monkeypatch.setattr(reingest_single_doc, "ensure_doc_identity_is_available", lambda **_: None)
    monkeypatch.setattr(reingest_single_doc, "fetch_collection_doc_identities", lambda *_, **__: [])

    exit_code = reingest_single_doc.main()

    assert exit_code == 1
    payload = json.loads(failure_report_path.read_text(encoding="utf-8"))
    assert payload["failure"]["stage"] == "rollback_old_doc"
    assert "simulated write failure" in payload["failure"]["error"]
    assert "simulated rollback failure" in payload["failure"]["error"]
