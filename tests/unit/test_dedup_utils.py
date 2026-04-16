from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.app.ingestion.dedup_utils import (
    DuplicateDocumentError,
    ensure_doc_identity_is_available,
    fetch_collection_doc_identities,
    validate_unique_doc_identities,
)


def test_validate_unique_doc_identities_rejects_duplicate_source_file() -> None:
    with pytest.raises(DuplicateDocumentError, match="source_file 'doc.pdf' is already registered"):
        validate_unique_doc_identities(
            [
                {
                    "doc_id": "DOC-1",
                    "source_file": "doc.pdf",
                    "local_file": "C:/docs/doc-one.pdf",
                },
                {
                    "doc_id": "DOC-2",
                    "source_file": "doc.pdf",
                    "local_file": "C:/docs/doc-two.pdf",
                },
            ],
            context="Rebuild manifest",
        )


def test_validate_unique_doc_identities_rejects_duplicate_source_sha256() -> None:
    with pytest.raises(DuplicateDocumentError, match="source_sha256 'abc123' is already registered"):
        validate_unique_doc_identities(
            [
                {
                    "doc_id": "DOC-1",
                    "source_file": "doc1.pdf",
                    "local_file": "C:/docs/doc-one.pdf",
                    "source_sha256": "abc123",
                },
                {
                    "doc_id": "DOC-2",
                    "source_file": "doc2.pdf",
                    "local_file": "C:/docs/doc-two.pdf",
                    "source_sha256": "abc123",
                },
            ],
            context="Rebuild manifest",
        )


def test_ensure_doc_identity_is_available_allows_same_doc_id_for_repair() -> None:
    ensure_doc_identity_is_available(
        doc_id="DOC-1",
        source_file="doc-repaired.pdf",
        local_file="C:/docs/doc-repaired.pdf",
        existing_entries=[
            {
                "doc_id": "DOC-1",
                "source_file": "doc.pdf",
                "local_file": "C:/docs/doc.pdf",
            }
        ],
        context="Qdrant collection 'medical_research_chunks_v1'",
        allowed_doc_ids={"DOC-1"},
    )


def test_ensure_doc_identity_is_available_rejects_other_doc_with_same_local_file() -> None:
    with pytest.raises(DuplicateDocumentError, match="local_file 'C:/docs/doc.pdf' is already registered"):
        ensure_doc_identity_is_available(
            doc_id="DOC-2",
            source_file="renamed.pdf",
            local_file="C:/docs/doc.pdf",
            existing_entries=[
                {
                    "doc_id": "DOC-1",
                    "source_file": "doc.pdf",
                    "local_file": "C:/docs/doc.pdf",
                }
            ],
            context="Registry collection 'medical_research_chunks_v1'",
        )


def test_ensure_doc_identity_is_available_rejects_other_doc_with_same_source_sha256() -> None:
    with pytest.raises(DuplicateDocumentError, match="source_sha256 'abc123' is already registered"):
        ensure_doc_identity_is_available(
            doc_id="DOC-2",
            source_file="renamed.pdf",
            local_file="C:/docs/renamed.pdf",
            source_sha256="abc123",
            existing_entries=[
                {
                    "doc_id": "DOC-1",
                    "source_file": "doc.pdf",
                    "local_file": "C:/docs/doc.pdf",
                    "source_sha256": "abc123",
                }
            ],
            context="Registry collection 'medical_research_chunks_v1'",
        )


class _FakeScrollClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name == "medical_research_chunks_v1"

    def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter: object,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
        offset: object,
    ) -> tuple[list[object], object]:
        self.calls.append(
            {
                "collection_name": collection_name,
                "scroll_filter": scroll_filter,
                "limit": limit,
                "with_payload": with_payload,
                "with_vectors": with_vectors,
                "offset": offset,
            }
        )
        if offset is None:
            return (
                [
                    SimpleNamespace(
                        payload={
                            "doc_id": "DOC-2",
                            "source_file": "doc2.pdf",
                            "local_file": "C:/docs/doc2.pdf",
                            "source_sha256": "hash-2",
                        }
                    ),
                    SimpleNamespace(
                        payload={
                            "doc_id": "DOC-1",
                            "source_file": "doc1.pdf",
                            "local_file": "C:/docs/doc1.pdf",
                            "source_sha256": "hash-1",
                        }
                    ),
                ],
                "page-2",
            )
        return (
            [
                SimpleNamespace(
                    payload={
                        "doc_id": "DOC-1",
                        "source_file": "",
                        "local_file": "",
                    }
                )
            ],
            None,
        )


def test_fetch_collection_doc_identities_reduces_qdrant_points_to_doc_summaries() -> None:
    client = _FakeScrollClient()

    identities = fetch_collection_doc_identities(client, collection_name="medical_research_chunks_v1")

    assert identities == [
        {
            "doc_id": "DOC-1",
            "source_file": "doc1.pdf",
            "local_file": "C:/docs/doc1.pdf",
            "source_sha256": "hash-1",
        },
        {
            "doc_id": "DOC-2",
            "source_file": "doc2.pdf",
            "local_file": "C:/docs/doc2.pdf",
            "source_sha256": "hash-2",
        },
    ]
    assert client.calls[0]["with_payload"] is True
    assert client.calls[0]["with_vectors"] is False
