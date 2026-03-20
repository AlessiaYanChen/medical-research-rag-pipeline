from __future__ import annotations

import logging
from types import SimpleNamespace

from src.app.adapters.vectorstores.qdrant_repository import QdrantRepository
from src.app.ports.repositories.vector_repository import MetadataFilter, VectorSearchFilters
from src.domain.models.chunk import Chunk, ChunkMetadata


class FakeQdrantClient:
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._fail_on_call = fail_on_call
        self._call_counter = 0
        self.query_calls: list[dict[str, object]] = []

    def upsert(self, collection_name: str, points: list[object]) -> None:
        self._call_counter += 1
        if self._fail_on_call is not None and self._call_counter == self._fail_on_call:
            raise RuntimeError("simulated upsert failure")
        self.calls.append({"collection_name": collection_name, "points": points})

    def query_points(
        self,
        collection_name: str,
        query: list[float],
        query_filter: object,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ) -> object:
        self.query_calls.append(
            {
                "collection_name": collection_name,
                "query": query,
                "query_filter": query_filter,
                "limit": limit,
                "with_payload": with_payload,
                "with_vectors": with_vectors,
            }
        )
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="ignored-qdrant-id",
                    payload={
                        "chunk_id": "DOC-1:00001",
                        "content": "retrieved content",
                        "doc_id": "DOC-1",
                        "chunk_type": "text",
                        "parent_header": "Results",
                        "page_number": 3,
                    },
                )
            ]
        )


def _build_chunks(count: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i in range(count):
        chunks.append(
            Chunk(
                id=f"DOC-1:{i:05d}",
                content=f"content-{i}",
                metadata=ChunkMetadata(
                    doc_id="DOC-1",
                    chunk_type="text",
                    parent_header="Results",
                    page_number=1,
                ),
            )
        )
    return chunks


def test_qdrant_repository_upserts_in_batches_of_100() -> None:
    client = FakeQdrantClient()
    repo = QdrantRepository(
        qdrant_client=client,
        collection_name="medical_chunks",
        embedding_fn=lambda texts: [[float(i)] * 3 for i, _ in enumerate(texts)],
    )
    chunks = _build_chunks(250)

    repo.upsert_chunks(chunks)

    assert len(client.calls) == 3
    assert len(client.calls[0]["points"]) == 100
    assert len(client.calls[1]["points"]) == 100
    assert len(client.calls[2]["points"]) == 50


def test_qdrant_repository_maps_chunk_metadata_into_payload() -> None:
    client = FakeQdrantClient()
    repo = QdrantRepository(
        qdrant_client=client,
        collection_name="medical_chunks",
        embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
    )
    chunk = Chunk(
        id="DOC-9:00001",
        content="table content",
        metadata=ChunkMetadata(
            doc_id="DOC-9",
            chunk_type="table",
            parent_header="Lipid Panel",
            page_number=7,
        ),
    )

    repo.upsert_chunks([chunk])

    point = client.calls[0]["points"][0]
    point_id = point["id"] if isinstance(point, dict) else point.id
    payload = point["payload"] if isinstance(point, dict) else point.payload

    assert point_id != "DOC-9:00001"
    assert payload["doc_id"] == "DOC-9"
    assert payload["chunk_id"] == "DOC-9:00001"
    assert payload["chunk_type"] == "table"
    assert payload["parent_header"] == "Lipid Panel"
    assert payload["page_number"] == 7


def test_qdrant_repository_logs_success_and_failure_counts(caplog) -> None:
    client = FakeQdrantClient(fail_on_call=2)
    repo = QdrantRepository(
        qdrant_client=client,
        collection_name="medical_chunks",
        embedding_fn=lambda texts: [[0.0, 0.0, 0.0] for _ in texts],
    )
    chunks = _build_chunks(200)

    with caplog.at_level(logging.ERROR):
        repo.upsert_chunks(chunks)

    assert "stored=100 failed=100 total=200" in caplog.text


def test_qdrant_repository_search_reconstructs_chunk_from_payload() -> None:
    client = FakeQdrantClient()
    repo = QdrantRepository(
        qdrant_client=client,
        collection_name="medical_chunks",
        embedding_fn=lambda texts: [[0.0, 0.0, 0.0] for _ in texts],
    )

    results = repo.search([0.1, 0.2, 0.3], doc_id="DOC-1", limit=1)

    assert client.query_calls[0]["collection_name"] == "medical_chunks"
    assert client.query_calls[0]["query"] == [0.1, 0.2, 0.3]
    assert len(results) == 1
    assert results[0].id == "DOC-1:00001"
    assert results[0].metadata.doc_id == "DOC-1"
    assert results[0].metadata.parent_header == "Results"


def test_qdrant_repository_search_can_run_without_doc_filter() -> None:
    client = FakeQdrantClient()
    repo = QdrantRepository(
        qdrant_client=client,
        collection_name="medical_chunks",
        embedding_fn=lambda texts: [[0.0, 0.0, 0.0] for _ in texts],
    )

    repo.search([0.1, 0.2, 0.3], limit=1)

    assert client.query_calls[0]["query_filter"] is None


def test_qdrant_repository_search_builds_metadata_filters() -> None:
    client = FakeQdrantClient()
    repo = QdrantRepository(
        qdrant_client=client,
        collection_name="medical_chunks",
        embedding_fn=lambda texts: [[0.0, 0.0, 0.0] for _ in texts],
    )

    repo.search(
        [0.1, 0.2, 0.3],
        limit=1,
        filters=VectorSearchFilters(
            doc_id="DOC-1",
            must=(MetadataFilter(key="content_role", value="table"),),
            must_not=(MetadataFilter(key="section_role", values=("references", "front_matter")),),
            should=(MetadataFilter(key="referenced_table_indices", values=(3,)),),
        ),
    )

    query_filter = client.query_calls[0]["query_filter"]
    assert query_filter is not None
    assert any(condition.key == "doc_id" for condition in query_filter.must)
    assert any(condition.key == "content_role" for condition in query_filter.must)
    assert any(condition.key == "section_role" for condition in query_filter.must_not)
    assert query_filter.min_should is not None
    assert any(condition.key == "referenced_table_indices" for condition in query_filter.min_should.conditions)
