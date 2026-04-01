from __future__ import annotations

from unittest.mock import patch

from src.app.ports.repositories.vector_repository import MetadataFilter, VectorSearchFilters
from src.app.services import retrieval_service as retrieval_service_module
from src.app.services.retrieval_service import RetrievalResult, RetrievalService
from src.domain.models.chunk import Chunk, ChunkMetadata


class FakeQdrantRepository:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self.search_calls: list[tuple[list[float], str | None, int, VectorSearchFilters | None]] = []

    def upsert_chunks(self, chunks: list[Chunk]) -> None:  # pragma: no cover
        raise NotImplementedError

    def search(
        self,
        vector: list[float],
        doc_id: str | None = None,
        limit: int = 5,
        filters: VectorSearchFilters | None = None,
    ) -> list[Chunk]:
        self.search_calls.append((vector, doc_id, limit, filters))
        filtered = self._chunks
        if doc_id is not None:
            filtered = [chunk for chunk in filtered if chunk.metadata.doc_id == doc_id]
        if filters is not None:
            if filters.doc_id is not None:
                filtered = [chunk for chunk in filtered if chunk.metadata.doc_id == filters.doc_id]
            filtered = [chunk for chunk in filtered if self._matches_all(chunk, filters.must)]
            filtered = [chunk for chunk in filtered if not self._matches_any(chunk, filters.must_not)]
            if filters.should:
                filtered = [
                    chunk for chunk in filtered if self._match_count(chunk, filters.should) >= filters.minimum_should_match
                ]
        return filtered[:limit]

    @staticmethod
    def _matches_all(chunk: Chunk, filters: tuple[MetadataFilter, ...]) -> bool:
        return all(FakeQdrantRepository._matches_filter(chunk, item) for item in filters)

    @staticmethod
    def _matches_any(chunk: Chunk, filters: tuple[MetadataFilter, ...]) -> bool:
        return any(FakeQdrantRepository._matches_filter(chunk, item) for item in filters)

    @staticmethod
    def _match_count(chunk: Chunk, filters: tuple[MetadataFilter, ...]) -> int:
        return sum(1 for item in filters if FakeQdrantRepository._matches_filter(chunk, item))

    @staticmethod
    def _matches_filter(chunk: Chunk, filter_item: MetadataFilter) -> bool:
        value = chunk.metadata.extra.get(filter_item.key)
        if value is None:
            if filter_item.key == "doc_id":
                value = chunk.metadata.doc_id
            elif filter_item.key == "chunk_type":
                value = chunk.metadata.chunk_type
            elif filter_item.key == "parent_header":
                value = chunk.metadata.parent_header

        if filter_item.values:
            if isinstance(value, list):
                return any(item in filter_item.values for item in value)
            return value in filter_item.values
        return value == filter_item.value


def test_retrieve_with_diagnostics_returns_wrapper_with_latency_and_candidate_count() -> None:
    retained_chunk = Chunk(
        id="DOC-1:00001",
        content="This retained evidence chunk contains enough detail to survive retrieval filtering.",
        metadata=ChunkMetadata(
            doc_id="DOC-1",
            chunk_type="text",
            parent_header="Results",
        ),
    )
    filtered_chunk = Chunk(
        id="DOC-1:00002",
        content="Contact: lead.author@example.com",
        metadata=ChunkMetadata(
            doc_id="DOC-1",
            chunk_type="text",
            parent_header="Front Matter",
        ),
    )
    repo = FakeQdrantRepository([retained_chunk, filtered_chunk])
    embedding_calls: list[list[str]] = []

    def embedding_fn(texts: list[str]) -> list[list[float]]:
        embedding_calls.append(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]

    service = RetrievalService(repo=repo, embedding_fn=embedding_fn)

    result = service.retrieve_with_diagnostics(query="retained evidence", doc_id="DOC-1", limit=2)

    assert isinstance(result, RetrievalResult)
    assert len(result.chunks) == 1
    assert result.chunks[0].doc_id == "DOC-1"
    assert result.chunks[0].source == "Results"
    assert result.latency_ms >= 0
    assert result.initial_candidate_count >= len(result.chunks)
    assert result.initial_candidate_count == 2
    assert len(embedding_calls) == 1
    assert len(repo.search_calls) == 1


def test_retrieve_with_diagnostics_times_full_retrieve_call() -> None:
    chunk = Chunk(
        id="DOC-3:00001",
        content="This retained evidence chunk contains enough detail to survive retrieval filtering.",
        metadata=ChunkMetadata(
            doc_id="DOC-3",
            chunk_type="text",
            parent_header="Results",
        ),
    )
    repo = FakeQdrantRepository([chunk])
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    with patch.object(retrieval_service_module.time, "perf_counter", side_effect=[100.0, 100.25]):
        result = service.retrieve_with_diagnostics(query="retained evidence", doc_id="DOC-3", limit=1)

    assert result.latency_ms == 250.0


def test_retrieve_still_returns_retrieved_chunks_for_backward_compatibility() -> None:
    chunk = Chunk(
        id="DOC-2:00001",
        content="This backward compatible retrieval chunk remains unchanged for existing callers.",
        metadata=ChunkMetadata(
            doc_id="DOC-2",
            chunk_type="text",
            parent_header="Discussion",
        ),
    )
    repo = FakeQdrantRepository([chunk])
    embedding_fn = lambda texts: [[0.4, 0.5, 0.6] for _ in texts]
    service = RetrievalService(repo=repo, embedding_fn=embedding_fn)

    result = service.retrieve(query="backward compatible retrieval", doc_id="DOC-2", limit=1)

    assert len(result) == 1
    assert result[0].doc_id == "DOC-2"
    assert result[0].source == "Discussion"
    assert result[0].content == "This backward compatible retrieval chunk remains unchanged for existing callers."
