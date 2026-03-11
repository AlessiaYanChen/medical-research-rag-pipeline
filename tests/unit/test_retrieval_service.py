from __future__ import annotations

from src.app.services.retrieval_service import RetrievalService
from src.domain.models.chunk import Chunk, ChunkMetadata


class FakeVectorRepository:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self.last_search_args: tuple[list[float], str | None, int] | None = None

    def upsert_chunks(self, chunks: list[Chunk]) -> None:  # pragma: no cover
        raise NotImplementedError

    def search(self, vector: list[float], doc_id: str | None = None, limit: int = 5) -> list[Chunk]:
        self.last_search_args = (vector, doc_id, limit)
        return self._chunks[:limit]


class FakeReRanker:
    def __init__(self, ordered_chunks: list[Chunk], should_fail: bool = False) -> None:
        self._ordered_chunks = ordered_chunks
        self._should_fail = should_fail
        self.last_args: tuple[str, list[Chunk], int] | None = None

    def rank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        self.last_args = (query, chunks, top_n)
        if self._should_fail:
            raise RuntimeError("re-ranker unavailable")
        return self._ordered_chunks[:top_n]


def test_retrieval_service_formats_header_and_content() -> None:
    chunks = [
        Chunk(
            id="DOC-1:00001",
            content="LDL-C was elevated in the treatment arm.",
            metadata=ChunkMetadata(
                doc_id="DOC-1",
                chunk_type="text",
                parent_header="Results",
                page_number=3,
            ),
        ),
        Chunk(
            id="DOC-1:00002",
            content="HDL-C remained stable across cohorts.",
            metadata=ChunkMetadata(
                doc_id="DOC-1",
                chunk_type="text",
                parent_header="Discussion",
                page_number=4,
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="lipid markers", doc_id="DOC-1", limit=2)

    assert repo.last_search_args == ([0.1, 0.2, 0.3], "DOC-1", 20)
    assert "Source: Results | Document: DOC-1\nLDL-C was elevated in the treatment arm." in result
    assert "Source: Discussion | Document: DOC-1\nHDL-C remained stable across cohorts." in result


def test_retrieval_service_strips_marker_anchor_and_image_noise() -> None:
    chunks = [
        Chunk(
            id="DOC-2:00001",
            content='<span id="page-3-0"></span>![](_page_3_Picture_1.jpeg)\n\nPneumonia is serious [1](#page-11-0).<br>Early treatment matters.',
            metadata=ChunkMetadata(
                doc_id="DOC-2",
                chunk_type="text",
                parent_header="**Introduction**",
                page_number=3,
            ),
        )
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="pneumonia", doc_id="DOC-2", limit=1)

    assert repo.last_search_args == ([0.1, 0.2, 0.3], "DOC-2", 20)
    assert "Source: Introduction | Document: DOC-2" in result
    assert "![](" not in result
    assert "<span" not in result
    assert "<br>" not in result
    assert "[1](#page-11-0)" not in result
    assert "Pneumonia is serious 1." in result
    assert "Early treatment matters." in result


def test_retrieval_service_can_query_entire_knowledge_base() -> None:
    chunks = [
        Chunk(
            id="DOC-A:00001",
            content="Biomarker A increased in severe cohorts.",
            metadata=ChunkMetadata(
                doc_id="DOC-A",
                chunk_type="text",
                parent_header="Results",
                page_number=2,
            ),
        )
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="biomarker", limit=1)

    assert repo.last_search_args == ([0.1, 0.2, 0.3], None, 20)
    assert "Document: DOC-A" in result


def test_retrieval_service_uses_re_ranker_output_order() -> None:
    first = Chunk(
        id="DOC-1:00001",
        content="Vector-first chunk",
        metadata=ChunkMetadata(
            doc_id="DOC-1",
            chunk_type="text",
            parent_header="Results",
            page_number=1,
        ),
    )
    second = Chunk(
        id="DOC-1:00002",
        content="Re-ranked best chunk",
        metadata=ChunkMetadata(
            doc_id="DOC-1",
            chunk_type="text",
            parent_header="Discussion",
            page_number=2,
        ),
    )
    repo = FakeVectorRepository([first, second])
    re_ranker = FakeReRanker([second, first])
    service = RetrievalService(
        repo=repo,
        embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        re_ranker=re_ranker,
    )

    result = service.retrieve(query="best evidence", doc_id="DOC-1", limit=2)

    assert re_ranker.last_args is not None
    assert re_ranker.last_args[0] == "best evidence"
    assert re_ranker.last_args[2] == 2
    assert result.index("Re-ranked best chunk") < result.index("Vector-first chunk")


def test_retrieval_service_falls_back_when_re_ranker_fails() -> None:
    first = Chunk(
        id="DOC-1:00001",
        content="Vector-first chunk",
        metadata=ChunkMetadata(
            doc_id="DOC-1",
            chunk_type="text",
            parent_header="Results",
            page_number=1,
        ),
    )
    second = Chunk(
        id="DOC-1:00002",
        content="Vector-second chunk",
        metadata=ChunkMetadata(
            doc_id="DOC-1",
            chunk_type="text",
            parent_header="Discussion",
            page_number=2,
        ),
    )
    repo = FakeVectorRepository([first, second])
    re_ranker = FakeReRanker([second, first], should_fail=True)
    service = RetrievalService(
        repo=repo,
        embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        re_ranker=re_ranker,
    )

    result = service.retrieve(query="fallback", doc_id="DOC-1", limit=2)

    assert result.index("Vector-first chunk") < result.index("Vector-second chunk")
