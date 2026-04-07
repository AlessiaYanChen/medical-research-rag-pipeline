from __future__ import annotations

from src.app.ports.repositories.vector_repository import MetadataFilter
from src.app.ports.repositories.vector_repository import VectorSearchFilters
from src.app.services.retrieval_service import RetrievalService
from src.domain.models.chunk import Chunk, ChunkMetadata


class FakeVectorRepository:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self.last_search_args: tuple[list[float], str | None, int, VectorSearchFilters | None] | None = None
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
        self.last_search_args = (vector, doc_id, limit, filters)
        self.search_calls.append((vector, doc_id, limit, filters))
        chunks = self._chunks
        if doc_id is None:
            filtered = chunks
        else:
            filtered = [chunk for chunk in chunks if chunk.metadata.doc_id == doc_id]

        if filters is not None:
            if filters.doc_id is not None:
                filtered = [chunk for chunk in filtered if chunk.metadata.doc_id == filters.doc_id]
            filtered = [chunk for chunk in filtered if self._matches_all(chunk, filters.must)]
            filtered = [chunk for chunk in filtered if not self._matches_any(chunk, filters.must_not)]
            if filters.should:
                filtered = [
                    chunk
                    for chunk in filtered
                    if self._match_count(chunk, filters.should) >= filters.minimum_should_match
                ]
        return filtered[:limit]

    @staticmethod
    def _matches_all(chunk: Chunk, filters: tuple[MetadataFilter, ...]) -> bool:
        return all(FakeVectorRepository._matches_filter(chunk, item) for item in filters)

    @staticmethod
    def _matches_any(chunk: Chunk, filters: tuple[MetadataFilter, ...]) -> bool:
        return any(FakeVectorRepository._matches_filter(chunk, item) for item in filters)

    @staticmethod
    def _match_count(chunk: Chunk, filters: tuple[MetadataFilter, ...]) -> int:
        return sum(1 for item in filters if FakeVectorRepository._matches_filter(chunk, item))

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

    assert repo.last_search_args is not None
    assert repo.last_search_args[:3] == ([0.1, 0.2, 0.3], "DOC-1", 40)
    assert result[0].source == "Results"
    assert result[0].doc_id == "DOC-1"
    assert result[0].content == "LDL-C was elevated in the treatment arm."
    assert result[0].chunk_type == "text"
    assert result[0].content_role == "text"
    assert result[1].source == "Discussion"
    assert result[1].content == "HDL-C remained stable across cohorts."


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

    assert repo.last_search_args is not None
    assert repo.last_search_args[:3] == ([0.1, 0.2, 0.3], "DOC-2", 40)
    assert result[0].source == "Introduction"
    assert "![](" not in result[0].content
    assert "<span" not in result[0].content
    assert "<br>" not in result[0].content
    assert "[1](#page-11-0)" not in result[0].content
    assert "Pneumonia is serious 1." in result[0].content
    assert "Early treatment matters." in result[0].content


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

    assert repo.search_calls[0][:3] == ([0.1, 0.2, 0.3], None, 40)
    assert result[0].doc_id == "DOC-A"


def test_retrieval_service_uses_re_ranker_output_order() -> None:
    first = Chunk(
        id="DOC-1:00001",
        content="Vector-first chunk with enough context to pass the noise filter.",
        metadata=ChunkMetadata(
            doc_id="DOC-1",
            chunk_type="text",
            parent_header="Results",
            page_number=1,
        ),
    )
    second = Chunk(
        id="DOC-1:00002",
        content="Re-ranked best chunk with enough context to remain visible.",
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
    assert re_ranker.last_args[2] == 20
    assert result[0].content.startswith("Re-ranked best chunk")
    assert result[1].content.startswith("Vector-first chunk")


def test_retrieval_service_falls_back_when_re_ranker_fails() -> None:
    first = Chunk(
        id="DOC-1:00001",
        content="Vector-first chunk with enough context to pass the noise filter.",
        metadata=ChunkMetadata(
            doc_id="DOC-1",
            chunk_type="text",
            parent_header="Results",
            page_number=1,
        ),
    )
    second = Chunk(
        id="DOC-1:00002",
        content="Vector-second chunk with enough context to pass the noise filter.",
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

    assert result[0].content.startswith("Vector-first chunk")
    assert result[1].content.startswith("Vector-second chunk")


def test_retrieval_service_filters_short_noise_chunks() -> None:
    chunks = [
        Chunk(
            id="DOC-3:00001",
            content="<sup></sup>author@example.com",
            metadata=ChunkMetadata(
                doc_id="DOC-3",
                chunk_type="text",
                parent_header="Front Matter",
                page_number=1,
            ),
        ),
        Chunk(
            id="DOC-3:00002",
            content="This retained chunk contains enough substantive text to survive filtering.",
            metadata=ChunkMetadata(
                doc_id="DOC-3",
                chunk_type="text",
                parent_header="Results",
                page_number=2,
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="substantive evidence", doc_id="DOC-3", limit=2)

    assert len(result) == 1
    assert result[0].source == "Results"
    assert "author@example.com" not in result[0].content
    assert "This retained chunk contains enough substantive text to survive filtering." in result[0].content


def test_retrieval_service_strips_empty_tags_and_email_noise() -> None:
    chunks = [
        Chunk(
            id="DOC-4:00001",
            content=(
                "<sup></sup><span></span>Contact: lead.author@example.com<br>"
                "The intervention reduced inflammatory markers across cohorts."
            ),
            metadata=ChunkMetadata(
                doc_id="DOC-4",
                chunk_type="text",
                parent_header="**Discussion**",
                page_number=5,
            ),
        )
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="inflammatory markers", doc_id="DOC-4", limit=1)

    assert "<sup>" not in result[0].content
    assert "<span>" not in result[0].content
    assert "lead.author@example.com" not in result[0].content
    assert "The intervention reduced inflammatory markers across cohorts." in result[0].content


def test_retrieval_service_keeps_document_metadata_abstract_available() -> None:
    chunks = [
        Chunk(
            id="DOC-4A:00001",
            content="Opening summary with enough context to be useful before structural headers appear.",
            metadata=ChunkMetadata(
                doc_id="DOC-4A",
                chunk_type="text",
                parent_header="Document Metadata/Abstract",
                page_number=1,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-4A:P00001",
                    "parent_content": "Opening summary with enough context to be useful before structural headers appear.",
                },
            ),
        )
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="opening summary", doc_id="DOC-4A", limit=1)

    assert len(result) == 1
    assert result[0].source == "Document Metadata/Abstract"


def test_retrieval_service_adds_table_caption_and_linked_context_to_returned_table_chunks() -> None:
    chunks = [
        Chunk(
            id="DOC-TABLE:T00001",
            content="Source File: study.pdf | Table Index: 3 | Section: Results\nMarker,Value\nSensitivity,0.88",
            metadata=ChunkMetadata(
                doc_id="DOC-TABLE",
                chunk_type="table",
                parent_header="Results",
                page_number=6,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "table_caption": "Table 3. Diagnostic accuracy summary",
                    "linked_table_contexts": [
                        "Table 3 summarizes discrepant diagnostic findings and explains the sensitivity gap."
                    ],
                },
            ),
        )
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(
        repo=repo,
        embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        include_tables=True,
    )

    result = service.retrieve(query="Which table reports diagnostic accuracy?", doc_id="DOC-TABLE", limit=1)

    assert "Table Caption: Table 3. Diagnostic accuracy summary" in result[0].content
    assert "Linked Context: Table 3 summarizes discrepant diagnostic findings and explains the sensitivity gap." in result[0].content
    assert "Marker,Value" in result[0].content


def test_retrieval_service_uses_normalized_header_for_display_and_ranking() -> None:
    chunks = [
        Chunk(
            id="DOC-4B:00001",
            content="Subsection content with enough context to remain visible.",
            metadata=ChunkMetadata(
                doc_id="DOC-4B",
                chunk_type="text",
                parent_header="Results",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-4B:P00001",
                    "parent_content": "Subsection content with enough context to remain visible.",
                    "original_parent_header": "Clinical Validation",
                    "normalized_parent_header": "Results",
                    "header_role": "subsection",
                },
            ),
        )
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="validation findings", doc_id="DOC-4B", limit=1)

    assert len(result) == 1
    assert result[0].source == "Results"


def test_retrieval_service_collapses_multiple_child_hits_to_one_parent() -> None:
    parent_text = (
        "This paragraph contains the full parent context for the intervention arm. "
        "It includes the biomarker result and the surrounding interpretation. "
        "The final sentence provides extra detail."
    )
    chunks = [
        Chunk(
            id="DOC-5:P00001:C01",
            content="biomarker result",
            metadata=ChunkMetadata(
                doc_id="DOC-5",
                chunk_type="text",
                parent_header="Results",
                page_number=7,
                extra={
                    "content_role": "child",
                    "parent_id": "DOC-5:P00001",
                    "parent_content": parent_text,
                    "parent_sentences": [
                        "This paragraph contains the full parent context for the intervention arm.",
                        "It includes the biomarker result and the surrounding interpretation.",
                        "The final sentence provides extra detail.",
                    ],
                    "child_index": 1,
                    "child_sentence_start": 1,
                    "child_sentence_end": 2,
                },
            ),
        ),
        Chunk(
            id="DOC-5:P00001:C02",
            content="surrounding interpretation",
            metadata=ChunkMetadata(
                doc_id="DOC-5",
                chunk_type="text",
                parent_header="Results",
                page_number=7,
                extra={
                    "content_role": "child",
                    "parent_id": "DOC-5:P00001",
                    "parent_content": parent_text,
                    "parent_sentences": [
                        "This paragraph contains the full parent context for the intervention arm.",
                        "It includes the biomarker result and the surrounding interpretation.",
                        "The final sentence provides extra detail.",
                    ],
                    "child_index": 2,
                    "child_sentence_start": 1,
                    "child_sentence_end": 2,
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="biomarker interpretation", doc_id="DOC-5", limit=2)

    assert len(result) == 1
    assert result[0].source == "Results"
    assert result[0].doc_id == "DOC-5"
    assert result[0].content == parent_text
    assert result[0].content_role == "child"


def test_retrieval_service_excludes_reference_and_unknown_sections() -> None:
    chunks = [
        Chunk(
            id="DOC-6:P00001:C01",
            content="Smith J et al. Pathogen detection study. doi:10.1000/example",
            metadata=ChunkMetadata(
                doc_id="DOC-6",
                chunk_type="text",
                parent_header="References",
                page_number=10,
                extra={
                    "content_role": "reference",
                    "section_role": "references",
                    "parent_id": "DOC-6:P00001",
                    "parent_content": "Smith J et al. Pathogen detection study. doi:10.1000/example",
                },
            ),
        ),
        Chunk(
            id="DOC-6:P00002:C01",
            content="Useful discussion sentence for the biomarker finding.",
            metadata=ChunkMetadata(
                doc_id="DOC-6",
                chunk_type="text",
                parent_header="Discussion",
                page_number=8,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-6:P00002",
                    "parent_content": "Useful discussion sentence for the biomarker finding.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="biomarker finding", doc_id="DOC-6", limit=2)

    assert len(result) == 1
    assert result[0].source == "Discussion"


def test_retrieval_service_pushes_static_exclusions_into_search_filters() -> None:
    chunks = [
        Chunk(
            id="DOC-6A:P00002:C01",
            content="Useful discussion sentence for the biomarker finding.",
            metadata=ChunkMetadata(
                doc_id="DOC-6A",
                chunk_type="text",
                parent_header="Discussion",
                page_number=8,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-6A:P00002",
                    "parent_content": "Useful discussion sentence for the biomarker finding.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    service.retrieve(query="biomarker finding", doc_id="DOC-6A", limit=1)

    assert repo.last_search_args is not None
    filters = repo.last_search_args[3]
    assert filters is not None
    assert filters.doc_id == "DOC-6A"
    assert any(item.key == "content_role" and item.values == ("reference", "front_matter") for item in filters.must_not)
    assert any(item.key == "section_role" and item.values == ("references", "front_matter", "unknown") for item in filters.must_not)
    assert any(item.key == "is_low_value" and item.value is True for item in filters.must_not)


def test_retrieval_service_excludes_tables_by_default() -> None:
    chunks = [
        Chunk(
            id="DOC-7:T00001",
            content="Source File: BAL.pdf | Table Index: 1 | Section: Results\nMarker,Value\nLDL-C,126",
            metadata=ChunkMetadata(
                doc_id="DOC-7",
                chunk_type="table",
                parent_header="Results",
                page_number=6,
                extra={"content_role": "table", "section_role": "body"},
            ),
        ),
        Chunk(
            id="DOC-7:P00001:C01",
            content="Narrative biomarker interpretation with enough context to rank.",
            metadata=ChunkMetadata(
                doc_id="DOC-7",
                chunk_type="text",
                parent_header="Results",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-7:P00001",
                    "parent_content": "Narrative biomarker interpretation with enough context to rank.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="biomarker interpretation", doc_id="DOC-7", limit=2)

    assert len(result) == 1
    assert "Narrative biomarker interpretation" in result[0].content


def test_retrieval_service_can_include_tables_when_requested() -> None:
    chunks = [
        Chunk(
            id="DOC-8:T00001",
            content="Source File: BAL.pdf | Table Index: 1 | Section: Results\nMarker,Value\nLDL-C,126\nHDL-C,52",
            metadata=ChunkMetadata(
                doc_id="DOC-8",
                chunk_type="table",
                parent_header="Results",
                page_number=6,
                extra={"content_role": "table", "section_role": "body"},
            ),
        )
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(
        repo=repo,
        embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        include_tables=True,
    )

    result = service.retrieve(query="table values", doc_id="DOC-8", limit=1)

    assert len(result) == 1
    assert "Marker,Value" in result[0].content
    assert result[0].chunk_type == "table"
    assert result[0].content_role == "table"


def test_retrieval_service_prefers_discussion_over_abstract() -> None:
    chunks = [
        Chunk(
            id="DOC-9:P00001:C01",
            content="Abstract summary of the diagnostic platform.",
            metadata=ChunkMetadata(
                doc_id="DOC-9",
                chunk_type="text",
                parent_header="Abstract",
                page_number=1,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-9:P00001",
                    "parent_content": "Abstract summary of the diagnostic platform with enough substance to be returned.",
                },
            ),
        ),
        Chunk(
            id="DOC-9:P00002:C01",
            content="Discussion evidence about clinical usefulness and limitations.",
            metadata=ChunkMetadata(
                doc_id="DOC-9",
                chunk_type="text",
                parent_header="Discussion",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-9:P00002",
                    "parent_content": "Discussion evidence about clinical usefulness and limitations with enough substance to be returned.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="clinical usefulness and limitations", doc_id="DOC-9", limit=2)

    assert result[0].source == "Discussion"
    assert len(result) == 1


def test_retrieval_service_suppresses_near_duplicate_parents() -> None:
    first_parent = (
        "PCR/ESI-MS demonstrated broad pathogen detection in BAL samples and showed promising clinical utility."
    )
    duplicate_parent = (
        "PCR/ESI-MS demonstrated broad pathogen detection in BAL samples and showed promising clinical utility in practice."
    )
    chunks = [
        Chunk(
            id="DOC-10:P00001:C01",
            content="broad pathogen detection",
            metadata=ChunkMetadata(
                doc_id="DOC-10",
                chunk_type="text",
                parent_header="Discussion",
                page_number=8,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-10:P00001",
                    "parent_content": first_parent,
                },
            ),
        ),
        Chunk(
            id="DOC-10:P00002:C01",
            content="promising clinical utility",
            metadata=ChunkMetadata(
                doc_id="DOC-10",
                chunk_type="text",
                parent_header="Discussion",
                page_number=9,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-10:P00002",
                    "parent_content": duplicate_parent,
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="clinical utility", doc_id="DOC-10", limit=2)

    assert len(result) == 1


def test_retrieval_service_returns_narrower_context_window_for_child_chunks() -> None:
    chunks = [
        Chunk(
            id="DOC-11:P00001:C02",
            content="Sentence two. Sentence three.",
            metadata=ChunkMetadata(
                doc_id="DOC-11",
                chunk_type="text",
                parent_header="Discussion",
                page_number=4,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-11:P00001",
                    "parent_content": "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five.",
                    "parent_sentences": [
                        "Sentence one.",
                        "Sentence two.",
                        "Sentence three.",
                        "Sentence four.",
                        "Sentence five.",
                    ],
                    "child_sentence_start": 1,
                    "child_sentence_end": 3,
                },
            ),
        )
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="sentence three", doc_id="DOC-11", limit=1)

    assert result[0].content == "Sentence one. Sentence two. Sentence three. Sentence four."


def test_retrieval_service_caps_repeated_section_headers() -> None:
    chunks = [
        Chunk(
            id="DOC-12:P00001:C01",
            content="Discussion evidence one with enough clinical detail to survive filtering.",
            metadata=ChunkMetadata(
                doc_id="DOC-12",
                chunk_type="text",
                parent_header="Discussion",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-12:P00001",
                    "parent_content": "Discussion evidence one with enough clinical detail to survive filtering.",
                },
            ),
        ),
        Chunk(
            id="DOC-12:P00002:C01",
            content="Discussion evidence two with different but still useful clinical detail for retrieval.",
            metadata=ChunkMetadata(
                doc_id="DOC-12",
                chunk_type="text",
                parent_header="Discussion",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-12:P00002",
                    "parent_content": "Discussion evidence two with different but still useful clinical detail for retrieval.",
                },
            ),
        ),
        Chunk(
            id="DOC-12:P00003:C01",
            content="Discussion evidence three that should be dropped by the diversity cap.",
            metadata=ChunkMetadata(
                doc_id="DOC-12",
                chunk_type="text",
                parent_header="Discussion",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-12:P00003",
                    "parent_content": "Discussion evidence three that should be dropped by the diversity cap.",
                },
            ),
        ),
        Chunk(
            id="DOC-12:P00004:C01",
            content="Abstract summary with enough substance to remain available after discussion caps.",
            metadata=ChunkMetadata(
                doc_id="DOC-12",
                chunk_type="text",
                parent_header="Abstract",
                page_number=1,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-12:P00004",
                    "parent_content": "Abstract summary with enough substance to remain available after discussion caps.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="clinical usefulness", doc_id="DOC-12", limit=4)

    assert [chunk.source for chunk in result] == ["Discussion", "Discussion"]


def test_retrieval_service_limits_per_document_for_cross_corpus_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-A:P00001:C01",
            content="Discussion evidence from document A with enough text to be returned safely.",
            metadata=ChunkMetadata(
                doc_id="DOC-A",
                chunk_type="text",
                parent_header="Discussion",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-A:P00001",
                    "parent_content": "Discussion evidence from document A with enough text to be returned safely.",
                },
            ),
        ),
        Chunk(
            id="DOC-A:P00002:C01",
            content="Results evidence from document A with enough text to be returned safely.",
            metadata=ChunkMetadata(
                doc_id="DOC-A",
                chunk_type="text",
                parent_header="Results",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-A:P00002",
                    "parent_content": "Results evidence from document A with enough text to be returned safely.",
                },
            ),
        ),
        Chunk(
            id="DOC-A:P00003:C01",
            content="Conclusion evidence from document A that should be skipped by the document cap.",
            metadata=ChunkMetadata(
                doc_id="DOC-A",
                chunk_type="text",
                parent_header="Conclusion",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-A:P00003",
                    "parent_content": "Conclusion evidence from document A that should be skipped by the document cap.",
                },
            ),
        ),
        Chunk(
            id="DOC-B:P00001:C01",
            content="Document B reports fungal DNA detection and antifungal follow-up recommendations in BAL samples.",
            metadata=ChunkMetadata(
                doc_id="DOC-B",
                chunk_type="text",
                parent_header="Discussion",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-B:P00001",
                    "parent_content": "Document B reports fungal DNA detection and antifungal follow-up recommendations in BAL samples.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="clinical utility", limit=4)

    assert [chunk.doc_id for chunk in result] == ["DOC-A", "DOC-B"]


def test_retrieval_service_prefers_results_for_performance_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-13:P00001:C01",
            content="Conclusion summary with enough detail to survive filtering.",
            metadata=ChunkMetadata(
                doc_id="DOC-13",
                chunk_type="text",
                parent_header="Conclusion",
                page_number=4,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-13:P00001",
                    "parent_content": "Conclusion summary with enough detail to survive filtering.",
                },
            ),
        ),
        Chunk(
            id="DOC-13:P00002:C01",
            content="Results evidence describing sensitivity and specificity findings in detail.",
            metadata=ChunkMetadata(
                doc_id="DOC-13",
                chunk_type="text",
                parent_header="Results",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-13:P00002",
                    "parent_content": "Results evidence describing sensitivity and specificity findings in detail.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="What sensitivity and specificity findings are reported?", doc_id="DOC-13", limit=2)

    assert [chunk.source for chunk in result] == ["Results", "Conclusion"]


def test_retrieval_service_suppresses_methods_and_introduction_tails_after_stronger_evidence() -> None:
    chunks = [
        Chunk(
            id="DOC-13B:P00001:C01",
            content="Results evidence describing diagnostic performance in detail.",
            metadata=ChunkMetadata(
                doc_id="DOC-13B",
                chunk_type="text",
                parent_header="Results",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-13B:P00001",
                    "parent_content": "Results evidence describing diagnostic performance in detail.",
                },
            ),
        ),
        Chunk(
            id="DOC-13B:P00002:C01",
            content="Discussion interpretation of the diagnostic performance findings.",
            metadata=ChunkMetadata(
                doc_id="DOC-13B",
                chunk_type="text",
                parent_header="Discussion",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-13B:P00002",
                    "parent_content": "Discussion interpretation of the diagnostic performance findings.",
                },
            ),
        ),
        Chunk(
            id="DOC-13B:P00003:C01",
            content="Methods detail that should be suppressed once stronger evidence is already selected.",
            metadata=ChunkMetadata(
                doc_id="DOC-13B",
                chunk_type="text",
                parent_header="Methods",
                page_number=3,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-13B:P00003",
                    "parent_content": "Methods detail that should be suppressed once stronger evidence is already selected.",
                },
            ),
        ),
        Chunk(
            id="DOC-13B:P00004:C01",
            content="Introduction detail that should also be suppressed after stronger evidence is selected.",
            metadata=ChunkMetadata(
                doc_id="DOC-13B",
                chunk_type="text",
                parent_header="Introduction",
                page_number=2,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-13B:P00004",
                    "parent_content": "Introduction detail that should also be suppressed after stronger evidence is selected.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="What evidence is presented for diagnostic performance and outcome interpretation?",
        doc_id="DOC-13B",
        limit=4,
    )

    assert [chunk.source for chunk in result] == ["Results", "Discussion"]


def test_retrieval_service_suppresses_results_tails_for_conclusion_usefulness_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-13C:P00001:C01",
            content="Discussion evidence describing the study's clinical usefulness in practice.",
            metadata=ChunkMetadata(
                doc_id="DOC-13C",
                chunk_type="text",
                parent_header="Discussion",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-13C:P00001",
                    "parent_content": "Discussion evidence describing the study's clinical usefulness in practice.",
                },
            ),
        ),
        Chunk(
            id="DOC-13C:P00002:C01",
            content="Second discussion chunk summarizing the conclusion-oriented interpretation.",
            metadata=ChunkMetadata(
                doc_id="DOC-13C",
                chunk_type="text",
                parent_header="Discussion",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-13C:P00002",
                    "parent_content": "Second discussion chunk summarizing the conclusion-oriented interpretation.",
                },
            ),
        ),
        Chunk(
            id="DOC-13C:P00003:C01",
            content="Results detail that should be dropped once stronger discussion evidence is already selected.",
            metadata=ChunkMetadata(
                doc_id="DOC-13C",
                chunk_type="text",
                parent_header="Results",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-13C:P00003",
                    "parent_content": "Results detail that should be dropped once stronger discussion evidence is already selected.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="What conclusions are drawn about the clinical usefulness of this assay?",
        doc_id="DOC-13C",
        limit=3,
    )

    assert [chunk.source for chunk in result] == ["Discussion", "Discussion"]


def test_retrieval_service_suppresses_results_tails_for_caveat_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-13D:P00001:C01",
            content="Discussion evidence about false negatives and why clinicians should stay cautious.",
            metadata=ChunkMetadata(
                doc_id="DOC-13D",
                chunk_type="text",
                parent_header="Discussion",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-13D:P00001",
                    "parent_content": "Discussion evidence about false negatives and why clinicians should stay cautious.",
                },
            ),
        ),
        Chunk(
            id="DOC-13D:P00002:C01",
            content="Second discussion chunk about organism coverage gaps and interpretive caveats.",
            metadata=ChunkMetadata(
                doc_id="DOC-13D",
                chunk_type="text",
                parent_header="Discussion",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-13D:P00002",
                    "parent_content": "Second discussion chunk about organism coverage gaps and interpretive caveats.",
                },
            ),
        ),
        Chunk(
            id="DOC-13D:P00003:C01",
            content="Results detail that should be dropped for a caveat-oriented query.",
            metadata=ChunkMetadata(
                doc_id="DOC-13D",
                chunk_type="text",
                parent_header="Results",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-13D:P00003",
                    "parent_content": "Results detail that should be dropped for a caveat-oriented query.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Where do the authors talk about false negatives, coverage gaps, or why I should still be cautious clinically?",
        doc_id="DOC-13D",
        limit=3,
    )

    assert [chunk.source for chunk in result] == ["Discussion", "Discussion"]


def test_retrieval_service_prefers_methods_for_optimization_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-14:P00001:C01",
            content="Conclusion summary with enough detail to survive filtering.",
            metadata=ChunkMetadata(
                doc_id="DOC-14",
                chunk_type="text",
                parent_header="Conclusion",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-14:P00001",
                    "parent_content": "Conclusion summary with enough detail to survive filtering.",
                },
            ),
        ),
        Chunk(
            id="DOC-14:P00002:C01",
            content="Methods evidence describing optimization steps before clinical validation.",
            metadata=ChunkMetadata(
                doc_id="DOC-14",
                chunk_type="text",
                parent_header="Methods",
                page_number=3,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-14:P00002",
                    "parent_content": "Methods evidence describing optimization steps before clinical validation.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="What experimental optimization steps improved pathogen detection?", doc_id="DOC-14", limit=2)

    assert [chunk.source for chunk in result] == ["Methods", "Conclusion"]


def test_retrieval_service_keeps_methods_for_methods_oriented_queries_even_after_results() -> None:
    chunks = [
        Chunk(
            id="DOC-14B:P00001:C01",
            content="Results evidence confirming the assay worked after optimization.",
            metadata=ChunkMetadata(
                doc_id="DOC-14B",
                chunk_type="text",
                parent_header="Results",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-14B:P00001",
                    "parent_content": "Results evidence confirming the assay worked after optimization.",
                },
            ),
        ),
        Chunk(
            id="DOC-14B:P00002:C01",
            content="Methods evidence describing optimization steps before clinical validation.",
            metadata=ChunkMetadata(
                doc_id="DOC-14B",
                chunk_type="text",
                parent_header="Methods",
                page_number=3,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-14B:P00002",
                    "parent_content": "Methods evidence describing optimization steps before clinical validation.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="What methods and optimization steps improved pathogen detection before validation?",
        doc_id="DOC-14B",
        limit=2,
    )

    assert [chunk.source for chunk in result] == ["Methods", "Results"]


def test_retrieval_service_prefers_methods_for_review_advantages_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-14B-ADV:P00001:C01",
            content=(
                "Metagenomic next-generation sequencing is a powerful method with high efficiency"
                " and accuracy in mixed-population pathogen detection."
            ),
            metadata=ChunkMetadata(
                doc_id="IJGM-393329-blood-culture-negative-endocarditis--a-review-of-laboratory-",
                chunk_type="text",
                parent_header="Introduction",
                page_number=2,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-14B-ADV:P00001",
                    "parent_content": (
                        "Metagenomic next-generation sequencing is a powerful method with high"
                        " efficiency and accuracy in mixed-population pathogen detection."
                    ),
                },
            ),
        ),
        Chunk(
            id="DOC-14B-ADV:P00002:C01",
            content=(
                "Compared with other methods, the mNGS technique has three main advantages."
                " Firstly, mNGS has unbiased sampling; secondly, it provides accessory genomic"
                " information; thirdly, it supports independent DNA fragment classification."
            ),
            metadata=ChunkMetadata(
                doc_id="IJGM-393329-blood-culture-negative-endocarditis--a-review-of-laboratory-",
                chunk_type="text",
                parent_header="Methods",
                page_number=4,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-14B-ADV:P00002",
                    "parent_content": (
                        "Compared with other methods, the mNGS technique has three main advantages."
                        " Firstly, mNGS has unbiased sampling; secondly, it provides accessory"
                        " genomic information; thirdly, it supports independent DNA fragment"
                        " classification."
                    ),
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="What are the three main advantages of metagenomic next-generation sequencing (mNGS) according to the review by Lin et al.",
        limit=1,
    )

    assert [chunk.source for chunk in result] == ["Methods"]


def test_retrieval_service_keeps_results_for_conclusion_queries_that_explicitly_request_performance() -> None:
    chunks = [
        Chunk(
            id="DOC-14C:P00001:C01",
            content="Discussion interpretation about overall performance.",
            metadata=ChunkMetadata(
                doc_id="DOC-14C",
                chunk_type="text",
                parent_header="Discussion",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-14C:P00001",
                    "parent_content": "Discussion interpretation about overall performance.",
                },
            ),
        ),
        Chunk(
            id="DOC-14C:P00002:C01",
            content="Results evidence with the underlying diagnostic performance details.",
            metadata=ChunkMetadata(
                doc_id="DOC-14C",
                chunk_type="text",
                parent_header="Results",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-14C:P00002",
                    "parent_content": "Results evidence with the underlying diagnostic performance details.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="What conclusions were drawn about diagnostic performance?",
        doc_id="DOC-14C",
        limit=2,
    )

    assert [chunk.source for chunk in result] == ["Results", "Discussion"]


def test_retrieval_service_prefers_discussion_for_stewardship_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-15:P00001:C01",
            content="Opening summary with enough context to survive filtering.",
            metadata=ChunkMetadata(
                doc_id="DOC-15",
                chunk_type="text",
                parent_header="Document Metadata/Abstract",
                page_number=1,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-15:P00001",
                    "parent_content": "Opening summary with enough context to survive filtering.",
                },
            ),
        ),
        Chunk(
            id="DOC-15:P00002:C01",
            content="Discussion of diagnostic stewardship and optimizing hospital blood culture use.",
            metadata=ChunkMetadata(
                doc_id="DOC-15",
                chunk_type="text",
                parent_header="Discussion",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-15:P00002",
                    "parent_content": "Discussion of diagnostic stewardship and optimizing hospital blood culture use.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="What arguments are made for optimizing blood culture use in the hospital setting?", doc_id="DOC-15", limit=2)

    assert [chunk.source for chunk in result] == ["Discussion"]


def test_retrieval_service_demotes_metadata_when_body_sections_exist() -> None:
    chunks = [
        Chunk(
            id="DOC-15B:P00001:C01",
            content="Opening summary with enough context to survive filtering.",
            metadata=ChunkMetadata(
                doc_id="DOC-15B",
                chunk_type="text",
                parent_header="Document Metadata/Abstract",
                page_number=1,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-15B:P00001",
                    "parent_content": "Opening summary with enough context to survive filtering.",
                },
            ),
        ),
        Chunk(
            id="DOC-15B:P00002:C01",
            content="Introduction to optimizing blood culture use in the hospital setting.",
            metadata=ChunkMetadata(
                doc_id="DOC-15B",
                chunk_type="text",
                parent_header="Introduction",
                page_number=2,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-15B:P00002",
                    "parent_content": "Introduction to optimizing blood culture use in the hospital setting.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="What arguments are made for optimizing blood culture use in the hospital setting?", doc_id="DOC-15B", limit=2)

    assert [chunk.source for chunk in result] == ["Introduction"]


def test_retrieval_service_suppresses_metadata_for_single_doc_queries_when_body_exists() -> None:
    chunks = [
        Chunk(
            id="DOC-15C:P00001:C01",
            content="Opening summary with enough context to survive filtering.",
            metadata=ChunkMetadata(
                doc_id="DOC-15C",
                chunk_type="text",
                parent_header="Document Metadata/Abstract",
                page_number=1,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-15C:P00001",
                    "parent_content": "Opening summary with enough context to survive filtering.",
                },
            ),
        ),
        Chunk(
            id="DOC-15C:P00002:C01",
            content="Discussion of arguments for optimizing blood culture use in the hospital setting.",
            metadata=ChunkMetadata(
                doc_id="DOC-15C",
                chunk_type="text",
                parent_header="Discussion",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-15C:P00002",
                    "parent_content": "Discussion of arguments for optimizing blood culture use in the hospital setting.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="What arguments are made for optimizing blood culture use in the hospital setting?", doc_id="DOC-15C", limit=2)

    assert [chunk.source for chunk in result] == ["Discussion"]


def test_retrieval_service_skips_cross_doc_metadata_when_same_doc_body_already_selected() -> None:
    chunks = [
        Chunk(
            id="DOC-RAPID:P00001:C01",
            content="Rapid testing discussion with enough context to remain visible in cross-document retrieval.",
            metadata=ChunkMetadata(
                doc_id="RAPID",
                chunk_type="text",
                parent_header="Discussion",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RAPID:P00001",
                    "parent_content": "Rapid testing discussion with enough context to remain visible in cross-document retrieval.",
                },
            ),
        ),
        Chunk(
            id="DOC-RAPID:P00002:C01",
            content="Opening summary for RAPID that should not be returned after the body section is already selected.",
            metadata=ChunkMetadata(
                doc_id="RAPID",
                chunk_type="text",
                parent_header="Document Metadata/Abstract",
                page_number=1,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RAPID:P00002",
                    "parent_content": "Opening summary for RAPID that should not be returned after the body section is already selected.",
                },
            ),
        ),
        Chunk(
            id="DOC-BAL:P00001:C01",
            content="Discussion of implementation implications in bronchoalveolar lavage diagnostics.",
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="text",
                parent_header="Discussion",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-BAL:P00001",
                    "parent_content": "Discussion of implementation implications in bronchoalveolar lavage diagnostics.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Across the indexed studies, which papers discuss clinical usefulness or implementation implications of rapid diagnostics?",
        limit=3,
    )

    assert [chunk.source for chunk in result] == ["Discussion", "Discussion"]
    assert set(chunk.doc_id for chunk in result) == {"RAPID", "BAL SM"}


def test_retrieval_service_prefers_doc_title_overlap_for_cross_document_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-RCT:P00001:C01",
            content="Discussion of antimicrobial timing in a randomized clinical trial.",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="text",
                parent_header="Discussion",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCT:P00001",
                    "parent_content": "Discussion of antimicrobial timing in a randomized clinical trial.",
                },
            ),
        ),
        Chunk(
            id="DOC-STEW:P00001:C01",
            content="Discussion of how diagnostic stewardship should optimize blood culture use in hospitals.",
            metadata=ChunkMetadata(
                doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
                chunk_type="text",
                parent_header="Discussion",
                page_number=4,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-STEW:P00001",
                    "parent_content": "Discussion of how diagnostic stewardship should optimize blood culture use in hospitals.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="Which documents discuss diagnostic stewardship for blood cultures?", limit=2)

    assert [chunk.doc_id for chunk in result] == [
        "fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
    ]


def test_retrieval_service_skips_zero_title_overlap_docs_once_topic_match_found() -> None:
    chunks = [
        Chunk(
            id="DOC-STEW:P00001:C01",
            content="Discussion of how diagnostic stewardship should optimize blood culture use in hospitals.",
            metadata=ChunkMetadata(
                doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
                chunk_type="text",
                parent_header="Discussion",
                page_number=4,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-STEW:P00001",
                    "parent_content": "Discussion of how diagnostic stewardship should optimize blood culture use in hospitals.",
                },
            ),
        ),
        Chunk(
            id="DOC-RCT:P00001:C01",
            content="Discussion of rapid blood culture diagnostics paired with antimicrobial stewardship support.",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="text",
                parent_header="Discussion",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCT:P00001",
                    "parent_content": "Discussion of rapid blood culture diagnostics paired with antimicrobial stewardship support.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(query="Which documents discuss diagnostic stewardship for blood cultures?", limit=2)

    assert [chunk.doc_id for chunk in result] == [
        "fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
    ]


def test_retrieval_service_limits_singular_cross_document_queries_to_top_document() -> None:
    chunks = [
        Chunk(
            id="DOC-STEW:P00001:C01",
            content="Discussion of how diagnostic stewardship should optimize blood culture use in hospitals.",
            metadata=ChunkMetadata(
                doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
                chunk_type="text",
                parent_header="Discussion",
                page_number=4,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-STEW:P00001",
                    "parent_content": "Discussion of how diagnostic stewardship should optimize blood culture use in hospitals.",
                },
            ),
        ),
        Chunk(
            id="DOC-RCT:P00001:C01",
            content="Discussion of patient outcomes in a randomized clinical trial.",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="text",
                parent_header="Discussion",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCT:P00001",
                    "parent_content": "Discussion of patient outcomes in a randomized clinical trial.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Which indexed paper focuses on optimizing blood culture utilization rather than reporting patient outcomes?",
        limit=2,
    )

    assert [chunk.doc_id for chunk in result] == [
        "fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
    ]


def test_retrieval_service_prefers_stewardship_review_over_trial_for_contrastive_review_query() -> None:
    chunks = [
        Chunk(
            id="DOC-RCT:P00001:C01",
            content="Discussion of patient outcomes in a randomized clinical trial with stewardship support.",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="text",
                parent_header="Discussion",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCT:P00001",
                    "parent_content": "Discussion of patient outcomes in a randomized clinical trial with stewardship support.",
                },
            ),
        ),
        Chunk(
            id="DOC-STEW:P00001:C01",
            content="Discussion of diagnostic stewardship review content focused on blood culture utilization in the hospital setting.",
            metadata=ChunkMetadata(
                doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
                chunk_type="text",
                parent_header="Discussion",
                page_number=4,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-STEW:P00001",
                    "parent_content": "Discussion of diagnostic stewardship review content focused on blood culture utilization in the hospital setting.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Journal club framing: which indexed paper is basically a blood-culture stewardship review instead of an interventional patient-outcomes trial?",
        limit=2,
    )

    assert [chunk.doc_id for chunk in result] == [
        "fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
    ]


def test_retrieval_service_prefers_blood_culture_process_paper_over_platform_query() -> None:
    chunks = [
        Chunk(
            id="DOC-RCT:P00001:C01",
            content="Discussion of a pathogen-detection platform in a randomized trial with clinical outcome changes.",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="text",
                parent_header="Discussion",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCT:P00001",
                    "parent_content": "Discussion of a pathogen-detection platform in a randomized trial with clinical outcome changes.",
                },
            ),
        ),
        Chunk(
            id="DOC-STEW:P00001:C01",
            content="Discussion of improving blood culture ordering and collection practices in the hospital setting.",
            metadata=ChunkMetadata(
                doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
                chunk_type="text",
                parent_header="Discussion",
                page_number=4,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-STEW:P00001",
                    "parent_content": "Discussion of improving blood culture ordering and collection practices in the hospital setting.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Adversarial wording check: which indexed paper is about improving when and how blood cultures get ordered or collected, not about a pathogen-detection platform at all?",
        limit=2,
    )

    assert [chunk.doc_id for chunk in result] == [
        "fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
    ]


def test_retrieval_service_penalizes_trial_results_for_contrastive_stewardship_review_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-RCT:P00001:C01",
            content="Results of a randomized trial focused on patient outcomes after rapid blood culture testing.",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="text",
                parent_header="RESULTS",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCT:P00001",
                    "parent_content": "Results of a randomized trial focused on patient outcomes after rapid blood culture testing.",
                },
            ),
        ),
        Chunk(
            id="DOC-STEW:P00001:C01",
            content="Introduction to a stewardship review focused on blood culture utilization and ordering practices in hospitals.",
            metadata=ChunkMetadata(
                doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
                chunk_type="text",
                parent_header="Introduction",
                page_number=4,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-STEW:P00001",
                    "parent_content": "Introduction to a stewardship review focused on blood culture utilization and ordering practices in hospitals.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Journal club framing: which indexed paper is basically a blood-culture stewardship review instead of an interventional patient-outcomes trial?",
        limit=2,
    )

    assert [chunk.doc_id for chunk in result] == [
        "fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
    ]


def test_retrieval_service_ignores_negative_clause_trial_tokens_for_contrastive_stewardship_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-RCT:P00001:C01",
            content="Discussion of interventional trial patient outcomes after rapid blood culture testing.",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="text",
                parent_header="Discussion",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCT:P00001",
                    "parent_content": "Discussion of interventional trial patient outcomes after rapid blood culture testing.",
                },
            ),
        ),
        Chunk(
            id="DOC-STEW:P00001:C01",
            content="Discussion of blood culture utilization and diagnostic stewardship in the hospital setting.",
            metadata=ChunkMetadata(
                doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
                chunk_type="text",
                parent_header="Discussion",
                page_number=4,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-STEW:P00001",
                    "parent_content": "Discussion of blood culture utilization and diagnostic stewardship in the hospital setting.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Journal club framing: which indexed paper is basically a blood-culture stewardship review instead of an interventional patient-outcomes trial?",
        limit=2,
    )

    assert [chunk.doc_id for chunk in result] == [
        "fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
    ]


def test_retrieval_service_locks_contrastive_stewardship_queries_to_best_document() -> None:
    chunks = [
        Chunk(
            id="DOC-RCT:T00001",
            content=(
                "Source File: Single site RCT.pdf | Table Index: 4 | Section: RESULTS\n"
                "Outcome,Control,Rapid Multiplex PCR,Rapid Multiplex PCR + Stewardship\n"
                "Time to first appropriate de-escalation,34,38,21"
            ),
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="table",
                parent_header="RESULTS",
                page_number=6,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "parent_content": (
                        "Source File: Single site RCT.pdf | Table Index: 4 | Section: RESULTS\n"
                        "Outcome,Control,Rapid Multiplex PCR,Rapid Multiplex PCR + Stewardship\n"
                        "Time to first appropriate de-escalation,34,38,21"
                    ),
                },
            ),
        ),
        Chunk(
            id="DOC-STEW:P00001:C01",
            content="Improving blood culture ordering practices is a core diagnostic-stewardship need in hospitals.",
            metadata=ChunkMetadata(
                doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
                chunk_type="text",
                parent_header="Introduction",
                page_number=4,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-STEW:P00001",
                    "parent_content": (
                        "Improving blood culture ordering practices is a core diagnostic-stewardship need "
                        "in hospitals."
                    ),
                },
            ),
        ),
        Chunk(
            id="DOC-STEW:P00002:C01",
            content="The review emphasizes blood culture collection and utilization rather than interventional outcomes.",
            metadata=ChunkMetadata(
                doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
                chunk_type="text",
                parent_header="Discussion",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-STEW:P00002",
                    "parent_content": (
                        "The review emphasizes blood culture collection and utilization rather than "
                        "interventional outcomes."
                    ),
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(
        repo=repo,
        embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        include_tables=True,
    )

    result = service.retrieve(
        query="Journal club framing: which indexed paper is basically a blood-culture stewardship review instead of an interventional patient-outcomes trial?",
        limit=2,
    )

    assert result
    assert [chunk.doc_id for chunk in result] == [
        "fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
    ]


def test_retrieval_service_does_not_doc_lock_plural_stewardship_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-STEW:P00001:C01",
            content="Discussion of diagnostic stewardship and blood culture utilization in hospitals.",
            metadata=ChunkMetadata(
                doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
                chunk_type="text",
                parent_header="Discussion",
                page_number=4,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-STEW:P00001",
                    "parent_content": "Discussion of diagnostic stewardship and blood culture utilization in hospitals.",
                },
            ),
        ),
        Chunk(
            id="DOC-RAPID:P00001:C01",
            content="Discussion of rapid diagnostic reporting with antimicrobial stewardship involvement.",
            metadata=ChunkMetadata(
                doc_id="RAPID",
                chunk_type="text",
                parent_header="Discussion",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RAPID:P00001",
                    "parent_content": "Discussion of rapid diagnostic reporting with antimicrobial stewardship involvement.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    assert (
        service._query_uses_contrastive_stewardship_doc_lock(
            "Which indexed papers discuss stewardship involvement alongside rapid diagnostic reporting or result communication?"
        )
        is False
    )


def test_retrieval_service_detects_contrastive_blood_culture_use_queries_without_explicit_stewardship_term() -> None:
    service = RetrievalService(repo=FakeVectorRepository([]), embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    assert (
        service._query_uses_contrastive_stewardship_doc_lock(
            "Which paper is about optimizing blood culture use rather than reporting rapid test outcomes?"
        )
        is True
    )


def test_retrieval_service_locks_blood_culture_use_query_to_stewardship_review() -> None:
    chunks = [
        Chunk(
            id="DOC-RCT:P00001:C01",
            content="Discussion of rapid test outcomes and prescribing changes in a single-site randomized trial.",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="text",
                parent_header="DISCUSSION",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCT:P00001",
                    "parent_content": "Discussion of rapid test outcomes and prescribing changes in a single-site randomized trial.",
                },
            ),
        ),
        Chunk(
            id="DOC-STEW:P00001:C01",
            content="Introduction to optimizing blood culture use, collection, and utilization in the hospital setting.",
            metadata=ChunkMetadata(
                doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
                chunk_type="text",
                parent_header="Introduction",
                page_number=4,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-STEW:P00001",
                    "parent_content": "Introduction to optimizing blood culture use, collection, and utilization in the hospital setting.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Which paper is about optimizing blood culture use rather than reporting rapid test outcomes?",
        limit=1,
    )

    assert [chunk.doc_id for chunk in result] == [
        "fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
    ]


def test_retrieval_service_detects_contrastive_turnaround_queries_against_stewardship_policy() -> None:
    service = RetrievalService(repo=FakeVectorRepository([]), embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    assert (
        service._query_uses_contrastive_single_doc_lock(
            "Which paper should I read for blood-culture turnaround improvements, not stewardship policy?"
        )
        is True
    )


def test_retrieval_service_prefers_ckd_hepcidin_review_over_assay_method_for_contrastive_single_doc_query() -> None:
    chunks = [
        Chunk(
            id="DOC-RCM:P00001:C01",
            content=(
                "Quantitative measurement of hepcidin could be a useful clinical tool for the diagnosis,"
                " monitoring and management of iron metabolism disorders. We developed a simple HPLC/MS/MS"
                " assay with inexpensive sample preparation."
            ),
            metadata=ChunkMetadata(
                doc_id="RCM publication",
                chunk_type="text",
                parent_header="5 | RESULTS AND DISCUSSION",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCM:P00001",
                    "parent_content": (
                        "Quantitative measurement of hepcidin could be a useful clinical tool for the diagnosis,"
                        " monitoring and management of iron metabolism disorders. We developed a simple"
                        " HPLC/MS/MS assay with inexpensive sample preparation."
                    ),
                    "local_file": "data/raw_pdfs/uploaded/stage1_20/RCM publication.pdf",
                    "source_file": "RCM publication.pdf",
                },
            ),
        ),
        Chunk(
            id="DOC-CKD:P00001:C01",
            content=(
                "Hepcidin is the key regulator of iron balance, and high hepcidin levels cause iron blockade"
                " and anemia in chronic disease. Elevated hepcidin appears to have a major role in the"
                " development and severity of anemia in CKD."
            ),
            metadata=ChunkMetadata(
                doc_id="hepcidin diagnostic tool",
                chunk_type="text",
                parent_header="CONCLUSIONS",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-CKD:P00001",
                    "parent_content": (
                        "Hepcidin is the key regulator of iron balance, and high hepcidin levels cause iron blockade"
                        " and anemia in chronic disease. Elevated hepcidin appears to have a major role in the"
                        " development and severity of anemia in CKD."
                    ),
                    "local_file": "data/raw_pdfs/uploaded/stage1_20/hepcidin diagnostic tool.pdf",
                    "source_file": "hepcidin diagnostic tool.pdf",
                },
            ),
        ),
        Chunk(
            id="DOC-CKD:P00002:C01",
            content=(
                "There is great interest in hepcidin assays as a diagnostic test, and targeting hepcidin as"
                " a therapeutic treatment for anemia in CKD."
            ),
            metadata=ChunkMetadata(
                doc_id="hepcidin diagnostic tool",
                chunk_type="text",
                parent_header="Introduction",
                page_number=1,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-CKD:P00002",
                    "parent_content": (
                        "There is great interest in hepcidin assays as a diagnostic test, and targeting hepcidin as"
                        " a therapeutic treatment for anemia in CKD."
                    ),
                    "local_file": "data/raw_pdfs/uploaded/stage1_20/hepcidin diagnostic tool.pdf",
                    "source_file": "hepcidin diagnostic tool.pdf",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query=(
            "Which indexed hepcidin paper argues for clinical use as a diagnostic marker and therapeutic"
            " target in CKD, rather than measuring hepcidin-25 behavior across renal-function strata?"
        ),
        limit=1,
    )

    assert [chunk.doc_id for chunk in result] == ["hepcidin diagnostic tool"]


def test_retrieval_service_treats_this_paper_queries_as_single_document_targets() -> None:
    service = RetrievalService(repo=FakeVectorRepository([]), embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    assert service._query_prefers_single_document_target(
        "What role does hepcidin play in the anemia of chronic disease according to this paper?"
    ) is True


def test_retrieval_service_prefers_doc_title_overlap_for_this_paper_hepcidin_query() -> None:
    chunks = [
        Chunk(
            id="DOC-RCM:P00001:C01",
            content=(
                "Quantitative measurement of hepcidin could be a useful clinical tool for diagnosis, monitoring,"
                " and management of iron disorders."
            ),
            metadata=ChunkMetadata(
                doc_id="RCM publication",
                chunk_type="text",
                parent_header="5 | RESULTS AND DISCUSSION",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCM:P00001",
                    "parent_content": (
                        "Quantitative measurement of hepcidin could be a useful clinical tool for diagnosis,"
                        " monitoring, and management of iron disorders."
                    ),
                    "local_file": "data/raw_pdfs/uploaded/stage1_20/RCM publication.pdf",
                    "source_file": "RCM publication.pdf",
                },
            ),
        ),
        Chunk(
            id="DOC-CKD:P00001:C01",
            content=(
                "Hepcidin is the key regulator of iron balance, and high hepcidin levels cause iron blockade"
                " and anemia in chronic disease."
            ),
            metadata=ChunkMetadata(
                doc_id="hepcidin diagnostic tool",
                chunk_type="text",
                parent_header="CONCLUSIONS",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-CKD:P00001",
                    "parent_content": (
                        "Hepcidin is the key regulator of iron balance, and high hepcidin levels cause iron blockade"
                        " and anemia in chronic disease."
                    ),
                    "local_file": "data/raw_pdfs/uploaded/stage1_20/hepcidin diagnostic tool.pdf",
                    "source_file": "hepcidin diagnostic tool.pdf",
                },
            ),
        ),
        Chunk(
            id="DOC-ANEMIA:P00001:C01",
            content=(
                "Hepcidin has a central role in anemia of chronic disease through hepcidin-ferroportin interaction,"
                " restricting iron availability and contributing to iron-restricted erythropoiesis."
            ),
            metadata=ChunkMetadata(
                doc_id="hep anemia",
                chunk_type="text",
                parent_header="Discussion",
                page_number=4,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-ANEMIA:P00001",
                    "parent_content": (
                        "Hepcidin has a central role in anemia of chronic disease through hepcidin-ferroportin interaction,"
                        " restricting iron availability and contributing to iron-restricted erythropoiesis."
                    ),
                    "local_file": "data/raw_pdfs/uploaded/stage1_20/hep anemia.pdf",
                    "source_file": "hep anemia.pdf",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="What role does hepcidin play in the anemia of chronic disease according to this paper?",
        limit=1,
    )

    assert [chunk.doc_id for chunk in result] == ["hep anemia"]


def test_retrieval_service_prefers_hep_anemia_for_anemia_of_chronic_disease_focus_query() -> None:
    chunks = [
        Chunk(
            id="DOC-DIAG:P00001:C01",
            content=(
                "Hepcidin is the key regulator of iron balance, and high hepcidin levels cause iron blockade"
                " and anemia in chronic disease. Studies show CKD patients have high hepcidin levels,"
                " likely contributing to anemia of CKD and ESA hyporesponsiveness."
            ),
            metadata=ChunkMetadata(
                doc_id="hepcidin diagnostic tool",
                chunk_type="text",
                parent_header="CONCLUSIONS",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-DIAG:P00001",
                    "parent_content": (
                        "Hepcidin is the key regulator of iron balance, and high hepcidin levels cause iron blockade"
                        " and anemia in chronic disease. Studies show CKD patients have high hepcidin levels,"
                        " likely contributing to anemia of CKD and ESA hyporesponsiveness."
                    ),
                    "local_file": "data/raw_pdfs/uploaded/stage1_20/hepcidin diagnostic tool.pdf",
                    "source_file": "hepcidin diagnostic tool.pdf",
                },
            ),
        ),
        Chunk(
            id="DOC-ANEMIA:P00002:C01",
            content=(
                "Hepcidin has a central role in anemia of chronic disease through hepcidin-ferroportin interaction,"
                " restricting iron availability and contributing to iron-restricted erythropoiesis."
            ),
            metadata=ChunkMetadata(
                doc_id="hep anemia",
                chunk_type="text",
                parent_header="Discussion",
                page_number=4,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-ANEMIA:P00002",
                    "parent_content": (
                        "Hepcidin has a central role in anemia of chronic disease through hepcidin-ferroportin interaction,"
                        " restricting iron availability and contributing to iron-restricted erythropoiesis."
                    ),
                    "local_file": "data/raw_pdfs/uploaded/stage1_20/hep anemia.pdf",
                    "source_file": "hep anemia.pdf",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="What were the main findings on hepcidin and anaemia of chronic disease in the paper focused on that topic?",
        limit=1,
    )

    assert [chunk.doc_id for chunk in result] == ["hep anemia"]


def test_retrieval_service_preserves_underscores_in_returned_doc_ids() -> None:
    chunks = [
        Chunk(
            id="DOC-CHEN:P00001:C01",
            content="An analytical workflow that coupled affinity purification and stable isotope dilution LC-MS/MS was developed to dissect IgG4 glycosylation profiles for autoimmune pancreatitis.",
            metadata=ChunkMetadata(
                doc_id="Chen_Michael_IntJMolSci_2021",
                chunk_type="text",
                parent_header="Introduction",
                page_number=1,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-CHEN:P00001",
                    "parent_content": "An analytical workflow that coupled affinity purification and stable isotope dilution LC-MS/MS was developed to dissect IgG4 glycosylation profiles for autoimmune pancreatitis.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="What IgG4 glycosylation profiling approach was used to study autoimmune pancreatitis?",
        doc_id="Chen_Michael_IntJMolSci_2021",
        limit=1,
    )

    assert [chunk.doc_id for chunk in result] == ["Chen_Michael_IntJMolSci_2021"]


def test_retrieval_service_expands_search_window_for_cross_document_limitation_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-LIMIT:P00001:C01",
            content="Single blood cultures are often treated as adequate despite lower sensitivity.",
            metadata=ChunkMetadata(
                doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
                chunk_type="text",
                parent_header="Conclusion",
                page_number=8,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-LIMIT:P00001",
                    "parent_content": "Single blood cultures are often treated as adequate despite lower sensitivity.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    service.retrieve(
        query="Contrast the reasons why single blood cultures are considered inadequate in the Fabre et al. minireview with the diagnostic limitations of urinalysis discussed by Nartey et al.",
        limit=3,
    )

    assert repo.last_search_args is not None
    assert repo.last_search_args[2] == 60


def test_retrieval_service_promotes_body_metadata_limitation_evidence_for_cross_document_limitation_queries() -> None:
    query = (
        "Contrast the reasons why single blood cultures are considered inadequate in the Fabre et al. "
        "minireview with the diagnostic limitations of urinalysis discussed by Nartey et al."
    )
    noise_chunk = Chunk(
        id="DOC-NOISE:P00001:C01",
        content="Urine culture is currently considered the gold standard for UTI diagnosis, but its main drawback is the lengthy turnaround time.",
        metadata=ChunkMetadata(
            doc_id="Culture-Free Lipidomics-Based Screening Test",
            chunk_type="text",
            parent_header="Discussion",
            page_number=7,
            extra={
                "content_role": "child",
                "section_role": "body",
                "parent_id": "DOC-NOISE:P00001",
                "parent_content": "Urine culture is currently considered the gold standard for UTI diagnosis, but its main drawback is the lengthy turnaround time.",
            },
        ),
    )
    limitation_chunk = Chunk(
        id="DOC-NARTEY:P00006:C02",
        content=(
            "Because of the time-consuming nature of urine culture, many clinical laboratories have instituted reflexive workflows. "
            "However, urine dipstick analysis for blood, nitrites, and leukocyte esterase can be prone to positive interferences "
            "leading to unnecessary urine culture investigations."
        ),
        metadata=ChunkMetadata(
            doc_id="nartey-et-al-2024-a-lipidomics-based-method-to-eliminate-negative-urine-culture-in-general-population",
            chunk_type="text",
            parent_header="Document Metadata/Abstract",
            page_number=2,
            extra={
                "content_role": "child",
                "section_role": "body",
                "parent_id": "DOC-NARTEY:P00006",
                "parent_content": (
                    "Many patients with uncomplicated UTIs present clinically as straightforward cases that may not require additional "
                    "testing beyond urinalysis. Because of the time-consuming nature of urine culture, many clinical laboratories have "
                    "instituted reflexive workflows. However, urine dipstick analysis for blood, nitrites, and leukocyte esterase can be "
                    "prone to positive interferences leading to unnecessary urine culture investigations."
                ),
            },
        ),
    )
    nartey_discussion_chunk = Chunk(
        id="DOC-NARTEY:P00020:C01",
        content="FLAT analysis identified urine samples without culturable pathogens with high agreement against urine culture.",
        metadata=ChunkMetadata(
            doc_id="nartey-et-al-2024-a-lipidomics-based-method-to-eliminate-negative-urine-culture-in-general-population",
            chunk_type="text",
            parent_header="Discussion",
            page_number=10,
            extra={
                "content_role": "child",
                "section_role": "body",
                "parent_id": "DOC-NARTEY:P00020",
                "parent_content": "FLAT analysis identified urine samples without culturable pathogens with high agreement against urine culture.",
            },
        ),
    )
    fabre_chunk = Chunk(
        id="DOC-FABRE:P00001:C01",
        content="Clinicians often justify solitary blood cultures based on comfort and a mistaken belief that one set is enough.",
        metadata=ChunkMetadata(
            doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
            chunk_type="text",
            parent_header="Conclusion",
            page_number=9,
            extra={
                "content_role": "child",
                "section_role": "body",
                "parent_id": "DOC-FABRE:P00001",
                "parent_content": "Clinicians often justify solitary blood cultures based on comfort and a mistaken belief that one set is enough.",
            },
        ),
    )
    service = RetrievalService(repo=FakeVectorRepository([]), embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    ranked = service._rank_chunks(query=query, chunks=[noise_chunk, limitation_chunk, nartey_discussion_chunk, fabre_chunk])

    assert ranked[0].metadata.doc_id == limitation_chunk.metadata.doc_id


def test_retrieval_service_locks_turnaround_query_to_rapid_paper_over_stewardship_review() -> None:
    chunks = [
        Chunk(
            id="DOC-STEW:P00001:C01",
            content="Introduction to diagnostic stewardship policy and optimizing blood culture use in hospitals.",
            metadata=ChunkMetadata(
                doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
                chunk_type="text",
                parent_header="Introduction",
                page_number=4,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-STEW:P00001",
                    "parent_content": "Introduction to diagnostic stewardship policy and optimizing blood culture use in hospitals.",
                },
            ),
        ),
        Chunk(
            id="DOC-RAPID:P00001:C01",
            content="Discussion of faster organism ID and AST turnaround from a rapid blood-culture diagnostic workflow.",
            metadata=ChunkMetadata(
                doc_id="RAPID",
                chunk_type="text",
                parent_header="DISCUSSION",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RAPID:P00001",
                    "parent_content": "Discussion of faster organism ID and AST turnaround from a rapid blood-culture diagnostic workflow.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Which paper should I read for blood-culture turnaround improvements, not stewardship policy?",
        limit=1,
    )

    assert [chunk.doc_id for chunk in result] == ["RAPID"]


def test_retrieval_service_prefers_tables_for_tabular_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-16:T00001",
            content="Sensitivity,Specificity\n0.88,0.91",
            metadata=ChunkMetadata(
                doc_id="DOC-16",
                chunk_type="table",
                parent_header="Results",
                page_number=7,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "parent_content": "Sensitivity,Specificity\n0.88,0.91",
                },
            ),
        ),
        Chunk(
            id="DOC-16:P00001:C01",
                content="Narrative discussion explaining why the assay panel improved triage decisions in practice.",
            metadata=ChunkMetadata(
                doc_id="DOC-16",
                chunk_type="text",
                parent_header="Discussion",
                page_number=8,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-16:P00001",
                    "parent_content": "Narrative discussion explaining why the assay panel improved triage decisions in practice.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(
        repo=repo,
        embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        include_tables=True,
    )

    result = service.retrieve(query="Which papers contain tabular sensitivity or specificity findings?", doc_id="DOC-16", limit=2)

    assert [chunk.chunk_type for chunk in result] == ["table"]


def test_retrieval_service_prefers_turnaround_gain_chunk_over_limitations_for_doc_navigation_query() -> None:
    chunks = [
        Chunk(
            id="DOC-RAPID:P00001:C01",
            content="The rapid testing platform did not have targets to identify one-third of the gram-negative bacilli and did not test all clinically important antibiotics.",
            metadata=ChunkMetadata(
                doc_id="RAPID",
                chunk_type="text",
                parent_header="DISCUSSION",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RAPID:P00001",
                    "parent_content": "The rapid testing platform did not have targets to identify one-third of the gram-negative bacilli and did not test all clinically important antibiotics.",
                },
            ),
        ),
        Chunk(
            id="DOC-RAPID:P00002:C01",
            content="Rapid organism ID and phenotypic AST led to significantly faster antibiotic modifications and shorter turnaround than standard of care.",
            metadata=ChunkMetadata(
                doc_id="RAPID",
                chunk_type="text",
                parent_header="DISCUSSION",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RAPID:P00002",
                    "parent_content": "Rapid organism ID and phenotypic AST led to significantly faster antibiotic modifications and shorter turnaround than standard of care.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Which paper should I read for blood-culture turnaround improvements, not stewardship policy?",
        limit=1,
    )

    assert [(chunk.doc_id, chunk.source) for chunk in result] == [("RAPID", "DISCUSSION")]
    assert "faster antibiotic modifications" in result[0].content.lower()


def test_retrieval_service_treats_confirmation_rate_queries_as_table_friendly() -> None:
    chunks = [
        Chunk(
            id="BAL-SM:T00001",
            content=(
                "Source File: BAL SM.pdf | Table Index: 1 | Section: Results\n"
                "Primary and potential pathogens.,IRIDICA-positive.Total,IRIDICA-positive.Confirmed by culture and/orPCR\n"
                "Staphylococcus aureus,33,27 a"
            ),
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="table",
                parent_header="Results",
                page_number=6,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "parent_content": (
                        "Primary and potential pathogens.,IRIDICA-positive.Total,"
                        "IRIDICA-positive.Confirmed by culture and/orPCR\n"
                        "Staphylococcus aureus,33,27 a"
                    ),
                },
            ),
        ),
        Chunk(
            id="BAL-SM:P00001:C01",
            content=(
                "PCR/ESI-MS demonstrated an overall higher sensitivity compared to routine culture-based"
                " microbiological diagnostics, with identification of microorganisms in 15/17 (88%)"
                " culture-negative samples."
            ),
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="text",
                parent_header="Discussion",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "BAL-SM:P00001",
                    "parent_content": (
                        "PCR/ESI-MS demonstrated an overall higher sensitivity compared to routine"
                        " culture-based microbiological diagnostics, with identification of microorganisms"
                        " in 15/17 (88%) culture-negative samples."
                    ),
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="What confirmation rate was achieved for Staphylococcus aureus by culture or PCR in the IRIDICA study?",
        limit=2,
    )

    assert repo.last_search_args is not None
    filters = repo.last_search_args[3]
    assert filters is not None
    assert not any(item.key == "content_role" and item.value == "table" for item in filters.must_not)
    assert result[0].chunk_type == "table"
    assert result[0].doc_id == "BAL SM"


def test_retrieval_service_prefers_results_for_confirmation_rate_queries_without_tables() -> None:
    chunks = [
        Chunk(
            id="BAL-SM:P00002:C01",
            content=(
                "All 15 samples with significant growth of S. aureus yielded levels above this threshold."
                " In 12/18 (67%) culture-negative samples, detection of S. aureus by PCR/ESI-MS was"
                " confirmed by species-specific PCR. Together, the presence of S. aureus was verified by"
                " culture and/or PCR in 27/33 (82%) PCR/ESI-MS-positive samples."
            ),
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="text",
                parent_header="Results",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "BAL-SM:P00002",
                    "parent_content": (
                        "All 15 samples with significant growth of S. aureus yielded levels above this threshold."
                        " In 12/18 (67%) culture-negative samples, detection of S. aureus by PCR/ESI-MS was"
                        " confirmed by species-specific PCR. Together, the presence of S. aureus was verified by"
                        " culture and/or PCR in 27/33 (82%) PCR/ESI-MS-positive samples."
                    ),
                },
            ),
        ),
        Chunk(
            id="BAL-SM:P00003:C01",
            content=(
                "PCR/ESI-MS is one of the most recent methods developed for detection and identification"
                " of microorganisms and selected resistance markers directly from clinical specimens."
                " PCR/ESI-MS could identify 60 different microorganisms in 121 BAL samples and"
                " demonstrated an overall higher sensitivity compared to routine culture-based diagnostics,"
                " with identification of microorganisms in 15/17 (88%) culture-negative samples."
            ),
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="text",
                parent_header="Discussion",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "BAL-SM:P00003",
                    "parent_content": (
                        "PCR/ESI-MS could identify 60 different microorganisms in 121 BAL samples and"
                        " demonstrated an overall higher sensitivity compared to routine culture-based diagnostics,"
                        " with identification of microorganisms in 15/17 (88%) culture-negative samples."
                    ),
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="What confirmation rate was achieved for Staphylococcus aureus by culture or PCR in the IRIDICA study?",
        doc_id="BAL SM",
        limit=2,
    )

    assert [chunk.source for chunk in result] == ["Results", "Discussion"]


def test_retrieval_service_prefers_bal_overall_detection_summary_for_overall_comparison_queries() -> None:
    chunks = [
        Chunk(
            id="BAL-SM:P00010:C01",
            content=(
                "S. pneumoniae was detected in 17 BAL samples by PCR/ESI-MS. The presence of"
                " S. pneumoniae was however confirmed in only 6/17 (35%) samples."
            ),
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="text",
                parent_header="Results",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "BAL-SM:P00010",
                    "parent_content": (
                        "S. pneumoniae was detected in 17 BAL samples by PCR/ESI-MS. The presence of"
                        " S. pneumoniae was however confirmed in only 6/17 (35%) samples."
                    ),
                },
            ),
        ),
        Chunk(
            id="BAL-SM:P00011:C01",
            content=(
                "PCR/ESI-MS could identify 60 different microorganisms in 121 BAL samples and"
                " demonstrated an overall higher sensitivity compared to routine culture-based"
                " microbiological diagnostics, with identification of microorganisms in 15/17 (88%)"
                " culture-negative samples."
            ),
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="text",
                parent_header="Discussion",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "BAL-SM:P00011",
                    "parent_content": (
                        "PCR/ESI-MS could identify 60 different microorganisms in 121 BAL samples and"
                        " demonstrated an overall higher sensitivity compared to routine culture-based"
                        " microbiological diagnostics, with identification of microorganisms in 15/17 (88%)"
                        " culture-negative samples."
                    ),
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="What did the BAL IRIDICA study find for overall detection versus routine culture?",
        doc_id="BAL SM",
        limit=2,
    )

    assert [chunk.source for chunk in result] == ["Discussion", "Results"]


def test_retrieval_service_prefers_bal_overall_detection_summary_for_detection_rate_wording() -> None:
    chunks = [
        Chunk(
            id="BAL-SM:P00012:C01",
            content=(
                "Detection of H. influenzae by PCR/ESI-MS was confirmed by culture in 16/20 (80%)"
                " BAL samples. Semi-quantitative PCR/ESI-MS levels were low in three culture-negative samples."
            ),
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="text",
                parent_header="Results",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "BAL-SM:P00012",
                    "parent_content": (
                        "Detection of H. influenzae by PCR/ESI-MS was confirmed by culture in 16/20 (80%)"
                        " BAL samples. Semi-quantitative PCR/ESI-MS levels were low in three culture-negative samples."
                    ),
                },
            ),
        ),
        Chunk(
            id="BAL-SM:P00013:C01",
            content=(
                "PCR/ESI-MS could identify 60 different microorganisms in 121 BAL samples and"
                " demonstrated an overall higher sensitivity compared to routine culture-based"
                " microbiological diagnostics, with identification of microorganisms in 15/17 (88%)"
                " culture-negative samples."
            ),
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="text",
                parent_header="Discussion",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "BAL-SM:P00013",
                    "parent_content": (
                        "PCR/ESI-MS could identify 60 different microorganisms in 121 BAL samples and"
                        " demonstrated an overall higher sensitivity compared to routine culture-based"
                        " microbiological diagnostics, with identification of microorganisms in 15/17 (88%)"
                        " culture-negative samples."
                    ),
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="What was the overall detection rate of PCR/ESI-MS compared to routine culture in BAL samples?",
        limit=1,
    )

    assert [(chunk.doc_id, chunk.source) for chunk in result] == [("BAL SM", "Discussion")]
    assert "overall higher sensitivity" in result[0].content.lower()


def test_retrieval_service_prefers_bal_abstract_for_explanatory_workflow_comparisons() -> None:
    chunks = [
        Chunk(
            id="BAL-SM:P00020:C01",
            content=(
                "The clinical demand on rapid microbiological diagnostic is constantly increasing."
                " PCR coupled to electrospray ionization-mass spectrometry rapidly provides sequence"
                " information from generated amplicons and enables direct species identification."
            ),
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="text",
                parent_header="Abstract",
                page_number=1,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "BAL-SM:P00020",
                    "parent_content": (
                        "The clinical demand on rapid microbiological diagnostic is constantly"
                        " increasing. PCR coupled to electrospray ionization-mass spectrometry"
                        " rapidly provides sequence information from generated amplicons and enables"
                        " direct species identification."
                    ),
                },
            ),
        ),
        Chunk(
            id="BAL-SM:P00021:C01",
            content=(
                "When the analytical performance of PCR/ESI-MS in detection of primary or potential"
                " pathogens was analyzed, the method was not inferior to culture-based methods."
            ),
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="text",
                parent_header="Discussion",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "BAL-SM:P00021",
                    "parent_content": (
                        "When the analytical performance of PCR/ESI-MS in detection of primary or"
                        " potential pathogens was analyzed, the method was not inferior to"
                        " culture-based methods."
                    ),
                },
            ),
        ),
        Chunk(
            id="FLAT:P00010:C01",
            content=(
                "The FLAT workflow directly detects microbial membrane lipids from urine without"
                " ex vivo growth and can return results within an hour."
            ),
            metadata=ChunkMetadata(
                doc_id="Culture-Free Lipidomics-Based Screening Test",
                chunk_type="text",
                parent_header="Introduction",
                page_number=2,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "FLAT:P00010",
                    "parent_content": (
                        "The FLAT workflow directly detects microbial membrane lipids from urine"
                        " without ex vivo growth and can return results within an hour."
                    ),
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Based on the provided papers, how do PCR/ESI-MS methods and FLAT lipidomics workflows differ in their approach to bypassing traditional culture times?",
        limit=2,
    )

    assert ("BAL SM", "Abstract") in [(chunk.doc_id, chunk.source) for chunk in result]
    assert ("BAL SM", "Discussion") not in [(chunk.doc_id, chunk.source) for chunk in result]
    assert ("Culture-Free Lipidomics-Based Screening Test", "Introduction") in [
        (chunk.doc_id, chunk.source) for chunk in result
    ]


def test_retrieval_service_treats_bal_iridica_study_query_as_single_document_target_without_doc_filter() -> None:
    chunks = [
        Chunk(
            id="BAL-SM:P00011:C01",
            content=(
                "PCR/ESI-MS could identify 60 different microorganisms in 121 BAL samples and"
                " demonstrated an overall higher sensitivity compared to routine culture-based"
                " microbiological diagnostics, with identification of microorganisms in 15/17 (88%)"
                " culture-negative samples."
            ),
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="text",
                parent_header="Discussion",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "BAL-SM:P00011",
                    "parent_content": (
                        "PCR/ESI-MS could identify 60 different microorganisms in 121 BAL samples and"
                        " demonstrated an overall higher sensitivity compared to routine culture-based"
                        " microbiological diagnostics, with identification of microorganisms in 15/17 (88%)"
                        " culture-negative samples."
                    ),
                },
            ),
        ),
        Chunk(
            id="FLAT:P00001:C01",
            content="Overall, the FLAT assay had a sensitivity of 70% and specificity of 99%.",
            metadata=ChunkMetadata(
                doc_id="nartey-et-al-2024-a-lipidomics-based-method-to-eliminate-negative-urine-culture-in-general-population",
                chunk_type="text",
                parent_header="RESULTS",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "FLAT:P00001",
                    "parent_content": "Overall, the FLAT assay had a sensitivity of 70% and specificity of 99%.",
                },
            ),
        ),
        Chunk(
            id="STEW:P00001:C01",
            content="Blood culture utilization improvement efforts should define when cultures are appropriate.",
            metadata=ChunkMetadata(
                doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
                chunk_type="text",
                parent_header="Discussion",
                page_number=9,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "STEW:P00001",
                    "parent_content": "Blood culture utilization improvement efforts should define when cultures are appropriate.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="What did the BAL IRIDICA study find for overall detection versus routine culture?",
        limit=3,
    )

    assert [chunk.doc_id for chunk in result] == ["BAL SM"]


def test_retrieval_service_prefers_resistance_marker_presence_evidence() -> None:
    chunks = [
        Chunk(
            id="BAL-SM:P00012:C01",
            content=(
                "Detection of H. influenzae by PCR/ESI-MS was confirmed by culture in 16/20 (80%)"
                " BAL samples. Semi-quantitative PCR/ESI-MS levels were low in three culture-negative samples."
            ),
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="text",
                parent_header="Results",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "BAL-SM:P00012",
                    "parent_content": (
                        "Detection of H. influenzae by PCR/ESI-MS was confirmed by culture in 16/20 (80%)"
                        " BAL samples. Semi-quantitative PCR/ESI-MS levels were low in three culture-negative samples."
                    ),
                },
            ),
        ),
        Chunk(
            id="BAL-SM:P00013:C01",
            content=(
                "The IRIDICA BAC LRT Assay panel also includes selected major resistance determinants,"
                " i.e. mecA, vanA, vanB, and blaKPC. The only gene detected in the samples investigated"
                " here was mecA."
            ),
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="text",
                parent_header="Discussion",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "BAL-SM:P00013",
                    "parent_content": (
                        "The IRIDICA BAC LRT Assay panel also includes selected major resistance determinants,"
                        " i.e. mecA, vanA, vanB, and blaKPC. The only gene detected in the samples investigated"
                        " here was mecA."
                    ),
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="What resistance markers were actually found in the BAL samples?",
        doc_id="BAL SM",
        limit=2,
    )

    assert [chunk.source for chunk in result] == ["Discussion", "Results"]
    assert "mecA" in result[0].content


def test_retrieval_service_requires_table_chunks_for_explicit_table_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-17:T00001",
            content="Source File: DOC-17.pdf | Table Index: 1 | Section: Results\nMarker,Count\nE. coli,12",
            metadata=ChunkMetadata(
                doc_id="DOC-17",
                chunk_type="table",
                parent_header="Results",
                page_number=7,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "parent_content": "Source File: DOC-17.pdf | Table Index: 1 | Section: Results\nMarker,Count\nE. coli,12",
                },
            ),
        ),
        Chunk(
            id="DOC-17:P00001:C01",
            content="Narrative discussion about organism counts that should be excluded for an explicit table query.",
            metadata=ChunkMetadata(
                doc_id="DOC-17",
                chunk_type="text",
                parent_header="Discussion",
                page_number=8,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-17:P00001",
                    "parent_content": "Narrative discussion about organism counts that should be excluded for an explicit table query.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(
        repo=repo,
        embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        include_tables=True,
    )

    result = service.retrieve(query="Which paper contains a results table comparing identified organisms?", limit=2)

    assert [chunk.chunk_type for chunk in result] == ["table"]


def test_retrieval_service_adds_should_filters_for_explicit_table_references() -> None:
    chunks = [
        Chunk(
            id="DOC-17A:P00001:C01",
            content="Table 3 compares discrepant respiratory pathogen results.",
            metadata=ChunkMetadata(
                doc_id="DOC-17A",
                chunk_type="text",
                parent_header="Results",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-17A:P00001",
                    "parent_content": "Table 3 compares discrepant respiratory pathogen results.",
                    "referenced_table_indices": [3],
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(
        repo=repo,
        embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        include_tables=True,
    )

    service.retrieve(
        query="Which indexed paper contains Table 3 for discrepant respiratory pathogen results?",
        limit=1,
    )

    assert repo.last_search_args is not None
    filters = repo.last_search_args[3]
    assert filters is not None
    assert any(item.key == "content_role" and item.value == "table" for item in filters.should)
    assert any(item.key == "referenced_table_indices" and item.values == (3,) for item in filters.should)


def test_retrieval_service_requires_metric_evidence_for_tabular_metric_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-18:T00001",
            content="Source File: DOC-18.pdf | Table Index: 1 | Section: Results\nSensitivity,Specificity\n0.88,0.91",
            metadata=ChunkMetadata(
                doc_id="DOC-18",
                chunk_type="table",
                parent_header="Results",
                page_number=6,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "parent_content": "Source File: DOC-18.pdf | Table Index: 1 | Section: Results\nSensitivity,Specificity\n0.88,0.91",
                },
            ),
        ),
        Chunk(
            id="DOC-19:T00001",
            content="Source File: DOC-19.pdf | Table Index: 2 | Section: Results\nPhenotypic Susceptibility Testing,No.\nMicrococcus luteus mismatch,4",
            metadata=ChunkMetadata(
                doc_id="DOC-19",
                chunk_type="table",
                parent_header="Results",
                page_number=7,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "parent_content": "Source File: DOC-19.pdf | Table Index: 2 | Section: Results\nPhenotypic Susceptibility Testing,No.\nMicrococcus luteus mismatch,4",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(
        repo=repo,
        embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        include_tables=True,
    )

    result = service.retrieve(query="Which papers contain tabular sensitivity or specificity findings?", limit=2)

    assert [chunk.doc_id for chunk in result] == ["DOC-18"]


def test_retrieval_service_excludes_non_quantitative_sensitivity_tables_for_metric_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-STEW:T00001",
            content="Source File: stewardship.pdf | Table Index: 1 | Section: Discussion\nEducate staff on blood culture sensitivity factors,Provide feedback on collection practices",
            metadata=ChunkMetadata(
                doc_id="DOC-STEW",
                chunk_type="table",
                parent_header="Discussion",
                page_number=3,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "parent_content": "Source File: stewardship.pdf | Table Index: 1 | Section: Discussion\nEducate staff on blood culture sensitivity factors,Provide feedback on collection practices",
                },
            ),
        ),
        Chunk(
            id="DOC-METRIC:T00001",
            content="Source File: metric.pdf | Table Index: 1 | Section: Results\nSensitivity,Specificity\n70%,99%",
            metadata=ChunkMetadata(
                doc_id="DOC-METRIC",
                chunk_type="table",
                parent_header="Results",
                page_number=3,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "parent_content": "Source File: metric.pdf | Table Index: 1 | Section: Results\nSensitivity,Specificity\n70%,99%",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(
        repo=repo,
        embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        include_tables=True,
    )

    result = service.retrieve(query="Which papers contain tabular sensitivity or specificity findings?", limit=2)

    assert [chunk.doc_id for chunk in result] == ["DOC-METRIC"]


def test_retrieval_service_prefers_metric_metadata_when_available() -> None:
    chunks = [
        Chunk(
            id="DOC-META:T00001",
            content="Source File: metric.pdf | Table Index: 1 | Section: Results\nSensitivity,Specificity\n70%,99%",
            metadata=ChunkMetadata(
                doc_id="DOC-META",
                chunk_type="table",
                parent_header="Results",
                page_number=3,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "contains_metric_values": True,
                    "table_semantics": ["metric", "comparison"],
                },
            ),
        ),
        Chunk(
            id="DOC-NONMETA:T00001",
            content="Source File: nonmetric.pdf | Table Index: 1 | Section: Discussion\nWorkflow,Recommendation\nA,B",
            metadata=ChunkMetadata(
                doc_id="DOC-NONMETA",
                chunk_type="table",
                parent_header="Discussion",
                page_number=3,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "contains_metric_values": False,
                    "table_semantics": ["comparison"],
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(
        repo=repo,
        embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        include_tables=True,
    )

    result = service.retrieve(query="Which papers contain tabular sensitivity or specificity findings?", limit=2)

    assert [chunk.doc_id for chunk in result] == ["DOC-META"]


def test_retrieval_service_limits_which_indexed_study_queries_to_top_document() -> None:
    chunks = [
        Chunk(
            id="DOC-RAPID:P00001:C01",
            content="Rapid testing enabled antibiotic modifications to occur nearly a day faster than standard of care.",
            metadata=ChunkMetadata(
                doc_id="RAPID",
                chunk_type="text",
                parent_header="Discussion",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RAPID:P00001",
                    "parent_content": "Rapid testing enabled antibiotic modifications to occur nearly a day faster than standard of care.",
                },
            ),
        ),
        Chunk(
            id="DOC-RCT:P00001:C01",
            content="Rapid test reporting with stewardship support reduced unnecessary antibiotic use in a separate trial.",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="text",
                parent_header="Discussion",
                page_number=8,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCT:P00001",
                    "parent_content": "Rapid test reporting with stewardship support reduced unnecessary antibiotic use in a separate trial.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Which indexed study reports rapid antibiotic deescalation timing benefits specifically from the RAPID platform?",
        limit=2,
    )

    assert [chunk.doc_id for chunk in result] == ["RAPID"]


def test_retrieval_service_limits_randomized_blood_culture_study_queries_to_top_document() -> None:
    chunks = [
        Chunk(
            id="DOC-RCT:T00001",
            content="Outcome,Control,Rapid Multiplex PCR,Rapid Multiplex PCR + Stewardship\nTime to first de-escalation,...",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="table",
                parent_header="RESULTS",
                page_number=6,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "parent_content": "Outcome,Control,Rapid Multiplex PCR,Rapid Multiplex PCR + Stewardship\nTime to first de-escalation,...",
                },
            ),
        ),
        Chunk(
            id="DOC-RAPID:T00001",
            content="Characteristic,Standard of Care,RAPID\nBlood culture organisms,...",
            metadata=ChunkMetadata(
                doc_id="RAPID",
                chunk_type="table",
                parent_header="RESULTS",
                page_number=5,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "parent_content": "Characteristic,Standard of Care,RAPID\nBlood culture organisms,...",
                },
            ),
        ),
        Chunk(
            id="DOC-STEW:P00001:C01",
            content="Summary of diagnostic stewardship and blood culture use in hospitals.",
            metadata=ChunkMetadata(
                doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
                chunk_type="text",
                parent_header="SUMMARY",
                page_number=1,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-STEW:P00001",
                    "parent_content": "Summary of diagnostic stewardship and blood culture use in hospitals.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(
        repo=repo,
        embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        include_tables=True,
    )

    result = service.retrieve(
        query="Which indexed randomized blood culture study compares plain rapid reporting versus rapid reporting with stewardship on top of it?",
        limit=4,
    )

    assert [chunk.doc_id for chunk in result] == ["Single site RCT"]


def test_retrieval_service_prefers_results_for_comparison_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-RAPID:P00001:C01",
            content="Discussion of a rapid diagnostic platform with clinically useful results.",
            metadata=ChunkMetadata(
                doc_id="RAPID",
                chunk_type="text",
                parent_header="Discussion",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RAPID:P00001",
                    "parent_content": "Discussion of a rapid diagnostic platform with clinically useful results.",
                },
            ),
        ),
        Chunk(
            id="DOC-RCT:T00001",
            content="Outcome,Control,Rapid Multiplex PCR,Rapid Multiplex PCR + Stewardship\nClinical outcome,...",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="table",
                parent_header="Results",
                page_number=6,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "parent_content": "Outcome,Control,Rapid Multiplex PCR,Rapid Multiplex PCR + Stewardship\nClinical outcome,...",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(
        repo=repo,
        embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        include_tables=True,
    )

    result = service.retrieve(
        query="Which indexed randomized trial compares rapid test reporting with and without real-time antimicrobial stewardship oversight?",
        limit=2,
    )

    assert [chunk.doc_id for chunk in result] == ["Single site RCT"]


def test_retrieval_service_prefers_outcome_tables_over_demographics_for_mortality_comparisons() -> None:
    chunks = [
        Chunk(
            id="DOC-RAPID:T00010",
            content=(
                "Source File: RAPID.pdf | Table Index: 1 | Section: RESULTS\n"
                "Characteristic,Standard of Care,RAPID\n"
                "Demographics,..."
            ),
            metadata=ChunkMetadata(
                doc_id="RAPID",
                chunk_type="table",
                parent_header="RESULTS",
                page_number=5,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "parent_content": (
                        "Source File: RAPID.pdf | Table Index: 1 | Section: RESULTS\n"
                        "Characteristic,Standard of Care,RAPID\n"
                        "Demographics,..."
                    ),
                },
            ),
        ),
        Chunk(
            id="DOC-RAPID:T00011",
            content=(
                "Source File: RAPID.pdf | Table Index: 4 | Section: DISCUSSION\n"
                "Outcome,Standard of Care,RAPID,P Value\n"
                "30 day mortality,12,11,0.85"
            ),
            metadata=ChunkMetadata(
                doc_id="RAPID",
                chunk_type="table",
                parent_header="DISCUSSION",
                page_number=8,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "parent_content": (
                        "Source File: RAPID.pdf | Table Index: 4 | Section: DISCUSSION\n"
                        "Outcome,Standard of Care,RAPID,P Value\n"
                        "30 day mortality,12,11,0.85"
                    ),
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(
        repo=repo,
        embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        include_tables=True,
    )

    result = service.retrieve(
        query="Compare the clinical impact of the rapid testing platforms evaluated in the two Banerjee et al. trials (2015 vs. 2021). Did either study find a significant difference in mortality rates between the rapid testing and standard-of-care groups?",
        limit=1,
    )

    assert [(chunk.doc_id, chunk.source) for chunk in result] == [("RAPID", "DISCUSSION")]


def test_retrieval_service_limits_where_in_indexed_corpus_queries_to_top_document() -> None:
    chunks = [
        Chunk(
            id="DOC-RAPID:P00001:C01",
            content="Rapid testing enabled gram-negative antibiotic modifications to occur a median of 24.8 hours faster than SOC.",
            metadata=ChunkMetadata(
                doc_id="RAPID",
                chunk_type="text",
                parent_header="DISCUSSION",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RAPID:P00001",
                    "parent_content": "Rapid testing enabled gram-negative antibiotic modifications to occur a median of 24.8 hours faster than SOC.",
                },
            ),
        ),
        Chunk(
            id="DOC-RCT:P00001:C01",
            content="Time to first appropriate escalation was faster in the stewardship arm than control.",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="text",
                parent_header="RESULTS",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCT:P00001",
                    "parent_content": "Time to first appropriate escalation was faster in the stewardship arm than control.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Where in the indexed corpus do they report a roughly 24-hour antibiotic-modification advantage from a rapid bacteremia workflow?",
        limit=2,
    )

    assert [chunk.doc_id for chunk in result] == ["RAPID"]


def test_retrieval_service_prefers_rapid_antibiotic_modification_timing_evidence_over_bal_intro_noise() -> None:
    chunks = [
        Chunk(
            id="DOC-BAL:P00001:C01",
            content="PCR/ESI-MS overcomes these limitations by combining multiple broad range PCR reactions with electrospray ionization-mass spectrometry.",
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="text",
                parent_header="Introduction",
                page_number=2,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-BAL:P00001",
                    "parent_content": "PCR/ESI-MS overcomes these limitations by combining multiple broad range PCR reactions with electrospray ionization-mass spectrometry.",
                },
            ),
        ),
        Chunk(
            id="DOC-RAPID:P00001:C01",
            content="Methods. Patients with positive blood cultures with Gram stains showing GNB were randomized to SOC testing with antimicrobial stewardship.",
            metadata=ChunkMetadata(
                doc_id="RAPID",
                chunk_type="text",
                parent_header="Structured Abstract",
                page_number=1,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RAPID:P00001",
                    "parent_content": "Methods. Patients with positive blood cultures with Gram stains showing GNB were randomized to SOC testing with antimicrobial stewardship.",
                },
            ),
        ),
        Chunk(
            id="DOC-RAPID:P00002:C01",
            content="Rapid testing enabled gram-negative antibiotic modifications to occur a median of 24.8 hours faster than SOC, which is a clinically significant improvement.",
            metadata=ChunkMetadata(
                doc_id="RAPID",
                chunk_type="text",
                parent_header="DISCUSSION",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RAPID:P00002",
                    "parent_content": "Rapid testing enabled gram-negative antibiotic modifications to occur a median of 24.8 hours faster than SOC, which is a clinically significant improvement.",
                },
            ),
        ),
        Chunk(
            id="DOC-RCT:P00001:C01",
            content="Within the first 4 days after enrollment, vancomycin duration was not different between groups, though escalation timing was faster in the stewardship arm.",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="text",
                parent_header="RESULTS",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCT:P00001",
                    "parent_content": "Within the first 4 days after enrollment, vancomycin duration was not different between groups, though escalation timing was faster in the stewardship arm.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Where in the indexed corpus do they report a roughly 24-hour antibiotic-modification advantage from a rapid bacteremia workflow?",
        limit=1,
    )

    assert [(chunk.doc_id, chunk.source) for chunk in result] == [("RAPID", "DISCUSSION")]


def test_retrieval_service_includes_stewardship_review_for_decision_making_vs_clinical_outcomes_query() -> None:
    chunks = [
        Chunk(
            id="DOC-RAPID:P00001:C01",
            content="Rapid testing enabled gram-negative antibiotic modifications to occur a median of 24.8 hours faster than SOC, with more stewardship recommendations.",
            metadata=ChunkMetadata(
                doc_id="RAPID",
                chunk_type="text",
                parent_header="DISCUSSION",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RAPID:P00001",
                    "parent_content": "Rapid testing enabled gram-negative antibiotic modifications to occur a median of 24.8 hours faster than SOC, with more stewardship recommendations.",
                },
            ),
        ),
        Chunk(
            id="DOC-RCT:P00001:C01",
            content="Time to de-escalation and escalation improved, but the study was not powered to detect mortality or cost outcomes.",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="text",
                parent_header="RESULTS",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCT:P00001",
                    "parent_content": "Time to de-escalation and escalation improved, but the study was not powered to detect mortality or cost outcomes.",
                },
            ),
        ),
        Chunk(
            id="DOC-FABRE:P00001:C01",
            content="Diagnostic stewardship for blood culture utilization focuses on improving antimicrobial decision making and avoiding unnecessary downstream testing.",
            metadata=ChunkMetadata(
                doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
                chunk_type="text",
                parent_header="Discussion",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-FABRE:P00001",
                    "parent_content": "Diagnostic stewardship for blood culture utilization focuses on improving antimicrobial decision making and avoiding unnecessary downstream testing.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="What themes across these papers suggest that rapid diagnostics improve antimicrobial decision-making more reliably than they improve hard clinical outcomes?",
        limit=3,
    )

    assert {chunk.doc_id for chunk in result} == {
        "RAPID",
        "Single site RCT",
        "fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
    }


def test_retrieval_service_expands_search_text_for_decision_making_vs_clinical_outcomes_query() -> None:
    captured_queries: list[str] = []

    def embedding_fn(texts: list[str]) -> list[list[float]]:
        captured_queries.extend(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]

    chunks = [
        Chunk(
            id="DOC-RAPID:P00001:C01",
            content="Rapid testing improved antimicrobial decision making.",
            metadata=ChunkMetadata(
                doc_id="RAPID",
                chunk_type="text",
                parent_header="DISCUSSION",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RAPID:P00001",
                    "parent_content": "Rapid testing improved antimicrobial decision making.",
                },
            ),
        )
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=embedding_fn)

    service.retrieve(
        query="What themes across these papers suggest that rapid diagnostics improve antimicrobial decision-making more reliably than they improve hard clinical outcomes?",
        limit=1,
    )

    assert len(captured_queries) == 1
    assert "diagnostic stewardship" in captured_queries[0]
    assert "blood culture utilization" in captured_queries[0]


def test_retrieval_service_prefers_results_for_limit_of_detection_workflow_query() -> None:
    chunks = [
        Chunk(
            id="DOC-CF:M00001:C01",
            content="Acquired mass spectral data were processed using the Bruker data analysis software and samples were called positive when diagnostic biomarker ions were detected with an S/N ratio greater than 10.",
            metadata=ChunkMetadata(
                doc_id="Culture-Free Lipidomics-Based Screening Test",
                chunk_type="text",
                parent_header="Materials and Methods",
                page_number=3,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-CF:M00001",
                    "parent_content": "Acquired mass spectral data were processed using the Bruker data analysis software and samples were called positive when diagnostic biomarker ions were detected with an S/N ratio greater than 10.",
                },
            ),
        ),
        Chunk(
            id="DOC-CF:I00001:C01",
            content="In this study, we first determined the optimal condition of lysozyme treatment and established the limit of detection of our assay.",
            metadata=ChunkMetadata(
                doc_id="Culture-Free Lipidomics-Based Screening Test",
                chunk_type="text",
                parent_header="Introduction",
                page_number=2,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-CF:I00001",
                    "parent_content": "In this study, we first determined the optimal condition of lysozyme treatment and established the limit of detection of our assay.",
                },
            ),
        ),
        Chunk(
            id="DOC-CF:R00001:C01",
            content="Incorporating lysozyme into the workflow improved detection limits to 10^2-10^3 CFU/uL. The combination of 100 ug lysozyme per 1 mL urine pellet and a 60-minute incubation provided the most efficient detection rates.",
            metadata=ChunkMetadata(
                doc_id="Culture-Free Lipidomics-Based Screening Test",
                chunk_type="text",
                parent_header="Results",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-CF:R00001",
                    "parent_content": "Incorporating lysozyme into the workflow improved detection limits to 10^2-10^3 CFU/uL. The combination of 100 ug lysozyme per 1 mL urine pellet and a 60-minute incubation provided the most efficient detection rates.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="What lower-limit-of-detection findings are reported for the lipidomics screening workflow?",
        doc_id="Culture-Free Lipidomics-Based Screening Test",
        limit=1,
    )

    assert [(chunk.doc_id, chunk.source) for chunk in result] == [
        ("Culture-Free Lipidomics-Based Screening Test", "Results")
    ]


def test_retrieval_service_keeps_multiple_same_doc_chunks_for_assay_optimization_query() -> None:
    chunks = [
        Chunk(
            id="DOC-CF:M00001:C01",
            content="Urine cultures confirmed gram-positive bacteria, and each sample was treated with lysozyme at the optimal concentration previously determined from contrived LOD tests.",
            metadata=ChunkMetadata(
                doc_id="Culture-Free Lipidomics-Based Screening Test",
                chunk_type="text",
                parent_header="Materials and Methods",
                page_number=4,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-CF:M00001",
                    "parent_content": "Urine cultures confirmed gram-positive bacteria, and each sample was treated with lysozyme at the optimal concentration previously determined from contrived LOD tests.",
                },
            ),
        ),
        Chunk(
            id="DOC-CF:R00001:C01",
            content="Optimal sensitivity was achieved by treating 1 mL of urine pellets with 100 ug lysozyme and incubating for 60 minutes, resulting in improved cardiolipin detection.",
            metadata=ChunkMetadata(
                doc_id="Culture-Free Lipidomics-Based Screening Test",
                chunk_type="text",
                parent_header="Results",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-CF:R00001",
                    "parent_content": "Optimal sensitivity was achieved by treating 1 mL of urine pellets with 100 ug lysozyme and incubating for 60 minutes, resulting in improved cardiolipin detection.",
                },
            ),
        ),
        Chunk(
            id="DOC-NARTEY:D00001:C01",
            content="Adding sonication increased sensitivity for gram-positive bacteria in a later cohort, but this sentence does not give the FLAT optimization parameters.",
            metadata=ChunkMetadata(
                doc_id="nartey-et-al-2024-a-lipidomics-based-method-to-eliminate-negative-urine-culture-in-general-population",
                chunk_type="text",
                parent_header="DISCUSSION",
                page_number=8,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-NARTEY:D00001",
                    "parent_content": "Adding sonication increased sensitivity for gram-positive bacteria in a later cohort, but this sentence does not give the FLAT optimization parameters.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="What lysozyme concentration and incubation time gave optimal cardiolipin detection in the FLAT assay?",
        limit=2,
    )

    assert [(chunk.doc_id, chunk.source) for chunk in result] == [
        ("Culture-Free Lipidomics-Based Screening Test", "Materials and Methods"),
        ("Culture-Free Lipidomics-Based Screening Test", "Results"),
    ]
    assert "100 ug lysozyme" in result[1].content
    assert "60 minutes" in result[1].content


def test_retrieval_service_filters_non_infectious_metric_tables_for_broad_cross_doc_metric_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-NARTEY:T00001",
            content="Source File: nartey.pdf | Table Index: 3 | Section: RESULTS\nSensitivity,Specificity\n70%,99%",
            metadata=ChunkMetadata(
                doc_id="nartey-et-al-2024-a-lipidomics-based-method-to-eliminate-negative-urine-culture-in-general-population",
                chunk_type="table",
                parent_header="RESULTS",
                page_number=5,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "parent_id": "DOC-NARTEY:T00001",
                    "parent_content": "Source File: nartey.pdf | Table Index: 3 | Section: RESULTS\nSensitivity,Specificity\n70%,99%",
                    "contains_metric_values": True,
                },
            ),
        ),
        Chunk(
            id="DOC-JOGC:T00001",
            content="Source File: jogc.pdf | Table Index: 4 | Section: DISCUSSION\nSensitivity,Specificity,PPV,NPV\n80%,75%,70%,85%",
            metadata=ChunkMetadata(
                doc_id="JOGC fibronectin",
                chunk_type="table",
                parent_header="DISCUSSION",
                page_number=7,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "parent_id": "DOC-JOGC:T00001",
                    "parent_content": "Source File: jogc.pdf | Table Index: 4 | Section: DISCUSSION\nSensitivity,Specificity,PPV,NPV\n80%,75%,70%,85%",
                    "contains_metric_values": True,
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(
        repo=repo,
        embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        include_tables=True,
    )

    result = service.retrieve(
        query="Across the indexed studies, which papers contain tabular sensitivity or specificity findings?",
        limit=2,
    )

    assert [chunk.doc_id for chunk in result] == [
        "nartey-et-al-2024-a-lipidomics-based-method-to-eliminate-negative-urine-culture-in-general-population"
    ]


def test_retrieval_service_filters_non_infectious_accuracy_docs_for_broad_metric_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-BAL:P00001:C01",
            content="The presence of S. pneumoniae was confirmed in only 6/17 samples, supporting BAL diagnostic accuracy reporting.",
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="text",
                parent_header="Results",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-BAL:P00001",
                    "parent_content": "The presence of S. pneumoniae was confirmed in only 6/17 samples, supporting BAL diagnostic accuracy reporting.",
                },
            ),
        ),
        Chunk(
            id="DOC-AUNE:T00001",
            content="Source File: hepcidin.pdf | Table Index: 2 | Section: Results\nAccuracy [Range],%.\n145%",
            metadata=ChunkMetadata(
                doc_id="Aune-2020-Optimizing hepcidin measurement with",
                chunk_type="table",
                parent_header="Results",
                page_number=4,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "parent_id": "DOC-AUNE:T00001",
                    "parent_content": "Source File: hepcidin.pdf | Table Index: 2 | Section: Results\nAccuracy [Range],%.\n145%",
                    "contains_metric_values": True,
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(
        repo=repo,
        embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        include_tables=True,
    )

    result = service.retrieve(
        query="Which papers in the indexed set report sensitivity, specificity, or other diagnostic accuracy metrics?",
        limit=2,
    )

    assert [chunk.doc_id for chunk in result] == ["BAL SM"]


def test_retrieval_service_does_not_treat_utilization_as_uti_for_broad_metric_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-BAL:P00001:C01",
            content="The presence of S. pneumoniae was confirmed in only 6/17 samples, supporting BAL diagnostic accuracy reporting.",
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="text",
                parent_header="Results",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-BAL:P00001",
                    "parent_content": "The presence of S. pneumoniae was confirmed in only 6/17 samples, supporting BAL diagnostic accuracy reporting.",
                },
            ),
        ),
        Chunk(
            id="DOC-UTIL:P00001:C01",
            content="This utilization study reports sensitivity and specificity for a non-infectious screening workflow.",
            metadata=ChunkMetadata(
                doc_id="Noninfectious utilization study",
                chunk_type="text",
                parent_header="Abstract",
                page_number=2,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-UTIL:P00001",
                    "parent_content": "This utilization study reports sensitivity and specificity for a non-infectious screening workflow.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Across the indexed studies, which papers report diagnostic performance outcomes for rapid testing?",
        limit=2,
    )

    assert [chunk.doc_id for chunk in result] == ["BAL SM"]


def test_retrieval_service_prefers_study_design_chunks_for_classification_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-RCT:P00001:C01",
            content="We report the first prospective RCT to demonstrate benefit of an rmPCR-based blood culture diagnostic test.",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="text",
                parent_header="DISCUSSION",
                page_number=8,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCT:P00001",
                    "parent_content": "We report the first prospective RCT to demonstrate benefit of an rmPCR-based blood culture diagnostic test.",
                },
            ),
        ),
        Chunk(
            id="DOC-REVIEW:P00001:C01",
            content="We reviewed the tools currently used and the associated approaches in the diagnosis of blood culture-negative endocarditis.",
            metadata=ChunkMetadata(
                doc_id="IJGM-393329-blood-culture-negative-endocarditis--a-review-of-laboratory-",
                chunk_type="text",
                parent_header="Methods",
                page_number=3,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-REVIEW:P00001",
                    "parent_content": "We reviewed the tools currently used and the associated approaches in the diagnosis of blood culture-negative endocarditis.",
                },
            ),
        ),
        Chunk(
            id="DOC-NOISE:P00001:C01",
            content="Mol. Sci. 2021, 22, x FOR PEER REVIEW 6 of 13 Table 1. MRM transitions and parameters for the internal standard.",
            metadata=ChunkMetadata(
                doc_id="ChenMichaelIntJMolSci2021",
                chunk_type="text",
                parent_header="2. Results",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-NOISE:P00001",
                    "parent_content": "Mol. Sci. 2021, 22, x FOR PEER REVIEW 6 of 13 Table 1. MRM transitions and parameters for the internal standard.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Which of these studies are randomized controlled trials, and which are observational or review papers?",
        limit=3,
    )

    assert [chunk.doc_id for chunk in result] == [
        "Single site RCT",
        "IJGM-393329-blood-culture-negative-endocarditis--a-review-of-laboratory-",
    ]


def test_retrieval_service_excludes_non_domain_retrospective_noise_for_classification_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-RCT:P00001:C01",
            content="We report the first prospective RCT to demonstrate benefit of an rmPCR-based blood culture diagnostic test.",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="text",
                parent_header="DISCUSSION",
                page_number=8,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCT:P00001",
                    "parent_content": "We report the first prospective RCT to demonstrate benefit of an rmPCR-based blood culture diagnostic test.",
                },
            ),
        ),
        Chunk(
            id="DOC-RAPID:P00001:C01",
            content="Strengths of this study include its pragmatic trial design and incorporation of baseline activities of AS programs.",
            metadata=ChunkMetadata(
                doc_id="RAPID",
                chunk_type="text",
                parent_header="DISCUSSION",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RAPID:P00001",
                    "parent_content": "Strengths of this study include its pragmatic trial design and incorporation of baseline activities of AS programs.",
                },
            ),
        ),
        Chunk(
            id="DOC-REVIEW:P00001:C01",
            content="In total, 18 studies were reviewed in full while evaluating blood culture negative endocarditis tools.",
            metadata=ChunkMetadata(
                doc_id="IJGM-393329-blood-culture-negative-endocarditis--a-review-of-laboratory-",
                chunk_type="text",
                parent_header="Methods",
                page_number=3,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-REVIEW:P00001",
                    "parent_content": "In total, 18 studies were reviewed in full while evaluating blood culture negative endocarditis tools.",
                },
            ),
        ),
        Chunk(
            id="DOC-NOISE:P00001:C01",
            content="This retrospective cross-sectional study cannot establish causality for ferritin and renal insufficiency associations.",
            metadata=ChunkMetadata(
                doc_id="jmsacl",
                chunk_type="text",
                parent_header="4. Discussion",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-NOISE:P00001",
                    "parent_content": "This retrospective cross-sectional study cannot establish causality for ferritin and renal insufficiency associations.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Which of these studies are randomized controlled trials, and which are observational or review papers?",
        limit=4,
    )

    assert [chunk.doc_id for chunk in result] == [
        "RAPID",
        "Single site RCT",
        "IJGM-393329-blood-culture-negative-endocarditis--a-review-of-laboratory-",
    ]


def test_retrieval_service_expands_search_text_for_classification_queries() -> None:
    captured_queries: list[str] = []

    def embedding_fn(texts: list[str]) -> list[list[float]]:
        captured_queries.extend(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]

    chunks = [
        Chunk(
            id="DOC-RCT:P00001:C01",
            content="We report the first prospective RCT to demonstrate benefit of an rmPCR-based blood culture diagnostic test.",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="text",
                parent_header="DISCUSSION",
                page_number=8,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCT:P00001",
                    "parent_content": "We report the first prospective RCT to demonstrate benefit of an rmPCR-based blood culture diagnostic test.",
                },
            ),
        )
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=embedding_fn)

    service.retrieve(
        query="Which of these studies are randomized controlled trials, and which are observational or review papers?",
        limit=1,
    )

    assert len(captured_queries) == 1
    assert "diagnostic stewardship" in captured_queries[0]
    assert "blood culture" in captured_queries[0]


def test_retrieval_service_prefers_domain_reviews_over_hepcidin_noise_for_classification_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-RCT:P00001:C01",
            content="We report the first prospective RCT to demonstrate benefit of an rmPCR-based blood culture diagnostic test.",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="text",
                parent_header="DISCUSSION",
                page_number=8,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCT:P00001",
                    "parent_content": "We report the first prospective RCT to demonstrate benefit of an rmPCR-based blood culture diagnostic test.",
                },
            ),
        ),
        Chunk(
            id="DOC-FABRE:P00001:C01",
            content="This review discusses diagnostic stewardship and blood culture utilization in the hospital setting.",
            metadata=ChunkMetadata(
                doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
                chunk_type="text",
                parent_header="Discussion",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-FABRE:P00001",
                    "parent_content": "This review discusses diagnostic stewardship and blood culture utilization in the hospital setting.",
                },
            ),
        ),
        Chunk(
            id="DOC-HEP:P00001:C01",
            content="This review summarizes hepcidin biology and inflammatory regulation in anemia of chronic disease.",
            metadata=ChunkMetadata(
                doc_id="hepcidin acute phase",
                chunk_type="text",
                parent_header="Introduction",
                page_number=2,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-HEP:P00001",
                    "parent_content": "This review summarizes hepcidin biology and inflammatory regulation in anemia of chronic disease.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Which of these studies are randomized controlled trials, and which are observational or review papers?",
        limit=3,
    )

    assert set(chunk.doc_id for chunk in result) == {
        "Single site RCT",
        "fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
    }


def test_retrieval_service_prefers_hepcidin_standardization_paper_for_disambiguation_query() -> None:
    chunks = [
        Chunk(
            id="DOC-AUNE:P00001:C01",
            content="This proficiency testing study proposes assay standardization across laboratories using a high-level calibrator and reference material.",
            metadata=ChunkMetadata(
                doc_id="Aune-2020-Optimizing hepcidin measurement with",
                chunk_type="text",
                parent_header="Discussion",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-AUNE:P00001",
                    "parent_content": "This proficiency testing study proposes assay standardization across laboratories using a high-level calibrator and reference material.",
                },
            ),
        ),
        Chunk(
            id="DOC-JMSACL:P00001:C01",
            content="This observational paper interprets hepcidin-25 in renal dysfunction and inflammation.",
            metadata=ChunkMetadata(
                doc_id="jmsacl",
                chunk_type="text",
                parent_header="2. Materials and Methods",
                page_number=2,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-JMSACL:P00001",
                    "parent_content": "This observational paper interprets hepcidin-25 in renal dysfunction and inflammation.",
                },
            ),
        ),
        Chunk(
            id="DOC-RCM:P00001:C01",
            content="We developed an HPLC/MS/MS assay with simple sample preparation for hepcidin measurement.",
            metadata=ChunkMetadata(
                doc_id="RCM publication",
                chunk_type="text",
                parent_header="5 | RESULTS AND DISCUSSION",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCM:P00001",
                    "parent_content": "We developed an HPLC/MS/MS assay with simple sample preparation for hepcidin measurement.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Which hepcidin paper in the indexed set focuses on proficiency testing and assay standardization rather than CKD pathophysiology or assay implementation?",
        limit=3,
    )

    assert result[0].doc_id == "Aune-2020-Optimizing hepcidin measurement with"


def test_retrieval_service_prefers_bloodstream_rapid_diagnostics_over_urine_or_bal_noise() -> None:
    chunks = [
        Chunk(
            id="DOC-RCT:P00001:C01",
            content="The single-site randomized trial reduced vancomycin exposure and improved escalation and de-escalation timing in bloodstream infection management.",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="text",
                parent_header="DISCUSSION",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCT:P00001",
                    "parent_content": "The single-site randomized trial reduced vancomycin exposure and improved escalation and de-escalation timing in bloodstream infection management.",
                },
            ),
        ),
        Chunk(
            id="DOC-RAPID:P00001:C01",
            content="RAPID improved organism ID and phenotypic AST turnaround compared with standard of care in bloodstream infection.",
            metadata=ChunkMetadata(
                doc_id="RAPID",
                chunk_type="text",
                parent_header="DISCUSSION",
                page_number=6,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RAPID:P00001",
                    "parent_content": "RAPID improved organism ID and phenotypic AST turnaround compared with standard of care in bloodstream infection.",
                },
            ),
        ),
        Chunk(
            id="DOC-FLAT:P00001:C01",
            content="The FLAT assay detects urine lipidomics biomarkers for direct pathogen detection.",
            metadata=ChunkMetadata(
                doc_id="Culture-Free Lipidomics-Based Screening Test",
                chunk_type="text",
                parent_header="Introduction",
                page_number=2,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-FLAT:P00001",
                    "parent_content": "The FLAT assay detects urine lipidomics biomarkers for direct pathogen detection.",
                },
            ),
        ),
        Chunk(
            id="DOC-BAL:P00001:C01",
            content="PCR/ESI-MS enables BAL pathogen detection directly from bronchoalveolar lavage samples.",
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="text",
                parent_header="Introduction",
                page_number=2,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-BAL:P00001",
                    "parent_content": "PCR/ESI-MS enables BAL pathogen detection directly from bronchoalveolar lavage samples.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Which indexed rapid-diagnostics papers report turnaround or stewardship improvements in bloodstream infection management rather than urine lipidomics or BAL pathogen detection?",
        limit=4,
    )

    assert set(chunk.doc_id for chunk in result[:2]) == {"Single site RCT", "RAPID"}


def test_retrieval_service_prefers_review_policy_docs_over_primary_study_noise() -> None:
    chunks = [
        Chunk(
            id="DOC-FABRE:P00001:C01",
            content="This review discusses diagnostic stewardship and blood culture utilization in the hospital setting.",
            metadata=ChunkMetadata(
                doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
                chunk_type="text",
                parent_header="Discussion",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-FABRE:P00001",
                    "parent_content": "This review discusses diagnostic stewardship and blood culture utilization in the hospital setting.",
                },
            ),
        ),
        Chunk(
            id="DOC-IJGM:P00001:C01",
            content="We reviewed the laboratory workup for blood culture-negative endocarditis and diagnostic approaches.",
            metadata=ChunkMetadata(
                doc_id="IJGM-393329-blood-culture-negative-endocarditis--a-review-of-laboratory-",
                chunk_type="text",
                parent_header="Methods",
                page_number=3,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-IJGM:P00001",
                    "parent_content": "We reviewed the laboratory workup for blood culture-negative endocarditis and diagnostic approaches.",
                },
            ),
        ),
        Chunk(
            id="DOC-RCT:P00001:C01",
            content="This randomized trial reports patient outcomes and stewardship timing changes.",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="text",
                parent_header="DISCUSSION",
                page_number=8,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RCT:P00001",
                    "parent_content": "This randomized trial reports patient outcomes and stewardship timing changes.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Which indexed reviews focus on diagnostic stewardship or laboratory workup policy rather than primary observational assay cohorts?",
        limit=3,
    )

    assert [chunk.doc_id for chunk in result[:2]] == [
        "fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
        "IJGM-393329-blood-culture-negative-endocarditis--a-review-of-laboratory-",
    ]


def test_retrieval_service_allows_linked_table_references_for_explicit_table_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-BAL:P00001:C01",
            content="Table 3. Data for patients with discrepant results with respect to respiratory pathogens obtained by PCR/ESI-MS and routine culture-based analysis.",
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="text",
                parent_header="Results",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-BAL:P00001",
                    "parent_content": "Table 3. Data for patients with discrepant results with respect to respiratory pathogens obtained by PCR/ESI-MS and routine culture-based analysis.",
                    "referenced_table_indices": [3],
                    "has_table_reference": True,
                },
            ),
        ),
        Chunk(
            id="DOC-RCT:T00001",
            content="Source File: Single site RCT.pdf | Table Index: 3 | Section: Results\nRapid Multiplex PCR Panel Result,Standard Culture and Phenotypic Susceptibility Testing,No.",
            metadata=ChunkMetadata(
                doc_id="Single site RCT",
                chunk_type="table",
                parent_header="Results",
                page_number=6,
                extra={
                    "content_role": "table",
                    "section_role": "body",
                    "parent_content": "Source File: Single site RCT.pdf | Table Index: 3 | Section: Results\nRapid Multiplex PCR Panel Result,Standard Culture and Phenotypic Susceptibility Testing,No.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(
        repo=repo,
        embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        include_tables=True,
    )

    result = service.retrieve(
        query="Which indexed paper contains a discrepant-results table for respiratory pathogens obtained by PCR/ESI-MS and routine culture?",
        limit=2,
    )

    assert [chunk.doc_id for chunk in result] == ["BAL SM"]


def test_retrieval_service_excludes_embedded_markdown_table_prose_for_non_table_queries() -> None:
    chunks = [
        Chunk(
            id="DOC-REVIEW:P00001:C01",
            content="| Advantage | Disadvantage |\n| --- | --- |\n| Rapid | Low specificity |",
            metadata=ChunkMetadata(
                doc_id="DOC-REVIEW",
                chunk_type="text",
                parent_header="Conclusion",
                page_number=7,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-REVIEW:P00001",
                    "parent_content": "| Advantage | Disadvantage |\n| --- | --- |\n| Rapid | Low specificity |",
                },
            ),
        ),
        Chunk(
            id="DOC-RAPID:P00001:C01",
            content="Discussion of rapid diagnostics implementation implications in clinical practice.",
            metadata=ChunkMetadata(
                doc_id="DOC-RAPID",
                chunk_type="text",
                parent_header="Discussion",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-RAPID:P00001",
                    "parent_content": "Discussion of rapid diagnostics implementation implications in clinical practice.",
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(repo=repo, embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    result = service.retrieve(
        query="Across the indexed studies, which papers discuss clinical usefulness or implementation implications of rapid diagnostics?",
        limit=2,
    )

    assert [chunk.doc_id for chunk in result] == ["DOC-RAPID"]


def test_retrieval_service_filters_generic_linked_table_prose_when_topic_overlap_is_weak() -> None:
    chunks = [
        Chunk(
            id="DOC-BAL:P00001:C01",
            content="Table 3. Data for patients with discrepant results with respect to respiratory pathogens obtained by PCR/ESI-MS and routine culture-based analysis.",
            metadata=ChunkMetadata(
                doc_id="BAL SM",
                chunk_type="text",
                parent_header="Results",
                page_number=5,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-BAL:P00001",
                    "parent_content": "Table 3. Data for patients with discrepant results with respect to respiratory pathogens obtained by PCR/ESI-MS and routine culture-based analysis.",
                    "referenced_table_indices": [3],
                    "has_table_reference": True,
                },
            ),
        ),
        Chunk(
            id="DOC-UTI:P00001:C01",
            content="The FLAT assay had a sensitivity of 70% and specificity of 99% with positive and negative predictive values of 93% and 99%, respectively (Table 3).",
            metadata=ChunkMetadata(
                doc_id="UTI Paper",
                chunk_type="text",
                parent_header="RESULTS",
                page_number=4,
                extra={
                    "content_role": "child",
                    "section_role": "body",
                    "parent_id": "DOC-UTI:P00001",
                    "parent_content": "The FLAT assay had a sensitivity of 70% and specificity of 99% with positive and negative predictive values of 93% and 99%, respectively (Table 3).",
                    "referenced_table_indices": [3],
                    "has_table_reference": True,
                },
            ),
        ),
    ]
    repo = FakeVectorRepository(chunks)
    service = RetrievalService(
        repo=repo,
        embedding_fn=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        include_tables=True,
    )

    result = service.retrieve(
        query="Which indexed paper contains a discrepant-results table for respiratory pathogens obtained by PCR/ESI-MS and routine culture?",
        limit=1,
    )

    assert [chunk.doc_id for chunk in result] == ["BAL SM"]
