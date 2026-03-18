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

    assert repo.last_search_args == ([0.1, 0.2, 0.3], "DOC-2", 20)
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

    assert repo.last_search_args == ([0.1, 0.2, 0.3], None, 20)
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
    assert result[1].source == "Abstract"


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

    assert [chunk.source for chunk in result] == ["Discussion", "Discussion", "Abstract"]


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

    assert [chunk.doc_id for chunk in result] == ["DOC-A", "DOC-A", "DOC-B"]
