from __future__ import annotations

from src.app.services.retrieval_service import RetrievedChunk
from src.app.services.reasoning_service import (
    ConfidenceLevel,
    ResearchAnswer,
    ReasoningService,
    _compute_confidence,
    _parse_llm_response,
)


class FakeRetrievalService:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.last_args: tuple[str, str | None, int] | None = None

    def retrieve(self, query: str, doc_id: str | None = None, limit: int = 8) -> list[RetrievedChunk]:
        self.last_args = (query, doc_id, limit)
        return self._chunks

    @staticmethod
    def serialize_for_prompt(chunks: list[RetrievedChunk]) -> str:
        return "\n\n".join(
            f"Source: {chunk.source} | Document: {chunk.doc_id}\n{chunk.content}"
            for chunk in chunks
        )


class FakeLLM:
    def __init__(self, response: str = "") -> None:
        self.last_prompt: str | None = None
        self._response = response or (
            "1. Research Insight\nLDL-C improved by 12% (p=0.01, 95% CI 4–20%).\n\n"
            "2. Evidence Basis\nDOC-1, Results section."
        )

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self._response


_CHUNK = RetrievedChunk(
    source="Results",
    doc_id="DOC-1",
    content="LDL-C improved in the intervention group.",
    chunk_type="text",
    content_role="child",
)


def test_reasoning_service_returns_research_answer() -> None:
    retrieval = FakeRetrievalService([_CHUNK])
    llm = FakeLLM()
    service = ReasoningService(retrieval_service=retrieval, llm_client=llm)

    result = service.research(query="What is the biomarker effect?", limit=6)

    assert isinstance(result, ResearchAnswer)
    assert retrieval.last_args == ("What is the biomarker effect?", None, 6)


def test_reasoning_service_prompt_contains_hardened_instructions() -> None:
    retrieval = FakeRetrievalService([_CHUNK])
    llm = FakeLLM()
    service = ReasoningService(retrieval_service=retrieval, llm_client=llm)

    service.research(query="What is the biomarker effect?", limit=6)

    assert llm.last_prompt is not None
    assert "Retrieved Context:" in llm.last_prompt
    assert "LDL-C improved in the intervention group." in llm.last_prompt
    assert "Research Question: What is the biomarker effect?" in llm.last_prompt
    # Prompt hardening: study design, effect sizes, uncertainty, limitations
    assert "study design" in llm.last_prompt
    assert "effect sizes" in llm.last_prompt
    assert "confidence intervals" in llm.last_prompt
    assert "limitations" in llm.last_prompt


def test_reasoning_service_parses_insight_and_evidence_basis() -> None:
    retrieval = FakeRetrievalService([_CHUNK])
    llm = FakeLLM()
    service = ReasoningService(retrieval_service=retrieval, llm_client=llm)

    result = service.research(query="What is the biomarker effect?", limit=6)

    assert "LDL-C improved" in result.insight
    assert "DOC-1" in result.evidence_basis


def test_reasoning_service_citations_match_retrieved_chunks() -> None:
    retrieval = FakeRetrievalService([_CHUNK])
    llm = FakeLLM()
    service = ReasoningService(retrieval_service=retrieval, llm_client=llm)

    result = service.research(query="What is the biomarker effect?", limit=6)

    assert result.citations == [_CHUNK]


def test_parse_llm_response_standard_format() -> None:
    text = (
        "1. Research Insight\nSome insight here.\n\n"
        "2. Evidence Basis\nDOC-1, Results."
    )
    insight, evidence_basis = _parse_llm_response(text)
    assert insight == "Some insight here."
    assert evidence_basis == "DOC-1, Results."


def test_parse_llm_response_without_numbering() -> None:
    text = "Research Insight\nInsight text.\n\nEvidence Basis\nSource A."
    insight, evidence_basis = _parse_llm_response(text)
    assert insight == "Insight text."
    assert evidence_basis == "Source A."


def test_parse_llm_response_fallback_when_no_headers() -> None:
    text = "Some unstructured response from the LLM."
    insight, evidence_basis = _parse_llm_response(text)
    assert insight == text
    assert evidence_basis == ""


def _make_chunk(doc_id: str = "DOC-1") -> RetrievedChunk:
    return RetrievedChunk(
        source="Results", doc_id=doc_id, content="text", chunk_type="text", content_role="child"
    )


# --- _compute_confidence ---

def test_compute_confidence_insufficient_when_no_citations() -> None:
    assert _compute_confidence("Some insight.", []) == ConfidenceLevel.INSUFFICIENT


def test_compute_confidence_insufficient_when_insight_says_so() -> None:
    assert _compute_confidence("Insufficient evidence.", [_make_chunk()]) == ConfidenceLevel.INSUFFICIENT


def test_compute_confidence_insufficient_case_insensitive() -> None:
    assert _compute_confidence("INSUFFICIENT EVIDENCE found.", [_make_chunk()]) == ConfidenceLevel.INSUFFICIENT


def test_compute_confidence_high() -> None:
    chunks = [_make_chunk("DOC-1"), _make_chunk("DOC-1"), _make_chunk("DOC-2"), _make_chunk("DOC-2")]
    assert _compute_confidence("Solid findings.", chunks) == ConfidenceLevel.HIGH


def test_compute_confidence_high_requires_two_distinct_docs() -> None:
    chunks = [_make_chunk("DOC-1")] * 4
    assert _compute_confidence("Solid findings.", chunks) == ConfidenceLevel.MEDIUM


def test_compute_confidence_medium() -> None:
    chunks = [_make_chunk("DOC-1"), _make_chunk("DOC-1")]
    assert _compute_confidence("Some findings.", chunks) == ConfidenceLevel.MEDIUM


def test_compute_confidence_low() -> None:
    assert _compute_confidence("One result.", [_make_chunk()]) == ConfidenceLevel.LOW


def test_reasoning_service_attaches_confidence() -> None:
    retrieval = FakeRetrievalService([_CHUNK])
    llm = FakeLLM()
    service = ReasoningService(retrieval_service=retrieval, llm_client=llm)
    result = service.research(query="What is the effect?", limit=6)
    assert result.confidence == ConfidenceLevel.LOW  # 1 chunk, 1 doc


def test_parse_llm_response_with_section_label_text() -> None:
    # LLM includes the label text from the prompt on the header line
    text = (
        "1. Research Insight — include study design, effect sizes with uncertainty, and limitations where present in the context.\n"
        "Patients showed improvement.\n\n"
        "2. Evidence Basis — list each source (document ID and section) that supports your answer.\n"
        "DOC-2, Methods."
    )
    insight, evidence_basis = _parse_llm_response(text)
    assert insight == "Patients showed improvement."
    assert evidence_basis == "DOC-2, Methods."
