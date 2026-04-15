from __future__ import annotations

from src.app.services.retrieval_service import RetrievedChunk
from src.app.services.reasoning_service import (
    ConfidenceLevel,
    ResearchAnswer,
    ReasoningService,
    _build_forced_abstention_answer,
    _citations_contain_quantitative_intervention_evidence,
    _compute_confidence,
    _parse_llm_response,
    _should_force_exact_intervention_metric_abstention,
    _should_force_known_gap_abstention,
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


def test_quantitative_intervention_evidence_detects_matching_marker_and_numeric_value() -> None:
    citations = [
        RetrievedChunk(
            source="Results",
            doc_id="DOC-1",
            content="Lysozyme treatment increased gram-positive detection from 8% to 51%.",
            chunk_type="text",
            content_role="child",
        )
    ]

    assert _citations_contain_quantitative_intervention_evidence(citations, "lysozyme") is True


def test_reasoning_service_forces_abstention_when_exact_intervention_metric_is_missing() -> None:
    citations = [
        RetrievedChunk(
            source="RESULTS",
            doc_id="nartey-et-al-2024-a-lipidomics-based-method-to-eliminate-negative-urine-culture-in-general-population",
            content="Only 8% of the 402 samples known to be Gram positive were detected by FLAT. Sonication increased detection to 51%.",
            chunk_type="text",
            content_role="child",
        ),
        RetrievedChunk(
            source="Results",
            doc_id="Culture-Free Lipidomics-Based Screening Test",
            content="The optimized FLAT method incorporating lysozyme treatment significantly enhanced the detection of gram-positive uropathogens.",
            chunk_type="text",
            content_role="child",
        ),
    ]
    retrieval = FakeRetrievalService(citations)
    llm = FakeLLM()
    service = ReasoningService(retrieval_service=retrieval, llm_client=llm)

    result = service.research(
        query="What was the detection rate for gram-positive bacteria before and after lysozyme treatment in the FLAT study?",
        limit=6,
    )

    assert _should_force_exact_intervention_metric_abstention(
        query="What was the detection rate for gram-positive bacteria before and after lysozyme treatment in the FLAT study?",
        citations=citations,
    ) is True
    assert result.confidence == ConfidenceLevel.INSUFFICIENT
    assert "Insufficient evidence." in result.insight
    assert llm.last_prompt is None


def test_build_forced_abstention_answer_lists_supporting_sources() -> None:
    citations = [
        RetrievedChunk(
            source="Results",
            doc_id="DOC-1",
            content="Qualitative lysozyme improvement only.",
            chunk_type="text",
            content_role="child",
        )
    ]

    result = _build_forced_abstention_answer(
        query="What was the detection rate before and after lysozyme treatment?",
        citations=citations,
    )

    assert result.confidence == ConfidenceLevel.INSUFFICIENT
    assert "DOC-1, Results" in result.evidence_basis


def test_reasoning_service_forces_abstention_for_species_specific_mz_query() -> None:
    citations = [
        RetrievedChunk(
            source="Materials and Methods",
            doc_id="Culture-Free Lipidomics-Based Screening Test",
            content="Gram-positive biomarkers were detected in the m/z 1000-1400 range.",
            chunk_type="text",
            content_role="child",
        ),
        RetrievedChunk(
            source="Discussion",
            doc_id="nartey-et-al-2024-a-lipidomics-based-method-to-eliminate-negative-urine-culture-in-general-population",
            content="Cardiolipin and lipid A biomarkers occur broadly between m/z 1000 and 2400.",
            chunk_type="text",
            content_role="child",
        ),
    ]
    retrieval = FakeRetrievalService(citations)
    llm = FakeLLM()
    service = ReasoningService(retrieval_service=retrieval, llm_client=llm)

    result = service.research(
        query="What cardiolipin m/z values identify S. aureus in the FLAT assay?",
        limit=6,
    )

    assert _should_force_known_gap_abstention(
        query="What cardiolipin m/z values identify S. aureus in the FLAT assay?",
        citations=citations,
    ) is True
    assert result.confidence == ConfidenceLevel.INSUFFICIENT
    assert llm.last_prompt is None


def test_reasoning_service_forces_abstention_for_figure_interpretation_query() -> None:
    citations = [
        RetrievedChunk(
            source="Results",
            doc_id="BAL SM",
            content="Figure 1 is referenced, but the retrieved text does not describe the plotted distribution.",
            chunk_type="text",
            content_role="child",
        )
    ]
    retrieval = FakeRetrievalService(citations)
    llm = FakeLLM()
    service = ReasoningService(retrieval_service=retrieval, llm_client=llm)

    result = service.research(
        query="What does the distribution of species per BAL sample look like in Figure 1?",
        limit=6,
    )

    assert result.confidence == ConfidenceLevel.INSUFFICIENT
    assert llm.last_prompt is None


def test_reasoning_service_forces_abstention_for_specific_scenario_list_query() -> None:
    citations = [
        RetrievedChunk(
            source="Discussion",
            doc_id="fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship",
            content="Repeat blood cultures were often inappropriate after prior negative cultures without new signs of infection.",
            chunk_type="text",
            content_role="child",
        )
    ]
    retrieval = FakeRetrievalService(citations)
    llm = FakeLLM()
    service = ReasoningService(retrieval_service=retrieval, llm_client=llm)

    result = service.research(
        query="According to the Fabre et al. minireview, what are three specific clinical scenarios where initial blood cultures are considered to have 'Low diagnostic value'?",
        limit=6,
    )

    assert result.confidence == ConfidenceLevel.INSUFFICIENT
    assert llm.last_prompt is None


def test_reasoning_service_forces_abstention_for_missing_named_comparator_query() -> None:
    citations = [
        RetrievedChunk(
            source="RESULTS",
            doc_id="RAPID",
            content="RAPID reduced time to first antibiotic change compared with standard of care.",
            chunk_type="text",
            content_role="child",
        ),
        RetrievedChunk(
            source="RESULTS",
            doc_id="Single site RCT",
            content="Rapid multiplex PCR reduced time to organism identification after positive Gram stain.",
            chunk_type="text",
            content_role="child",
        ),
    ]
    retrieval = FakeRetrievalService(citations)
    llm = FakeLLM()
    service = ReasoningService(retrieval_service=retrieval, llm_client=llm)

    result = service.research(
        query="How did RAPID compare with the BioFire BCID2 platform for organism identification turnaround time?",
        limit=6,
    )

    assert _should_force_known_gap_abstention(
        query="How did RAPID compare with the BioFire BCID2 platform for organism identification turnaround time?",
        citations=citations,
    ) is True
    assert result.confidence == ConfidenceLevel.INSUFFICIENT
    assert llm.last_prompt is None


def test_reasoning_service_forces_abstention_for_exact_subgroup_summary_query() -> None:
    citations = [
        RetrievedChunk(
            source="Results",
            doc_id="jmsacl",
            content="CKD stage 4+ had the highest hepcidin-25 concentration, but the subgroup median eGFR was not reported.",
            chunk_type="text",
            content_role="child",
        ),
        RetrievedChunk(
            source="Results",
            doc_id="hepcidin ckd",
            content="Ferritin was the only independent predictor of hepcidin-25, and no significant correlation with eGFR was found.",
            chunk_type="text",
            content_role="child",
        ),
    ]
    retrieval = FakeRetrievalService(citations)
    llm = FakeLLM()
    service = ReasoningService(retrieval_service=retrieval, llm_client=llm)

    result = service.research(
        query="What was the exact median eGFR of the subgroup with the highest hepcidin-25 concentration in the CKD hepcidin paper?",
        limit=6,
    )

    assert _should_force_known_gap_abstention(
        query="What was the exact median eGFR of the subgroup with the highest hepcidin-25 concentration in the CKD hepcidin paper?",
        citations=citations,
    ) is True
    assert result.confidence == ConfidenceLevel.INSUFFICIENT
    assert llm.last_prompt is None


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
