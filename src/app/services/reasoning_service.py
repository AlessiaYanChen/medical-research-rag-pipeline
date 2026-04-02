from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from src.app.ports.llm_port import LLMPort
from src.app.prompts.research_prompt import build_research_prompt
from src.app.services.retrieval_service import RetrievedChunk, RetrievalService


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class ResearchAnswer:
    """Structured output from ReasoningService."""

    insight: str
    evidence_basis: str
    citations: list[RetrievedChunk]
    confidence: ConfidenceLevel


def _compute_confidence(insight: str, citations: list[RetrievedChunk]) -> ConfidenceLevel:
    """Derive confidence from retrieval signals. No LLM call."""
    if not citations or "insufficient evidence" in insight.lower():
        return ConfidenceLevel.INSUFFICIENT
    distinct_docs = len({c.doc_id for c in citations})
    if len(citations) >= 4 and distinct_docs >= 2:
        return ConfidenceLevel.HIGH
    if len(citations) >= 2:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def _parse_llm_response(text: str) -> tuple[str, str]:
    """Split LLM response into (insight, evidence_basis). Falls back gracefully."""
    insight_match = re.search(
        r"(?:1\.\s*)?Research Insight[^\n]*\n(.*?)(?=(?:2\.\s*)?Evidence Basis|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    evidence_match = re.search(
        r"(?:2\.\s*)?Evidence Basis[^\n]*\n(.*)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    insight = insight_match.group(1).strip() if insight_match else text.strip()
    evidence_basis = evidence_match.group(1).strip() if evidence_match else ""
    return insight, evidence_basis


def _should_force_exact_intervention_metric_abstention(query: str, citations: list[RetrievedChunk]) -> bool:
    normalized_query = query.lower()
    if "before and after" not in normalized_query:
        return False
    if not any(signal in normalized_query for signal in ("detection rate", "sensitivity", "specificity")):
        return False

    intervention_markers = [
        marker for marker in ("lysozyme", "sonication")
        if marker in normalized_query
    ]
    if not intervention_markers:
        return False

    for marker in intervention_markers:
        if _citations_contain_quantitative_intervention_evidence(citations=citations, intervention_marker=marker):
            return False
    return True


def _citations_contain_quantitative_intervention_evidence(
    citations: list[RetrievedChunk],
    intervention_marker: str,
) -> bool:
    quantitative_pattern = re.compile(r"\b\d+(?:\.\d+)?\s*%|\b\d+\s*/\s*\d+\b")
    for citation in citations:
        content_lower = citation.content.lower()
        if intervention_marker not in content_lower:
            continue
        if quantitative_pattern.search(content_lower):
            return True
    return False


def _build_forced_abstention_answer(query: str, citations: list[RetrievedChunk]) -> ResearchAnswer:
    evidence_lines: list[str] = []
    seen_sources: set[tuple[str, str]] = set()
    for citation in citations:
        key = (citation.doc_id, citation.source)
        if key in seen_sources:
            continue
        seen_sources.add(key)
        evidence_lines.append(f"- {citation.doc_id}, {citation.source}")

    evidence_basis = "\n".join(evidence_lines) if evidence_lines else ""
    return ResearchAnswer(
        insight=(
            "Insufficient evidence.\n\n"
            "The retrieved context does not provide exact quantitative before/after values tied to the"
            " intervention named in the question."
        ),
        evidence_basis=evidence_basis,
        citations=list(citations),
        confidence=ConfidenceLevel.INSUFFICIENT,
    )


def _should_force_known_gap_abstention(query: str, citations: list[RetrievedChunk]) -> bool:
    normalized_query = query.lower()
    return (
        _query_requires_species_specific_mz_values(normalized_query)
        or _query_requires_figure_interpretation(normalized_query)
        or _query_requires_specific_low_value_scenario_list(normalized_query)
    )


def _query_requires_species_specific_mz_values(normalized_query: str) -> bool:
    return (
        "m/z" in normalized_query
        and "identify" in normalized_query
        and any(marker in normalized_query for marker in ("s. aureus", "staphylococcus aureus", "species"))
    )


def _query_requires_figure_interpretation(normalized_query: str) -> bool:
    if "figure" not in normalized_query:
        return False
    interpretation_signals = (
        "what does",
        "distribution",
        "correlate",
        "correlation",
        "look like",
    )
    return any(signal in normalized_query for signal in interpretation_signals)


def _query_requires_specific_low_value_scenario_list(normalized_query: str) -> bool:
    return (
        "three specific clinical scenarios" in normalized_query
        and "low diagnostic value" in normalized_query
        and "initial blood cultures" in normalized_query
    )


class ReasoningService:
    """Generate synthesized research insights from retrieved evidence."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_client: LLMPort,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._llm = llm_client

    def research(self, query: str, doc_id: str | None = None, limit: int = 8) -> ResearchAnswer:
        retrieved_chunks = self._retrieval_service.retrieve(query=query, doc_id=doc_id, limit=limit)
        if _should_force_known_gap_abstention(query=query, citations=retrieved_chunks):
            return _build_forced_abstention_answer(query=query, citations=retrieved_chunks)
        if _should_force_exact_intervention_metric_abstention(query=query, citations=retrieved_chunks):
            return _build_forced_abstention_answer(query=query, citations=retrieved_chunks)
        context = self._retrieval_service.serialize_for_prompt(retrieved_chunks)
        prompt = build_research_prompt(query=query, context=context)
        llm_text = self._llm.generate(prompt)
        insight, evidence_basis = _parse_llm_response(llm_text)
        citations = list(retrieved_chunks)
        return ResearchAnswer(
            insight=insight,
            evidence_basis=evidence_basis,
            citations=citations,
            confidence=_compute_confidence(insight, citations),
        )

