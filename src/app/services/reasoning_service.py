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

