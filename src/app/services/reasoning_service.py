from __future__ import annotations

from src.app.ports.llm_port import LLMPort
from src.app.prompts.research_prompt import build_research_prompt
from src.app.services.retrieval_service import RetrievalService


class ReasoningService:
    """Generate synthesized research insights from retrieved evidence."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_client: LLMPort,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._llm = llm_client

    def research(self, query: str, doc_id: str | None = None, limit: int = 8) -> str:
        retrieved_chunks = self._retrieval_service.retrieve(query=query, doc_id=doc_id, limit=limit)
        context = self._retrieval_service.serialize_for_prompt(retrieved_chunks)
        prompt = build_research_prompt(query=query, context=context)
        return self._llm.generate(prompt)

