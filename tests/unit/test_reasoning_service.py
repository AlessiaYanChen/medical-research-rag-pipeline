from __future__ import annotations

from src.app.services.retrieval_service import RetrievedChunk
from src.app.services.reasoning_service import ReasoningService


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
    def __init__(self) -> None:
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "Research Insight\nSynthetic answer\n\nEvidence Basis\nDOC-1"


def test_reasoning_service_uses_retrieval_context_and_llm() -> None:
    retrieval = FakeRetrievalService(
        [
            RetrievedChunk(
                source="Results",
                doc_id="DOC-1",
                content="LDL-C improved in the intervention group.",
                chunk_type="text",
                content_role="child",
            )
        ]
    )
    llm = FakeLLM()
    service = ReasoningService(retrieval_service=retrieval, llm_client=llm)

    result = service.research(query="What is the biomarker effect?", limit=6)

    assert retrieval.last_args == ("What is the biomarker effect?", None, 6)
    assert llm.last_prompt is not None
    assert "Retrieved Context:" in llm.last_prompt
    assert "LDL-C improved in the intervention group." in llm.last_prompt
    assert "Research Question: What is the biomarker effect?" in llm.last_prompt
    assert result.startswith("Research Insight")

