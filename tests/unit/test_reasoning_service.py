from __future__ import annotations

from src.app.services.reasoning_service import ReasoningService


class FakeRetrievalService:
    def __init__(self, context: str) -> None:
        self._context = context
        self.last_args: tuple[str, str | None, int] | None = None

    def retrieve(self, query: str, doc_id: str | None = None, limit: int = 8) -> str:
        self.last_args = (query, doc_id, limit)
        return self._context


class FakeLLM:
    def __init__(self) -> None:
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "Research Insight\nSynthetic answer\n\nEvidence Basis\nDOC-1"


def test_reasoning_service_uses_retrieval_context_and_llm() -> None:
    retrieval = FakeRetrievalService(
        "Source: Results | Document: DOC-1\nLDL-C improved in the intervention group."
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

