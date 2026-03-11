from __future__ import annotations

from collections.abc import Callable
import logging
import re

from src.app.ports.re_ranker_port import ReRankerPort
from src.app.ports.repositories.vector_repository import VectorRepositoryPort
from src.domain.models.chunk import Chunk


logger = logging.getLogger(__name__)


class RetrievalService:
    """Application service for semantic retrieval over the active knowledge base."""

    def __init__(
        self,
        repo: VectorRepositoryPort,
        embedding_fn: Callable[[list[str]], list[list[float]]],
        re_ranker: ReRankerPort | None = None,
    ) -> None:
        self._repo = repo
        self._embedding_fn = embedding_fn
        self._re_ranker = re_ranker

    def retrieve(self, query: str, doc_id: str | None = None, limit: int = 5) -> str:
        query_vector = self._embedding_fn([query])[0]
        initial_limit = max(limit * 4, 20)
        initial_chunks = self._repo.search(query_vector, doc_id=doc_id, limit=initial_limit)
        chunks = self._select_final_chunks(query=query, initial_chunks=initial_chunks, limit=limit)

        formatted_chunks = [
            (
                f"Source: {self._clean_markdown(chunk.metadata.parent_header)}"
                f" | Document: {self._clean_markdown(chunk.metadata.doc_id)}\n"
                f"{self._clean_markdown(chunk.content)}"
            )
            for chunk in chunks
        ]
        return "\n\n".join(formatted_chunks)

    def _select_final_chunks(self, query: str, initial_chunks: list[Chunk], limit: int) -> list[Chunk]:
        if self._re_ranker is None:
            return initial_chunks[:limit]

        try:
            return self._re_ranker.rank(query=query, chunks=initial_chunks, top_n=limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Re-ranker failed; falling back to vector order: %s", exc)
            return initial_chunks[:limit]

    @staticmethod
    def _clean_markdown(text: str) -> str:
        cleaned = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"<span[^>]*></span>", "", cleaned)
        cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", cleaned)
        cleaned = re.sub(r"\[([^\]]+)\]\(#page-[^)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"[*_`]+", "", cleaned)
        cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
        return cleaned.strip()
