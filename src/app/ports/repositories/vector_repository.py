from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.models.chunk import Chunk


class VectorRepositoryPort(ABC):
    @abstractmethod
    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Upsert chunks into a vector store."""

    @abstractmethod
    def search(self, vector: list[float], doc_id: str | None = None, limit: int = 5) -> list[Chunk]:
        """Search for chunks by vector similarity, optionally scoped to a document."""
