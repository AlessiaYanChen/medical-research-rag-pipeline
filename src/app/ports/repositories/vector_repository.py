from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.domain.models.chunk import Chunk


@dataclass(frozen=True)
class MetadataFilter:
    key: str
    value: Any | None = None
    values: tuple[Any, ...] = ()


@dataclass(frozen=True)
class VectorSearchFilters:
    doc_id: str | None = None
    must: tuple[MetadataFilter, ...] = ()
    must_not: tuple[MetadataFilter, ...] = ()
    should: tuple[MetadataFilter, ...] = ()
    minimum_should_match: int = 1


class VectorRepositoryPort(ABC):
    @abstractmethod
    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Upsert chunks into a vector store."""

    @abstractmethod
    def search(
        self,
        vector: list[float],
        doc_id: str | None = None,
        limit: int = 5,
        filters: VectorSearchFilters | None = None,
    ) -> list[Chunk]:
        """Search for chunks by vector similarity, optionally scoped by metadata filters."""
