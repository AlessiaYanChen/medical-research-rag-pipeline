from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.models.chunk import Chunk


class ReRankerPort(ABC):
    @abstractmethod
    def rank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        """Return the best-ranked chunks for a query."""

