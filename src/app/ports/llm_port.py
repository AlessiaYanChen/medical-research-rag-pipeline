from __future__ import annotations

from abc import ABC, abstractmethod


class LLMPort(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from a prompt."""

