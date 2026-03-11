from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParsedTable:
    """Structured representation of an extracted table."""

    headers: list[str]
    rows: list[dict[str, str]]
    csv: str


@dataclass(frozen=True)
class ParsedDocument:
    """Parser output for one document."""

    source_path: Path
    markdown_text: str
    tables: list[ParsedTable]


class ParserPort(ABC):
    """Contract for PDF parsers that separate narrative text from tables."""

    @abstractmethod
    def parse(self, pdf_path: Path | str) -> ParsedDocument:
        """Parse a single PDF and return markdown text plus extracted tables."""

