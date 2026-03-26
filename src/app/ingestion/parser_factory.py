from __future__ import annotations

from typing import Literal

from src.adapters.parsing.docling_parser import DoclingParser
from src.adapters.parsing.marker_parser import MarkerParser
from src.ports.parser_port import ParserPort


ParserName = Literal["marker", "docling"]
DEFAULT_PARSER_NAME: ParserName = "marker"
PARSER_CHOICES: tuple[ParserName, ParserName] = ("marker", "docling")


def build_parser(parser_name: str) -> ParserPort:
    normalized = str(parser_name).strip().lower()
    if normalized == "marker":
        return MarkerParser()
    if normalized == "docling":
        return DoclingParser()
    raise ValueError(f"Unsupported parser: {parser_name}")
