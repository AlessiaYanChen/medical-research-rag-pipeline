from __future__ import annotations

import pytest

from src.adapters.parsing.docling_parser import DoclingParser
from src.adapters.parsing.marker_parser import MarkerParser
from src.app.ingestion.parser_factory import (
    DEFAULT_PARSER_NAME,
    PARSER_CHOICES,
    build_parser,
)


def test_parser_factory_defaults_to_marker() -> None:
    assert DEFAULT_PARSER_NAME == "marker"
    assert PARSER_CHOICES == ("marker", "docling")
    assert isinstance(build_parser(DEFAULT_PARSER_NAME), MarkerParser)


def test_parser_factory_builds_docling() -> None:
    assert isinstance(build_parser("docling"), DoclingParser)


def test_parser_factory_rejects_unknown_parser() -> None:
    with pytest.raises(ValueError, match="Unsupported parser"):
        build_parser("unknown")
