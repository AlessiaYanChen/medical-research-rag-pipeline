from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.parsing.docling_parser import DoclingParser


def test_docling_parser_reads_markdown_only_document(tmp_path: Path) -> None:
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\n% Dummy test PDF\n")

    parser = DoclingParser(
        document_converter=lambda _: {
            "markdown": "# Study\n\nNarrative results only.",
            "tables": [],
        }
    )

    parsed = parser.parse(dummy_pdf)

    assert parsed.source_path == dummy_pdf
    assert parsed.markdown_text == "# Study\n\nNarrative results only."
    assert parsed.tables == []


def test_docling_parser_normalizes_structured_table_payload(tmp_path: Path) -> None:
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\n% Dummy test PDF\n")

    parser = DoclingParser(
        document_converter=lambda _: {
            "document": {
                "markdown_text": "# Results\n\nTable below.",
                "tables": [
                    {
                        "headers": ["Metric", "Group A", "Group B"],
                        "rows": [
                            {"Metric": "Response Rate", "Group A": "81%", "Group B": "67%"},
                            {"Metric": "Adverse Events", "Group A": "9%", "Group B": "14%"},
                        ],
                    }
                ],
            }
        }
    )

    parsed = parser.parse(dummy_pdf)

    assert parsed.markdown_text == "# Results\n\nTable below."
    assert len(parsed.tables) == 1
    table = parsed.tables[0]
    assert table.headers == ["Metric", "Group A", "Group B"]
    assert table.rows == [
        {"Metric": "Response Rate", "Group A": "81%", "Group B": "67%"},
        {"Metric": "Adverse Events", "Group A": "9%", "Group B": "14%"},
    ]
    assert table.csv == (
        "Metric,Group A,Group B\n"
        "Response Rate,81%,67%\n"
        "Adverse Events,9%,14%"
    )


def test_docling_parser_rejects_unsupported_table_format(tmp_path: Path) -> None:
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\n% Dummy test PDF\n")

    parser = DoclingParser(
        document_converter=lambda _: {
            "markdown": "# Results",
            "tables": [object()],
        }
    )

    with pytest.raises(RuntimeError, match="Unsupported Docling table format"):
        parser.parse(dummy_pdf)


def test_docling_parser_rejects_empty_output(tmp_path: Path) -> None:
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\n% Dummy test PDF\n")

    parser = DoclingParser(
        document_converter=lambda _: {
            "markdown": "   ",
            "tables": [],
        }
    )

    with pytest.raises(RuntimeError, match="Docling returned no markdown text or tables"):
        parser.parse(dummy_pdf)
