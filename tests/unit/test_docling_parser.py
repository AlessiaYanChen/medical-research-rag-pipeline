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


def test_docling_parser_normalizes_dataframe_export_table(tmp_path: Path) -> None:
    import pandas as pd

    class FakeDoclingTable:
        def export_to_dataframe(self):
            return pd.DataFrame(
                [
                    {"Metric": "Response Rate", "Group A": "81%", "Group B": "67%"},
                    {"Metric": "Adverse Events", "Group A": "9%", "Group B": "14%"},
                ]
            )

    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\n% Dummy test PDF\n")

    parser = DoclingParser(
        document_converter=lambda _: {
            "markdown": "# Results\n\nTable below.",
            "tables": [FakeDoclingTable()],
        }
    )

    parsed = parser.parse(dummy_pdf)

    assert len(parsed.tables) == 1
    table = parsed.tables[0]
    assert table.headers == ["Metric", "Group A", "Group B"]
    assert table.rows == [
        {"Metric": "Response Rate", "Group A": "81%", "Group B": "67%"},
        {"Metric": "Adverse Events", "Group A": "9%", "Group B": "14%"},
    ]


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


def test_docling_parser_cleans_markdown_noise_artifacts(tmp_path: Path) -> None:
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\n% Dummy test PDF\n")

    noisy_markdown = """
<!-- image -->
The Author(s) 2020. Published by Oxford University Press for the Infectious Diseases Society of America. All rights reserved. For permissions...
The Author(s) 2020. Published by Oxford University Press for the Infectious Diseases Society of America. All rights reserved. For permissions...
Methods. Patients with positive blood cultures with Gram stains showing GNB were randomized to SOC testing with antimicrobial stewardship (ASP).
Methods. Patients with positive blood cultures with Gram stains showing GNB were randomized to SOC testing with antimicrobial stewardship (ASP).
Within the fi rst 4 days after enrollment, duration of vancomycin was not different between groups.
"""

    parser = DoclingParser(
        document_converter=lambda _: {
            "markdown": noisy_markdown,
            "tables": [],
        }
    )

    parsed = parser.parse(dummy_pdf)

    assert "<!-- image -->" not in parsed.markdown_text
    assert parsed.markdown_text.count("Published by Oxford University Press") == 1
    assert parsed.markdown_text.count("randomized to SOC testing") == 1
    assert "fi rst" not in parsed.markdown_text
    assert "first 4 days" in parsed.markdown_text


def test_docling_parser_normalizes_opening_structured_abstract(tmp_path: Path) -> None:
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\n% Dummy test PDF\n")

    markdown = """
## Trial Title

Background. Rapid diagnostics were evaluated in a randomized study.

Methods. Patients were randomized to control or intervention arms.

Results. Rapid testing shortened time to therapy change.

Conclusions. Rapid testing improved antibiotic modification timing.

## METHODS

Body section starts here.
"""

    parser = DoclingParser(
        document_converter=lambda _: {
            "markdown": markdown,
            "tables": [],
        }
    )

    parsed = parser.parse(dummy_pdf)

    assert "## Structured Abstract" in parsed.markdown_text
    assert parsed.markdown_text.count("## Structured Abstract") == 1
    assert "## METHODS" in parsed.markdown_text


def test_docling_parser_strips_inline_numeric_citation_runs(tmp_path: Path) -> None:
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\n% Dummy test PDF\n")

    markdown = """
## DISCUSSION

Rapid testing enabled faster modifications [21]. Notably, RAPID enabled escalation sooner for resistant infections [6, 22]. This remained clinically meaningful.
"""

    parser = DoclingParser(
        document_converter=lambda _: {
            "markdown": markdown,
            "tables": [],
        }
    )

    parsed = parser.parse(dummy_pdf)

    assert "[21]" not in parsed.markdown_text
    assert "[6, 22]" not in parsed.markdown_text
    assert "Rapid testing enabled faster modifications." in parsed.markdown_text
    assert "Notably, RAPID enabled escalation sooner for resistant infections." in parsed.markdown_text
    assert "This remained clinically meaningful." in parsed.markdown_text
