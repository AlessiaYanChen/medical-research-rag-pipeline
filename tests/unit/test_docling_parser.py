from __future__ import annotations

from pathlib import Path

import pandas as pd
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


def test_docling_parser_drops_structurally_empty_tables(tmp_path: Path) -> None:
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\n% Dummy test PDF\n")

    parser = DoclingParser(
        document_converter=lambda _: {
            "markdown": "# Results\n\nTable below.",
            "tables": [
                {
                    "headers": ["0", "1", "2"],
                    "rows": [
                        {"0": "", "1": "", "2": ""},
                        {"0": " ", "1": "", "2": ""},
                    ],
                },
                {
                    "headers": ["Metric", "Value"],
                    "rows": [
                        {"Metric": "Sensitivity", "Value": "94%"},
                    ],
                },
            ],
        }
    )

    parsed = parser.parse(dummy_pdf)

    assert len(parsed.tables) == 1
    assert parsed.tables[0].headers == ["Metric", "Value"]
    assert parsed.tables[0].rows == [{"Metric": "Sensitivity", "Value": "94%"}]


def test_docling_parser_prefers_markdown_table_order_when_structured_tables_are_duplicated(tmp_path: Path) -> None:
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\n% Dummy test PDF\n")

    markdown = """
# Results

| Metric | Value |
| --- | --- |
| Sensitivity | 94% |

| Organism | Count |
| --- | --- |
| E. coli | 7 |
"""

    parser = DoclingParser(
        document_converter=lambda _: {
            "markdown": markdown,
            "tables": [
                {
                    "headers": ["Metric", "Value"],
                    "rows": [{"Metric": "Sensitivity", "Value": "94%"}],
                },
                {
                    "headers": ["Metric", "Value"],
                    "rows": [{"Metric": "Sensitivity", "Value": "94%"}],
                },
            ],
        }
    )

    parsed = parser.parse(dummy_pdf)

    assert len(parsed.tables) == 2
    assert parsed.tables[0].headers == ["Metric", "Value"]
    assert parsed.tables[0].rows == [{"Metric": "Sensitivity", "Value": "94%"}]
    assert parsed.tables[1].headers == ["Organism", "Count"]
    assert parsed.tables[1].rows == [{"Organism": "E. coli", "Count": "7"}]


def test_docling_parser_recovers_pathological_lod_table_from_page_text_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCell:
        def __init__(self) -> None:
            self.column_header = False

    class FakeData:
        num_cols = 11
        table_cells = [FakeCell()]

    class FakeProv:
        page_no = 5

    class FakeDoclingTable:
        data = FakeData()
        prov = [FakeProv()]

        def export_to_dataframe(self):
            return pd.DataFrame(
                [
                    ["LOD without lysozyme treatment Gram-positive", "Gram-negative E. coli", "10 2", "LOD with lysozyme treatment"],
                    ["", "E. faecalis", "10 5", "LOD with lysozyme treatment"],
                ]
            )

    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\n% Dummy test PDF\n")

    page_text = """
                                                                     Table   2.   LOD determination using contrived samples            (CFU/L).a

                                                                                           LOD without lysozyme treatment

                                                                                                   Gram-positive

                                                                                                                                                                               Gram-negative

             Test strain                         S. aureus                        S.  epidermidis                       E. avium                  E. faecalis                  E. coli

             CFU/\u7121                              104                                  105                                 104                          105                     102

                                                                                           LOD with lysozyme treatment

                                                                                                           Gram-positive

                                                                                                                                                                          Gram-negative

             Test strain                                         S.  aureus                S. epidermidis                    E. avium                  E.  faecalis            E. coli

             Incubation (mins)                              30               60       30           60                   30             60         30                 60   30            60

             Amount of lysozyme (\u75e2/mL)

             10 (CFU/\u7121)                                    104              104      104          104                  104            104        105                104  102           102

             50 (CFU/\u7121)                                    104              104      104          104                  104            104        104                104  102           102

             100b (CFU/\u7121)                                  102              102      103          103                  103            102        103                103  102           102

             500 (CFU/\u7121)                                   102              102      103          103                  103            102        103                103  102           102

             1000 (CFU/\u7121)                                  102              102      103          103                  102            102        103                103  102           102
"""

    monkeypatch.setattr(
        DoclingParser,
        "_extract_pdftotext_page_text",
        staticmethod(lambda **_: page_text),
    )

    parser = DoclingParser(
        document_converter=lambda _: {
            "markdown": "# Results\n\nLOD summary.",
            "tables": [FakeDoclingTable()],
        }
    )

    parsed = parser.parse(dummy_pdf)

    assert len(parsed.tables) == 1
    table = parsed.tables[0]
    assert table.headers == [
        "Test strain",
        "LOD without lysozyme treatment (CFU/\u00b5L)",
        "10 \u00b5g/mL 30 mins (CFU/\u00b5L)",
        "10 \u00b5g/mL 60 mins (CFU/\u00b5L)",
        "50 \u00b5g/mL 30 mins (CFU/\u00b5L)",
        "50 \u00b5g/mL 60 mins (CFU/\u00b5L)",
        "100 \u00b5g/mL 30 mins (CFU/\u00b5L)",
        "100 \u00b5g/mL 60 mins (CFU/\u00b5L)",
        "500 \u00b5g/mL 30 mins (CFU/\u00b5L)",
        "500 \u00b5g/mL 60 mins (CFU/\u00b5L)",
        "1000 \u00b5g/mL 30 mins (CFU/\u00b5L)",
        "1000 \u00b5g/mL 60 mins (CFU/\u00b5L)",
    ]
    assert table.rows[0]["Test strain"] == "S. aureus"
    assert table.rows[0]["LOD without lysozyme treatment (CFU/\u00b5L)"] == "104"
    assert table.rows[0]["100 \u00b5g/mL 60 mins (CFU/\u00b5L)"] == "102"
    assert table.rows[-1]["Test strain"] == "E. coli"
    assert table.rows[-1]["500 \u00b5g/mL 30 mins (CFU/\u00b5L)"] == "102"
