from pathlib import Path

from src.adapters.parsing.marker_parser import MarkerParser


def test_marker_parser_separates_tables_from_main_text(tmp_path: Path) -> None:
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\n% Dummy test PDF\n")

    markdown_with_table = """# Study Results

This trial compares treatment outcomes.

| Metric | Group A | Group B |
| --- | --- | --- |
| Response Rate | 81% | 67% |
| Adverse Events | 9% | 14% |

Final interpretation is based on statistically significant differences.
"""

    parser = MarkerParser(markdown_converter=lambda _: markdown_with_table)
    parsed = parser.parse(dummy_pdf)

    assert parsed.source_path == dummy_pdf
    assert "This trial compares treatment outcomes." in parsed.markdown_text
    assert "Final interpretation is based on statistically significant differences." in parsed.markdown_text

    # Main text should not contain raw markdown table lines.
    assert "| Metric | Group A | Group B |" not in parsed.markdown_text
    assert "| Response Rate | 81% | 67% |" not in parsed.markdown_text

    assert len(parsed.tables) == 1
    table = parsed.tables[0]
    assert table.headers == ["Metric", "Group A", "Group B"]
    assert table.rows == [
        {"Metric": "Response Rate", "Group A": "81%", "Group B": "67%"},
        {"Metric": "Adverse Events", "Group A": "9%", "Group B": "14%"},
    ]
    assert "Metric,Group A,Group B" in table.csv
    assert "Response Rate,81%,67%" in table.csv

