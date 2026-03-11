from __future__ import annotations

import json
from pathlib import Path

from src.app.tables.table_chunker import UnifiedChunker
from src.domain.models.chunk import Chunk


def test_unified_chunker_keeps_table_atomic_and_preserves_parent_header() -> None:
    markdown = """# Introduction

This study explores lipidomics biomarkers in sepsis cohorts.

## Results

The following table summarizes the primary lipid markers.

| Marker | Case | Control |
| --- | --- | --- |
| LDL-C | 126 | 101 |
| HDL-C | 52 | 61 |

Interpretation indicates a stable separation between groups.

See details on page 7 for expanded stratification notes.
"""
    tables = [
        {
            "csv": "Marker,Case,Control\nLDL-C,126,101\nHDL-C,52,61",
            "rows": [
                {"Marker": "LDL-C", "Case": "126", "Control": "101"},
                {"Marker": "HDL-C", "Case": "52", "Control": "61"},
            ],
            "page_number": 6,
        }
    ]

    chunker = UnifiedChunker(max_chars=120, overlap_paragraphs=1)
    chunks = chunker.chunk_document(
        doc_id="DOC-001",
        source_file="RAPID.pdf",
        markdown_text=markdown,
        tables=tables,
    )

    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.id for chunk in chunks)

    table_chunks = [chunk for chunk in chunks if chunk.metadata.chunk_type == "table"]
    assert len(table_chunks) == 1
    table_chunk = table_chunks[0]
    assert "Source File: RAPID.pdf" in table_chunk.content
    assert "Table Index: 1" in table_chunk.content
    assert "Section: Results" in table_chunk.content
    assert "Marker,Case,Control" in table_chunk.content
    assert table_chunk.metadata.parent_header == "Results"
    assert table_chunk.metadata.page_number == 6

    text_chunks = [chunk for chunk in chunks if chunk.metadata.chunk_type == "text"]
    assert any(
        "The following table summarizes the primary lipid markers." in chunk.content
        and chunk.metadata.parent_header == "Results"
        for chunk in text_chunks
    )
    assert any(
        "Interpretation indicates a stable separation between groups." in chunk.content
        and chunk.metadata.parent_header == "Results"
        for chunk in text_chunks
    )


def test_unified_chunker_loads_sibling_table_artifacts_from_document_path(tmp_path: Path) -> None:
    data_subdir = tmp_path / "data" / "marker_markdown"
    data_subdir.mkdir(parents=True, exist_ok=True)

    doc_stem = "RAPID"
    md_path = data_subdir / f"{doc_stem}.main_text.md"
    md_path.write_text(
        "# Results\n\n"
        "Narrative paragraph before finalization.\n",
        encoding="utf-8",
    )

    (data_subdir / f"{doc_stem}.table_01.csv").write_text(
        "Marker,Case,Control\nLDL-C,126,101",
        encoding="utf-8",
    )
    (data_subdir / f"{doc_stem}.table_01.json").write_text(
        json.dumps({"rows": [{"Marker": "LDL-C", "Case": "126", "Control": "101"}], "page_number": 4}),
        encoding="utf-8",
    )

    markdown_text = md_path.read_text(encoding="utf-8")
    chunker = UnifiedChunker(max_chars=120, overlap_paragraphs=1)
    chunks = chunker.chunk_document(
        doc_id="DOC-002",
        source_file="RAPID.pdf",
        markdown_text=markdown_text,
        tables=None,
        document_path=md_path,
    )

    table_chunks = [chunk for chunk in chunks if chunk.metadata.chunk_type == "table"]
    assert len(table_chunks) == 1
    assert "Table Index: 1" in table_chunks[0].content
    assert "Source File: RAPID.pdf" in table_chunks[0].content
    assert table_chunks[0].metadata.page_number == 4
