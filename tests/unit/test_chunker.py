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
    assert all(chunk.metadata.extra["content_role"] == "child" for chunk in text_chunks)
    assert all(chunk.metadata.extra["parent_id"].startswith("DOC-001:P") for chunk in text_chunks)
    assert all(chunk.metadata.extra["parent_content"] for chunk in text_chunks)
    assert any(
        "The following table summarizes the primary lipid markers." in chunk.metadata.extra["parent_content"]
        and chunk.metadata.parent_header == "Results"
        for chunk in text_chunks
    )
    assert any(
        "Interpretation indicates a stable separation between groups." in chunk.metadata.extra["parent_content"]
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


def test_unified_chunker_emits_multiple_child_spans_for_one_parent() -> None:
    markdown = """# Results

Sentence one introduces the cohort. Sentence two describes the intervention. Sentence three reports the main biomarker effect. Sentence four gives the safety outcome.
"""
    chunker = UnifiedChunker(max_chars=400, overlap_paragraphs=0, child_sentence_window=2, child_sentence_overlap=1)

    chunks = chunker.chunk_document(
        doc_id="DOC-003",
        source_file="RAPID.pdf",
        markdown_text=markdown,
        tables=[],
    )

    text_chunks = [chunk for chunk in chunks if chunk.metadata.chunk_type == "text"]
    assert len(text_chunks) == 3
    assert len({chunk.metadata.extra["parent_id"] for chunk in text_chunks}) == 1
    assert text_chunks[0].metadata.extra["child_sentence_start"] == 0
    assert text_chunks[0].metadata.extra["child_sentence_end"] == 2
    assert text_chunks[1].metadata.extra["child_sentence_start"] == 1
    assert text_chunks[1].metadata.extra["child_sentence_end"] == 3
    assert text_chunks[0].content == "Sentence one introduces the cohort. Sentence two describes the intervention."
    assert text_chunks[1].content == "Sentence two describes the intervention. Sentence three reports the main biomarker effect."
    assert "Sentence four gives the safety outcome." in text_chunks[2].content


def test_unified_chunker_marks_reference_sections_as_reference_role() -> None:
    markdown = """# References

Smith J, Doe P, et al. Detection of pathogens in bronchoalveolar lavage. J Clin Microbiol. doi:10.1000/example
"""
    chunker = UnifiedChunker(max_chars=400, overlap_paragraphs=0)

    chunks = chunker.chunk_document(
        doc_id="DOC-004",
        source_file="RAPID.pdf",
        markdown_text=markdown,
        tables=[],
    )

    text_chunks = [chunk for chunk in chunks if chunk.metadata.chunk_type == "text"]
    assert len(text_chunks) >= 1
    assert all(chunk.metadata.extra["content_role"] == "reference" for chunk in text_chunks)
    assert all(chunk.metadata.extra["section_role"] == "references" for chunk in text_chunks)


def test_unified_chunker_normalizes_title_like_opening_header_to_document_metadata_abstract() -> None:
    markdown = """# Blood Culture Negative Endocarditis: A Review of Laboratory Diagnostic Approaches

This opening block behaves like title and abstract material rather than a stable section label.

## Discussion

Discussion evidence follows after the opening summary.
"""
    chunker = UnifiedChunker(max_chars=400, overlap_paragraphs=0)

    chunks = chunker.chunk_document(
        doc_id="DOC-005",
        source_file="review.pdf",
        markdown_text=markdown,
        tables=[],
    )

    text_chunks = [chunk for chunk in chunks if chunk.metadata.chunk_type == "text"]
    assert text_chunks[0].metadata.parent_header == UnifiedChunker.DEFAULT_OPENING_HEADER
    assert text_chunks[0].metadata.extra["section_role"] == "body"
    assert text_chunks[0].metadata.extra["original_parent_header"] == "Blood Culture Negative Endocarditis: A Review of Laboratory Diagnostic Approaches"
    assert text_chunks[0].metadata.extra["normalized_parent_header"] == UnifiedChunker.DEFAULT_OPENING_HEADER
    assert text_chunks[0].metadata.extra["header_role"] == "title_like"
    assert text_chunks[1].metadata.parent_header == "Discussion"


def test_unified_chunker_collapses_subsection_header_to_active_structural_header() -> None:
    markdown = """# Results

Lead results paragraph.

## Clinical Validation

Clinical validation details remain under the Results umbrella for retrieval.
"""
    chunker = UnifiedChunker(max_chars=400, overlap_paragraphs=0)

    chunks = chunker.chunk_document(
        doc_id="DOC-006",
        source_file="study.pdf",
        markdown_text=markdown,
        tables=[],
    )

    text_chunks = [chunk for chunk in chunks if chunk.metadata.chunk_type == "text"]
    subsection_chunk = text_chunks[1]
    assert subsection_chunk.metadata.parent_header == "Results"
    assert subsection_chunk.metadata.extra["original_parent_header"] == "Clinical Validation"
    assert subsection_chunk.metadata.extra["normalized_parent_header"] == "Results"
    assert subsection_chunk.metadata.extra["header_role"] == "subsection"


def test_unified_chunker_marks_citation_like_header_as_low_value() -> None:
    markdown = """# Discussion

Main discussion summary.

## Clinical Infectious Diseases 2015;61(7):1071-80

Editorial or citation-like content under a citation-style header.
"""
    chunker = UnifiedChunker(max_chars=400, overlap_paragraphs=0)

    chunks = chunker.chunk_document(
        doc_id="DOC-007",
        source_file="study.pdf",
        markdown_text=markdown,
        tables=[],
    )

    text_chunks = [chunk for chunk in chunks if chunk.metadata.chunk_type == "text"]
    citation_chunk = text_chunks[1]
    assert citation_chunk.metadata.parent_header == "Discussion"
    assert citation_chunk.metadata.extra["original_parent_header"] == "Clinical Infectious Diseases 2015;61(7):1071-80"
    assert citation_chunk.metadata.extra["header_role"] == "citation_like"
    assert citation_chunk.metadata.extra["is_low_value"] is True
