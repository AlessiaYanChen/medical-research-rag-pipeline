from __future__ import annotations

from pathlib import Path

import pytest

from src.app.ingestion.doc_id_utils import doc_id_from_path, normalize_doc_id


def test_doc_id_from_path_uses_pdf_stem_without_changing_existing_style() -> None:
    assert doc_id_from_path(Path("data/raw_pdfs/uploaded/Single site RCT.pdf")) == "Single site RCT"


def test_normalize_doc_id_collapses_extra_whitespace() -> None:
    assert normalize_doc_id("  Single   site   RCT  ") == "Single site RCT"


def test_normalize_doc_id_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_doc_id("   ")
