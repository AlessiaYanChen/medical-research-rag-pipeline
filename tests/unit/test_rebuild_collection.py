from __future__ import annotations

import json
from pathlib import Path

from scripts.rebuild_collection import build_failure_record, write_failure_report


def test_build_failure_record_uses_explicit_doc_id_and_error_message(tmp_path: Path) -> None:
    pdf_path = tmp_path / "study.pdf"
    pdf_path.write_text("placeholder", encoding="utf-8")

    failure = build_failure_record(
        pdf_path=pdf_path,
        doc_id="DOC-1",
        stage="rebuild_document",
        error=RuntimeError("parse failed"),
    )

    assert failure == {
        "pdf_path": str(pdf_path),
        "doc_id": "DOC-1",
        "stage": "rebuild_document",
        "error": "parse failed",
    }


def test_build_failure_record_leaves_doc_id_blank_when_unknown(tmp_path: Path) -> None:
    pdf_path = tmp_path / "study.pdf"
    pdf_path.write_text("placeholder", encoding="utf-8")

    failure = build_failure_record(
        pdf_path=pdf_path,
        stage="rebuild_document",
        error=ValueError("bad filename"),
    )

    assert failure["doc_id"] == ""
    assert failure["error"] == "bad filename"


def test_write_failure_report_writes_summary_payload(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "rebuild_failures.json"
    failures = [
        {
            "pdf_path": "C:/docs/a.pdf",
            "doc_id": "DOC-A",
            "stage": "rebuild_document",
            "error": "parse failed",
        },
        {
            "pdf_path": "C:/docs/b.pdf",
            "doc_id": "",
            "stage": "rebuild_document",
            "error": "bad filename",
        },
    ]

    written_path = write_failure_report(
        output_path=output_path,
        collection="medical_research_chunks_v1",
        pdf_dir="data/raw_pdfs/uploaded",
        failures=failures,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert written_path == output_path
    assert payload["collection"] == "medical_research_chunks_v1"
    assert payload["pdf_dir"] == "data/raw_pdfs/uploaded"
    assert payload["failure_count"] == 2
    assert payload["failures"] == failures
