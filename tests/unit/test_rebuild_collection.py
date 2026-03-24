from __future__ import annotations

import json
import shlex
from pathlib import Path

from scripts.rebuild_collection import (
    build_failure_record,
    parse_args,
    resolve_failure_report_path,
    resolve_manifest_output_path,
    write_failure_report,
)


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


def test_resolve_failure_report_path_uses_default_collection_path() -> None:
    resolved = resolve_failure_report_path(
        output_path="",
        collection="medical_research_chunks_v1",
    )

    assert resolved == Path("data/eval/results/rebuild_failures_medical_research_chunks_v1.json")


def test_resolve_failure_report_path_preserves_explicit_override() -> None:
    resolved = resolve_failure_report_path(
        output_path="custom/rebuild_failures.json",
        collection="medical_research_chunks_v1",
    )

    assert resolved == Path("custom/rebuild_failures.json")


def test_resolve_manifest_output_path_uses_collection_specific_default() -> None:
    resolved = resolve_manifest_output_path(
        output_path="",
        collection="medical_research_chunks_v1",
    )

    assert resolved == Path("data/ingestion_manifests/medical_research_chunks_v1_rebuild_manifest.json")


def test_resolve_manifest_output_path_preserves_explicit_override() -> None:
    resolved = resolve_manifest_output_path(
        output_path="custom/manifest.json",
        collection="medical_research_chunks_v1",
    )

    assert resolved == Path("custom/manifest.json")


def test_parse_args_leaves_manifest_out_blank_until_collection_is_known(monkeypatch) -> None:
    argv = shlex.split("--pdf-dir data/raw_pdfs/uploaded --collection medical_research_chunks_v1")
    monkeypatch.setattr("sys.argv", ["rebuild_collection.py", *argv])

    args = parse_args()

    assert args.manifest_out == ""
