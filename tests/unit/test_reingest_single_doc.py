from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.reingest_single_doc import (
    build_failure_record,
    load_manifest_json_object,
    resolve_failure_report_path,
    write_failure_report,
)


def test_reingest_build_failure_record_includes_stage_and_collection(tmp_path: Path) -> None:
    pdf_path = tmp_path / "study.pdf"
    pdf_path.write_text("placeholder", encoding="utf-8")

    failure = build_failure_record(
        pdf_path=pdf_path,
        doc_id="DOC-7",
        collection="medical_research_chunks_v1",
        stage="parse",
        error=RuntimeError("parse failed"),
    )

    assert failure == {
        "pdf_path": str(pdf_path),
        "doc_id": "DOC-7",
        "collection": "medical_research_chunks_v1",
        "stage": "parse",
        "error": "parse failed",
    }


def test_reingest_write_failure_report_writes_single_failure_payload(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "reingest_failure.json"
    failure = {
        "pdf_path": "C:/docs/study.pdf",
        "doc_id": "DOC-7",
        "collection": "medical_research_chunks_v1",
        "stage": "manifest_update",
        "error": "manifest write failed",
    }

    written_path = write_failure_report(
        output_path=output_path,
        failure=failure,
        manifest_path="data/ingestion_manifests/medical_research_chunks_v1_rebuild_manifest.json",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert written_path == output_path
    assert payload["doc_id"] == "DOC-7"
    assert payload["collection"] == "medical_research_chunks_v1"
    assert payload["pdf_path"] == "C:/docs/study.pdf"
    assert payload["manifest_path"] == "data/ingestion_manifests/medical_research_chunks_v1_rebuild_manifest.json"
    assert payload["failure"] == failure


def test_reingest_resolve_failure_report_path_uses_default_collection_and_doc_id() -> None:
    resolved = resolve_failure_report_path(
        output_path="",
        collection="medical_research_chunks_v1",
        doc_id="DOC-7",
    )

    assert resolved == Path("data/eval/results/reingest_failure_medical_research_chunks_v1_DOC-7.json")


def test_reingest_resolve_failure_report_path_sanitizes_doc_id() -> None:
    resolved = resolve_failure_report_path(
        output_path="",
        collection="medical_research_chunks_v1",
        doc_id="DOC:7/alpha",
    )

    assert resolved == Path("data/eval/results/reingest_failure_medical_research_chunks_v1_DOC_7_alpha.json")


def test_load_manifest_json_object_reads_object_payload(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"collection": "medical_research_chunks_v1"}), encoding="utf-8")

    payload = load_manifest_json_object(manifest_path)

    assert payload == {"collection": "medical_research_chunks_v1"}


def test_load_manifest_json_object_rejects_invalid_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match=r"manifest is not valid JSON: .*line 1, column 2"):
        load_manifest_json_object(manifest_path)


def test_load_manifest_json_object_rejects_non_object_payload(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest is not a JSON object"):
        load_manifest_json_object(manifest_path)
