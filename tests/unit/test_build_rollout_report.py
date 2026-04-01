from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_rollout_report import (
    FAIL_STATUS,
    PASS_STATUS,
    REVIEW_REQUIRED_STATUS,
    build_eval_gate,
    build_manual_spot_check_gate,
    build_rebuild_gate,
    build_rollout_report,
    determine_overall_status,
    load_manual_spot_checks,
    resolve_audit_path,
    resolve_json_output_path,
    resolve_manifest_path,
    resolve_markdown_output_path,
    resolve_rebuild_failure_path,
)


def test_resolve_paths_use_collection_specific_defaults() -> None:
    collection = "medical_research_chunks_docling_v2_batch1"

    assert resolve_manifest_path(collection, "") == Path(
        "data/ingestion_manifests/medical_research_chunks_docling_v2_batch1_rebuild_manifest.json"
    )
    assert resolve_rebuild_failure_path(collection, "") == Path(
        "data/eval/results/rebuild_failures_medical_research_chunks_docling_v2_batch1.json"
    )
    assert resolve_audit_path(collection, "") == Path(
        "data/eval/results/collection_audit_medical_research_chunks_docling_v2_batch1.json"
    )
    assert resolve_json_output_path(collection, "") == Path(
        "data/eval/results/rollout_report_medical_research_chunks_docling_v2_batch1.json"
    )
    assert resolve_markdown_output_path(collection, "") == Path(
        "data/eval/results/rollout_report_medical_research_chunks_docling_v2_batch1.md"
    )


def test_build_rebuild_gate_passes_when_manifest_matches_and_no_failures() -> None:
    gate = build_rebuild_gate(
        collection="medical_research_chunks_docling_v2_batch1",
        manifest_payload={
            "collection": "medical_research_chunks_docling_v2_batch1",
            "doc_count": 20,
            "chunk_count": 4000,
            "parser": "docling",
        },
        manifest_path=Path("manifest.json"),
        failure_payload=None,
        failure_path=Path("failures.json"),
        target_pdf_count=20,
    )

    assert gate["status"] == PASS_STATUS
    assert gate["doc_count"] == 20
    assert gate["failure_count"] == 0


def test_build_rebuild_gate_fails_on_target_count_mismatch() -> None:
    gate = build_rebuild_gate(
        collection="medical_research_chunks_docling_v2_batch1",
        manifest_payload={
            "collection": "medical_research_chunks_docling_v2_batch1",
            "doc_count": 18,
            "chunk_count": 4000,
            "parser": "docling",
        },
        manifest_path=Path("manifest.json"),
        failure_payload=None,
        failure_path=Path("failures.json"),
        target_pdf_count=20,
    )

    assert gate["status"] == FAIL_STATUS
    assert "Target PDF count mismatch" in gate["issues"][0]


def test_build_eval_gate_requires_review_without_baseline() -> None:
    gate = build_eval_gate(
        label="runtime",
        candidate_payload={
            "summary": {
                "expected_doc_hit_rate": 1.0,
                "expected_header_hit_rate": 1.0,
                "top1_expected_doc_hit_rate": 1.0,
                "top1_expected_header_hit_rate": 0.82,
                "average_doc_precision": 0.95,
                "average_header_precision": 0.81,
                "cross_document_average_doc_precision": 0.95,
            }
        },
        candidate_path=Path("runtime.json"),
        baseline_payload=None,
        baseline_path=None,
        max_metric_drop=0.02,
    )

    assert gate["status"] == REVIEW_REQUIRED_STATUS
    assert "No baseline provided" in gate["issues"][0]


def test_build_eval_gate_fails_when_metric_drop_exceeds_tolerance() -> None:
    gate = build_eval_gate(
        label="stable",
        candidate_payload={
            "summary": {
                "expected_doc_hit_rate": 0.97,
                "expected_header_hit_rate": 0.98,
                "top1_expected_doc_hit_rate": 0.97,
                "top1_expected_header_hit_rate": 0.94,
                "average_doc_precision": 0.95,
                "average_header_precision": 0.94,
                "cross_document_average_doc_precision": 0.95,
            }
        },
        candidate_path=Path("stable.json"),
        baseline_payload={
            "summary": {
                "expected_doc_hit_rate": 1.0,
                "expected_header_hit_rate": 1.0,
                "top1_expected_doc_hit_rate": 1.0,
                "top1_expected_header_hit_rate": 0.95,
                "average_doc_precision": 0.96,
                "average_header_precision": 0.95,
                "cross_document_average_doc_precision": 0.96,
            }
        },
        baseline_path=Path("baseline_stable.json"),
        max_metric_drop=0.02,
    )

    assert gate["status"] == FAIL_STATUS
    assert any("expected_doc_hit_rate" in issue for issue in gate["issues"])


def test_load_manual_spot_checks_accepts_object_wrapper(tmp_path: Path) -> None:
    path = tmp_path / "manual_checks.json"
    path.write_text(
        """
        {
          "checks": [
            {
              "query": "What is the reported turnaround time?",
              "status": "pass",
              "observed": "Returned the opening-body FLAT timing chunk.",
              "expected": "Prefer the within-an-hour FLAT evidence.",
              "repeated": true
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    checks = load_manual_spot_checks(path)

    assert checks == [
        {
            "query": "What is the reported turnaround time?",
            "status": "pass",
            "observed": "Returned the opening-body FLAT timing chunk.",
            "expected": "Prefer the within-an-hour FLAT evidence.",
            "repeated": True,
        }
    ]


def test_build_manual_spot_check_gate_fails_when_any_check_fails(tmp_path: Path) -> None:
    path = tmp_path / "manual_checks.json"
    path.write_text(
        """
        [
          {
            "query": "Which paper should I read for blood-culture turnaround improvements, not stewardship policy?",
            "status": "pass"
          },
          {
            "query": "What themes across these papers suggest that rapid diagnostics improve antimicrobial decision-making more reliably than they improve hard clinical outcomes?",
            "status": "fail"
          }
        ]
        """,
        encoding="utf-8",
    )

    gate = build_manual_spot_check_gate(path)

    assert gate["status"] == FAIL_STATUS
    assert gate["checks_total"] == 2
    assert gate["fail_count"] == 1


def test_determine_overall_status_prefers_fail_over_review_required() -> None:
    assert determine_overall_status([{"status": PASS_STATUS}, {"status": REVIEW_REQUIRED_STATUS}]) == REVIEW_REQUIRED_STATUS
    assert determine_overall_status([{"status": PASS_STATUS}, {"status": FAIL_STATUS}]) == FAIL_STATUS


def test_build_rollout_report_combines_gate_states() -> None:
    report = build_rollout_report(
        collection="medical_research_chunks_docling_v2_batch1",
        stage_label="stage-1-20-pdfs",
        target_pdf_count=20,
        max_metric_drop=0.02,
        manifest_payload={
            "collection": "medical_research_chunks_docling_v2_batch1",
            "doc_count": 20,
            "chunk_count": 4000,
            "parser": "docling",
        },
        manifest_path=Path("manifest.json"),
        rebuild_failure_payload=None,
        rebuild_failure_path=Path("failures.json"),
        audit_payload={
            "issue_count": 0,
            "cleanup_plan_count": 0,
            "manifest_version_issues": [],
        },
        audit_path=Path("audit.json"),
        eval_inputs=[
            {
                "label": "stable",
                "candidate_path": Path("stable.json"),
                "candidate_payload": {
                    "summary": {
                        "expected_doc_hit_rate": 1.0,
                        "expected_header_hit_rate": 1.0,
                        "top1_expected_doc_hit_rate": 1.0,
                        "top1_expected_header_hit_rate": 1.0,
                        "average_doc_precision": 0.99,
                        "average_header_precision": 0.99,
                        "cross_document_average_doc_precision": 0.99,
                    }
                },
                "baseline_path": Path("stable_baseline.json"),
                "baseline_payload": {
                    "summary": {
                        "expected_doc_hit_rate": 1.0,
                        "expected_header_hit_rate": 1.0,
                        "top1_expected_doc_hit_rate": 1.0,
                        "top1_expected_header_hit_rate": 1.0,
                        "average_doc_precision": 1.0,
                        "average_header_precision": 1.0,
                        "cross_document_average_doc_precision": 1.0,
                    }
                },
            }
        ],
        manual_spot_check_path=None,
    )

    assert report["overall_status"] == FAIL_STATUS
    assert report["rebuild_gate"]["status"] == PASS_STATUS
    assert report["audit_gate"]["status"] == PASS_STATUS
    assert report["evaluation_gates"][0]["status"] == PASS_STATUS
    assert report["manual_spot_check_gate"]["status"] == FAIL_STATUS


def test_load_manual_spot_checks_rejects_invalid_status(tmp_path: Path) -> None:
    path = tmp_path / "manual_checks.json"
    path.write_text('[{"query": "test", "status": "review"}]', encoding="utf-8")

    with pytest.raises(ValueError, match="must use status 'pass' or 'fail'"):
        load_manual_spot_checks(path)
