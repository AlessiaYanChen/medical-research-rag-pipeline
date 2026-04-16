from __future__ import annotations

from src.app.ingestion.reconciliation_utils import (
    build_reconciliation_repair_plan,
    build_duplicate_cleanup_plan,
    build_manifest_doc_identities,
    build_manifest_doc_summary,
    build_qdrant_doc_identities,
    build_qdrant_doc_summary,
    build_registry_doc_identities,
    build_registry_doc_summary,
    find_doc_metadata_mismatch_issues,
    find_chunk_count_sanity_issues,
    reconcile_collection_state,
)


def test_build_qdrant_doc_summary_counts_text_and_table_records() -> None:
    summary = build_qdrant_doc_summary(
        [
            {"doc_id": "DOC-1", "chunk_type": "text"},
            {"doc_id": "DOC-1", "chunk_type": "table"},
            {"doc_id": "DOC-1", "chunk_type": "text"},
        ]
    )

    assert summary["DOC-1"] == {
        "chunks": 3,
        "text_chunks": 2,
        "table_chunks": 1,
    }


def test_build_manifest_doc_summary_reads_manifest_shape() -> None:
    summary = build_manifest_doc_summary(
        {
            "docs": [
                {
                    "doc_id": "DOC-1",
                    "chunk_count": 4,
                    "text_chunk_count": 3,
                    "table_chunk_count": 1,
                }
            ]
        }
    )

    assert summary["DOC-1"]["chunks"] == 4
    assert summary["DOC-1"]["text_chunks"] == 3
    assert summary["DOC-1"]["table_chunks"] == 1


def test_build_manifest_doc_identities_reads_manifest_shape() -> None:
    identities = build_manifest_doc_identities(
        {
            "docs": [
                {
                    "doc_id": "DOC-1",
                    "source_file": "doc1.pdf",
                    "local_file": "C:/docs/doc1.pdf",
                }
            ]
        }
    )

    assert identities == [
        {
            "doc_id": "DOC-1",
            "source_file": "doc1.pdf",
            "local_file": "C:/docs/doc1.pdf",
            "source_sha256": "",
        }
    ]


def test_build_registry_doc_summary_reads_nested_registry_shape() -> None:
    summary = build_registry_doc_summary(
        {
            "docs": {
                "DOC-1": {
                    "chunks": 4,
                    "text_chunks": 3,
                    "table_chunks": 1,
                }
            }
        }
    )

    assert summary["DOC-1"]["chunks"] == 4


def test_build_registry_doc_identities_reads_nested_registry_shape() -> None:
    identities = build_registry_doc_identities(
        {
            "docs": {
                "DOC-1": {
                    "doc_id": "DOC-1",
                    "source_file": "doc1.pdf",
                    "pdf_path": "C:/docs/doc1.pdf",
                }
            }
        }
    )

    assert identities == [
        {
            "doc_id": "DOC-1",
            "source_file": "doc1.pdf",
            "local_file": "C:/docs/doc1.pdf",
            "source_sha256": "",
        }
    ]


def test_build_qdrant_doc_identities_reads_payload_metadata() -> None:
    identities = build_qdrant_doc_identities(
        [
            {
                "doc_id": "DOC-1",
                "payload": {
                    "source_file": "doc1.pdf",
                    "local_file": "C:/docs/doc1.pdf",
                    "source_sha256": "hash-1",
                },
            },
            {
                "doc_id": "DOC-1",
                "payload": {},
            },
        ]
    )

    assert identities == [
        {
            "doc_id": "DOC-1",
            "source_file": "doc1.pdf",
            "local_file": "C:/docs/doc1.pdf",
            "source_sha256": "hash-1",
        }
    ]


def test_reconcile_collection_state_reports_missing_and_count_mismatch() -> None:
    audit = reconcile_collection_state(
        qdrant_docs={
            "DOC-1": {"chunks": 4, "text_chunks": 3, "table_chunks": 1},
            "DOC-2": {"chunks": 5, "text_chunks": 5, "table_chunks": 0},
        },
        manifest_docs={
            "DOC-1": {"chunks": 4, "text_chunks": 3, "table_chunks": 1},
        },
        registry_docs={
            "DOC-1": {"chunks": 3, "text_chunks": 2, "table_chunks": 1},
        },
        qdrant_identities=[],
        manifest_identities=[],
        registry_identities=[],
    )

    assert audit["issue_count"] == 2
    assert audit["issue_types"] == {"count_mismatch": 1, "missing_doc": 1}


def test_reconcile_collection_state_reports_duplicate_identity_conflicts() -> None:
    audit = reconcile_collection_state(
        qdrant_docs={
            "DOC-1": {"chunks": 4, "text_chunks": 3, "table_chunks": 1},
            "DOC-2": {"chunks": 5, "text_chunks": 5, "table_chunks": 0},
        },
        manifest_docs={
            "DOC-1": {"chunks": 4, "text_chunks": 3, "table_chunks": 1},
            "DOC-2": {"chunks": 5, "text_chunks": 5, "table_chunks": 0},
        },
        registry_docs={
            "DOC-1": {"chunks": 4, "text_chunks": 3, "table_chunks": 1},
            "DOC-2": {"chunks": 5, "text_chunks": 5, "table_chunks": 0},
        },
        qdrant_identities=[],
        manifest_identities=[
            {
                "doc_id": "DOC-1",
                "source_file": "shared.pdf",
                "local_file": "C:/docs/one.pdf",
                "source_sha256": "",
            },
            {
                "doc_id": "DOC-2",
                "source_file": "shared.pdf",
                "local_file": "C:/docs/two.pdf",
                "source_sha256": "",
            },
        ],
        registry_identities=[
            {
                "doc_id": "DOC-1",
                "source_file": "doc1.pdf",
                "local_file": "C:/docs/shared.pdf",
                "source_sha256": "",
            },
            {
                "doc_id": "DOC-2",
                "source_file": "doc2.pdf",
                "local_file": "C:/docs/shared.pdf",
                "source_sha256": "",
            },
        ],
    )

    duplicate_issues = [issue for issue in audit["issues"] if issue["issue_type"] == "duplicate_identity"]

    assert audit["issue_count"] == 4
    assert audit["issue_types"] == {"duplicate_identity": 2, "metadata_mismatch": 2}
    assert duplicate_issues == [
        {
            "issue_type": "duplicate_identity",
            "source": "manifest",
            "field": "source_file",
            "value": "shared.pdf",
            "doc_ids": ["DOC-1", "DOC-2"],
        },
        {
            "issue_type": "duplicate_identity",
            "source": "registry",
            "field": "local_file",
            "value": "C:/docs/shared.pdf",
            "doc_ids": ["DOC-1", "DOC-2"],
        },
    ]


def test_build_duplicate_cleanup_plan_prefers_doc_id_present_in_more_sources() -> None:
    plan = build_duplicate_cleanup_plan(
        qdrant_identities=[
            {
            "doc_id": "DOC-KEEP",
            "source_file": "shared.pdf",
            "local_file": "C:/docs/shared.pdf",
            "source_sha256": "",
        },
        {
            "doc_id": "DOC-DROP",
            "source_file": "shared.pdf",
            "local_file": "",
            "source_sha256": "",
        },
        ],
        manifest_identities=[
            {
            "doc_id": "DOC-KEEP",
            "source_file": "shared.pdf",
            "local_file": "C:/docs/shared.pdf",
            "source_sha256": "",
        }
        ],
        registry_identities=[
            {
                "doc_id": "DOC-KEEP",
                "source_file": "shared.pdf",
                "local_file": "C:/docs/shared.pdf",
                "source_sha256": "",
            }
        ],
    )

    assert plan == [
        {
            "action": "drop_duplicate_doc_ids",
            "reason": "canonical doc_id supported by the broadest source coverage",
            "field": "source_file",
            "value": "shared.pdf",
            "keep_doc_id": "DOC-KEEP",
            "drop_doc_ids": ["DOC-DROP"],
            "sources_by_doc_id": {
                "DOC-DROP": ["qdrant"],
                "DOC-KEEP": ["manifest", "qdrant", "registry"],
            },
        },
    ]


def test_build_duplicate_cleanup_plan_marks_tied_duplicate_for_manual_review() -> None:
    plan = build_duplicate_cleanup_plan(
        qdrant_identities=[
            {
            "doc_id": "DOC-A",
            "source_file": "shared.pdf",
            "local_file": "",
            "source_sha256": "",
        }
        ],
        manifest_identities=[
            {
                "doc_id": "DOC-B",
                "source_file": "shared.pdf",
                "local_file": "",
                "source_sha256": "",
            }
        ],
        registry_identities=[],
    )

    assert plan == [
        {
            "action": "manual_review",
            "reason": "no canonical doc_id established across sources",
            "field": "source_file",
            "value": "shared.pdf",
            "doc_ids": ["DOC-A", "DOC-B"],
            "sources_by_doc_id": {
                "DOC-A": ["qdrant"],
                "DOC-B": ["manifest"],
            },
        }
    ]


def test_reconcile_collection_state_reports_duplicate_source_sha256_conflicts() -> None:
    audit = reconcile_collection_state(
        qdrant_docs={
            "DOC-1": {"chunks": 4, "text_chunks": 3, "table_chunks": 1},
            "DOC-2": {"chunks": 5, "text_chunks": 5, "table_chunks": 0},
        },
        manifest_docs={
            "DOC-1": {"chunks": 4, "text_chunks": 3, "table_chunks": 1},
            "DOC-2": {"chunks": 5, "text_chunks": 5, "table_chunks": 0},
        },
        registry_docs={
            "DOC-1": {"chunks": 4, "text_chunks": 3, "table_chunks": 1},
            "DOC-2": {"chunks": 5, "text_chunks": 5, "table_chunks": 0},
        },
        qdrant_identities=[],
        manifest_identities=[
            {
                "doc_id": "DOC-1",
                "source_file": "doc1.pdf",
                "local_file": "C:/docs/one.pdf",
                "source_sha256": "hash-1",
            },
            {
                "doc_id": "DOC-2",
                "source_file": "doc2.pdf",
                "local_file": "C:/docs/two.pdf",
                "source_sha256": "hash-1",
            },
        ],
        registry_identities=[],
    )

    assert audit["issue_count"] == 1
    assert audit["issues"] == [
        {
            "issue_type": "duplicate_identity",
            "source": "manifest",
            "field": "source_sha256",
            "value": "hash-1",
            "doc_ids": ["DOC-1", "DOC-2"],
        }
    ]


def test_find_doc_metadata_mismatch_issues_reports_per_doc_field_diffs() -> None:
    issues = find_doc_metadata_mismatch_issues(
        qdrant_identities=[
            {
                "doc_id": "DOC-1",
                "source_file": "doc1.pdf",
                "local_file": "C:/docs/doc1.pdf",
                "source_sha256": "hash-1",
            }
        ],
        manifest_identities=[
            {
                "doc_id": "DOC-1",
                "source_file": "doc1-renamed.pdf",
                "local_file": "C:/docs/doc1.pdf",
                "source_sha256": "hash-1",
            }
        ],
        registry_identities=[
            {
                "doc_id": "DOC-1",
                "source_file": "doc1.pdf",
                "local_file": "C:/docs/doc1-copy.pdf",
                "source_sha256": "hash-2",
            }
        ],
    )

    assert issues == [
        {
            "issue_type": "metadata_mismatch",
            "doc_id": "DOC-1",
            "fields": ["source_file", "local_file", "source_sha256"],
            "sources": {
                "source_file": {
                    "qdrant": "doc1.pdf",
                    "manifest": "doc1-renamed.pdf",
                    "registry": "doc1.pdf",
                },
                "local_file": {
                    "qdrant": "C:/docs/doc1.pdf",
                    "manifest": "C:/docs/doc1.pdf",
                    "registry": "C:/docs/doc1-copy.pdf",
                },
                "source_sha256": {
                    "qdrant": "hash-1",
                    "manifest": "hash-1",
                    "registry": "hash-2",
                },
            },
        }
    ]


def test_find_chunk_count_sanity_issues_reports_internal_breakdown_problems() -> None:
    issues = find_chunk_count_sanity_issues(
        {
            "DOC-1": {"chunks": 0, "text_chunks": 0, "table_chunks": 0},
            "DOC-2": {"chunks": 4, "text_chunks": 1, "table_chunks": 1},
            "DOC-3": {"chunks": 3, "text_chunks": 0, "table_chunks": 3},
        },
        source_name="manifest",
    )

    assert issues == [
        {
            "issue_type": "chunk_count_sanity",
            "source": "manifest",
            "doc_id": "DOC-1",
            "checks": ["no_chunks"],
            "summary": {"chunks": 0, "text_chunks": 0, "table_chunks": 0},
            "median_chunk_count": 3.5,
            "low_outlier_threshold": 1,
            "high_outlier_threshold": 17,
        },
        {
            "issue_type": "chunk_count_sanity",
            "source": "manifest",
            "doc_id": "DOC-2",
            "checks": ["count_breakdown_mismatch"],
            "summary": {"chunks": 4, "text_chunks": 1, "table_chunks": 1},
            "median_chunk_count": 3.5,
            "low_outlier_threshold": 1,
            "high_outlier_threshold": 17,
        },
        {
            "issue_type": "chunk_count_sanity",
            "source": "manifest",
            "doc_id": "DOC-3",
            "checks": ["no_text_chunks"],
            "summary": {"chunks": 3, "text_chunks": 0, "table_chunks": 3},
            "median_chunk_count": 3.5,
            "low_outlier_threshold": 1,
            "high_outlier_threshold": 17,
        },
    ]


def test_find_chunk_count_sanity_issues_reports_extreme_outliers_when_collection_is_large_enough() -> None:
    issues = find_chunk_count_sanity_issues(
        {
            "DOC-A": {"chunks": 100, "text_chunks": 90, "table_chunks": 10},
            "DOC-B": {"chunks": 110, "text_chunks": 100, "table_chunks": 10},
            "DOC-C": {"chunks": 95, "text_chunks": 90, "table_chunks": 5},
            "DOC-D": {"chunks": 105, "text_chunks": 95, "table_chunks": 10},
            "DOC-E": {"chunks": 98, "text_chunks": 93, "table_chunks": 5},
            "DOC-LOW": {"chunks": 20, "text_chunks": 19, "table_chunks": 1},
            "DOC-HIGH": {"chunks": 700, "text_chunks": 690, "table_chunks": 10},
        },
        source_name="qdrant",
    )

    assert [issue["doc_id"] for issue in issues] == ["DOC-HIGH", "DOC-LOW"]
    assert issues[0]["checks"] == ["unusually_high_chunk_count"]
    assert issues[1]["checks"] == ["unusually_low_chunk_count"]
    assert issues[0]["median_chunk_count"] == 100
    assert issues[0]["low_outlier_threshold"] == 20
    assert issues[0]["high_outlier_threshold"] == 500


def test_reconcile_collection_state_includes_chunk_count_sanity_issues() -> None:
    audit = reconcile_collection_state(
        qdrant_docs={
            "DOC-1": {"chunks": 0, "text_chunks": 0, "table_chunks": 0},
        },
        manifest_docs={
            "DOC-1": {"chunks": 0, "text_chunks": 0, "table_chunks": 0},
        },
        registry_docs={
            "DOC-1": {"chunks": 0, "text_chunks": 0, "table_chunks": 0},
        },
        qdrant_identities=[],
        manifest_identities=[],
        registry_identities=[],
    )

    assert audit["issue_count"] == 3
    assert audit["issue_types"] == {"chunk_count_sanity": 3}


def test_reconcile_collection_state_includes_metadata_mismatch_and_repair_plan() -> None:
    audit = reconcile_collection_state(
        qdrant_docs={
            "DOC-1": {"chunks": 4, "text_chunks": 3, "table_chunks": 1},
        },
        manifest_docs={
            "DOC-1": {"chunks": 4, "text_chunks": 3, "table_chunks": 1},
        },
        registry_docs={
            "DOC-1": {"chunks": 4, "text_chunks": 3, "table_chunks": 1},
        },
        qdrant_identities=[
            {
                "doc_id": "DOC-1",
                "source_file": "doc1.pdf",
                "local_file": "C:/docs/doc1.pdf",
                "source_sha256": "hash-1",
            }
        ],
        manifest_identities=[
            {
                "doc_id": "DOC-1",
                "source_file": "doc1-renamed.pdf",
                "local_file": "C:/docs/doc1.pdf",
                "source_sha256": "hash-1",
            }
        ],
        registry_identities=[
            {
                "doc_id": "DOC-1",
                "source_file": "doc1.pdf",
                "local_file": "C:/docs/doc1-copy.pdf",
                "source_sha256": "hash-2",
            }
        ],
    )

    assert audit["issue_count"] == 1
    assert audit["issue_types"] == {"metadata_mismatch": 1}
    assert audit["repair_plan_count"] == 1
    assert audit["repair_plan"] == [
        {
            "action": "review_doc_metadata",
            "doc_id": "DOC-1",
            "fields": ["source_file", "local_file", "source_sha256"],
            "reason": "doc metadata differs across sources and needs canonical reconciliation",
        }
    ]


def test_build_reconciliation_repair_plan_suggests_registry_sync_for_manifest_only_doc() -> None:
    plan = build_reconciliation_repair_plan(
        [
            {
                "issue_type": "missing_doc",
                "doc_id": "DOC-9",
                "present_in": ["qdrant", "manifest"],
            }
        ]
    )

    assert plan == [
        {
            "action": "sync_registry_from_manifest",
            "doc_id": "DOC-9",
            "reason": "manifest and registry are out of sync for an otherwise indexed document",
        }
    ]
