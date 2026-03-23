from __future__ import annotations

from src.app.ingestion.reconciliation_utils import (
    build_duplicate_cleanup_plan,
    build_manifest_doc_identities,
    build_manifest_doc_summary,
    build_qdrant_doc_identities,
    build_qdrant_doc_summary,
    build_registry_doc_identities,
    build_registry_doc_summary,
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
            },
            {
                "doc_id": "DOC-2",
                "source_file": "shared.pdf",
                "local_file": "C:/docs/two.pdf",
            },
        ],
        registry_identities=[
            {
                "doc_id": "DOC-1",
                "source_file": "doc1.pdf",
                "local_file": "C:/docs/shared.pdf",
            },
            {
                "doc_id": "DOC-2",
                "source_file": "doc2.pdf",
                "local_file": "C:/docs/shared.pdf",
            },
        ],
    )

    assert audit["issue_count"] == 2
    assert audit["issue_types"] == {"duplicate_identity": 2}
    assert audit["issues"] == [
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
            },
            {
                "doc_id": "DOC-DROP",
                "source_file": "shared.pdf",
                "local_file": "",
            },
        ],
        manifest_identities=[
            {
                "doc_id": "DOC-KEEP",
                "source_file": "shared.pdf",
                "local_file": "C:/docs/shared.pdf",
            }
        ],
        registry_identities=[
            {
                "doc_id": "DOC-KEEP",
                "source_file": "shared.pdf",
                "local_file": "C:/docs/shared.pdf",
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
            }
        ],
        manifest_identities=[
            {
                "doc_id": "DOC-B",
                "source_file": "shared.pdf",
                "local_file": "",
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
