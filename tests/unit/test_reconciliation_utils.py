from __future__ import annotations

from src.app.ingestion.reconciliation_utils import (
    build_manifest_doc_summary,
    build_qdrant_doc_summary,
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
    )

    assert audit["issue_count"] == 2
    assert audit["issue_types"] == {"count_mismatch": 1, "missing_doc": 1}
