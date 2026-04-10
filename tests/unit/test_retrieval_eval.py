from __future__ import annotations

import json
from pathlib import Path

from src.app.evaluation.retrieval_eval import (
    EvaluationQuery,
    build_summary,
    evaluate_retrieval_results,
    load_evaluation_queries,
)
from src.app.services.retrieval_service import RetrievedChunk


def test_load_evaluation_queries_reads_expected_fields(tmp_path: Path) -> None:
    dataset_path = tmp_path / "queries.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "Q01",
                    "query": "What does the paper conclude?",
                    "doc_id": "DOC-1",
                    "expected_docs": ["DOC-1"],
                    "expected_headers": ["Discussion", "Conclusion"],
                    "labels": ["single_doc", "discussion_preference"],
                    "include_tables": True,
                    "notes": "starter",
                }
            ]
        ),
        encoding="utf-8",
    )

    queries = load_evaluation_queries(dataset_path)

    assert len(queries) == 1
    assert queries[0].id == "Q01"
    assert queries[0].doc_id == "DOC-1"
    assert queries[0].expected_docs == ("DOC-1",)
    assert queries[0].expected_headers == ("Discussion", "Conclusion")
    assert queries[0].labels == ("single_doc", "discussion_preference")
    assert queries[0].include_tables is True


def test_evaluate_retrieval_results_counts_noise_and_duplicates(tmp_path: Path) -> None:
    query = load_evaluation_queries(
        _write_dataset(
            tmp_path,
            [
                {
                    "id": "Q01",
                    "query": "clinical utility",
                    "expected_docs": ["DOC-1"],
                    "expected_headers": ["Discussion"],
                }
            ]
        )
    )[0]

    chunks = [
        RetrievedChunk(
            source="Discussion",
            doc_id="DOC-1",
            content="Clinical utility is discussed with enough detail to remain useful.",
            chunk_type="text",
            content_role="child",
        ),
        RetrievedChunk(
            source="PCR/ESI-MS allows detection of non-cultivable and fastidious pathogens",
            doc_id="DOC-1",
            content="Smith et al. doi:10.1000/example [1, 2, 3]",
            chunk_type="text",
            content_role="reference",
        ),
        RetrievedChunk(
            source="Discussion",
            doc_id="DOC-1",
            content="Clinical utility is discussed with enough detail to remain useful in practice.",
            chunk_type="text",
            content_role="child",
        ),
    ]

    evaluation = evaluate_retrieval_results(query, chunks)

    assert evaluation["expected_doc_hit"] is True
    assert evaluation["expected_header_hit"] is True
    assert evaluation["top1_expected_doc_hit"] is True
    assert evaluation["top1_expected_header_hit"] is True
    assert evaluation["labels"] == []
    assert evaluation["doc_precision"] == 1.0
    assert evaluation["header_precision"] == 0.6667
    assert evaluation["citation_noise_hits"] == 1
    assert evaluation["duplicate_hits"] == 1
    assert evaluation["non_structural_header_hits"] == 1
    assert evaluation["non_structural_headers"] == ["PCR/ESI-MS allows detection of non-cultivable and fastidious pathogens"]


def test_evaluate_retrieval_results_matches_numbered_and_compound_headers() -> None:
    query = EvaluationQuery(
        id="Q02",
        query="What did the study report?",
        expected_docs=("DOC-1",),
        expected_headers=("Results", "Discussion", "Conclusion"),
    )

    chunks = [
        RetrievedChunk(
            source="5 | RESULTS AND DISCUSSION",
            doc_id="DOC-1",
            content="The primary outcome improved after intervention.",
            chunk_type="text",
            content_role="child",
        ),
        RetrievedChunk(
            source="6. Conclusions",
            doc_id="DOC-1",
            content="The intervention appears useful.",
            chunk_type="text",
            content_role="child",
        ),
        RetrievedChunk(
            source="2. Materials and Methods",
            doc_id="DOC-1",
            content="This section describes the study design.",
            chunk_type="text",
            content_role="child",
        ),
    ]

    evaluation = evaluate_retrieval_results(query, chunks)

    assert evaluation["expected_header_hit"] is True
    assert evaluation["top1_expected_header_hit"] is True
    assert evaluation["header_precision"] == 0.6667


def test_evaluate_retrieval_results_treats_summary_as_abstract() -> None:
    query = EvaluationQuery(
        id="Q03",
        query="What is the paper about?",
        expected_docs=("DOC-1",),
        expected_headers=("Abstract",),
    )

    chunks = [
        RetrievedChunk(
            source="SUMMARY",
            doc_id="DOC-1",
            content="This review summarizes current laboratory stewardship practice.",
            chunk_type="text",
            content_role="child",
        ),
    ]

    evaluation = evaluate_retrieval_results(query, chunks)

    assert evaluation["expected_header_hit"] is True
    assert evaluation["top1_expected_header_hit"] is True
    assert evaluation["header_precision"] == 1.0


def test_build_summary_aggregates_query_metrics() -> None:
    summary = build_summary(
        [
            {
                "expected_doc_hit": True,
                "expected_header_hit": True,
                "top1_expected_doc_hit": True,
                "top1_expected_header_hit": True,
                "doc_precision": 1.0,
                "header_precision": 1.0,
                "citation_noise_hits": 0,
                "table_hits": 0,
                "duplicate_hits": 0,
                "non_structural_header_hits": 0,
                "doc_filter": "DOC-1",
            },
            {
                "expected_doc_hit": False,
                "expected_header_hit": True,
                "top1_expected_doc_hit": False,
                "top1_expected_header_hit": False,
                "doc_precision": 0.25,
                "header_precision": 0.5,
                "citation_noise_hits": 2,
                "table_hits": 1,
                "duplicate_hits": 1,
                "non_structural_header_hits": 2,
                "doc_filter": "",
            },
        ]
    )

    assert summary["queries_total"] == 2
    assert summary["expected_doc_hit_rate"] == 0.5
    assert summary["expected_header_hit_rate"] == 1.0
    assert summary["top1_expected_doc_hit_rate"] == 0.5
    assert summary["top1_expected_header_hit_rate"] == 0.5
    assert summary["average_doc_precision"] == 0.625
    assert summary["average_header_precision"] == 0.75
    assert summary["queries_with_citation_noise"] == 1
    assert summary["queries_with_table_hits"] == 1
    assert summary["queries_with_duplicate_hits"] == 1
    assert summary["queries_with_non_structural_headers"] == 1
    assert summary["total_non_structural_header_hits"] == 2
    assert summary["cross_document_queries"] == 1
    assert summary["cross_document_average_doc_precision"] == 0.25


def test_repo_evaluation_datasets_load_cleanly() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_paths = (
        repo_root / "data" / "eval" / "sample_queries.json",
        repo_root / "data" / "eval" / "expanded_queries.json",
        repo_root / "data" / "eval" / "ood_adversarial_queries.json",
    )

    for dataset_path in dataset_paths:
        queries = load_evaluation_queries(dataset_path)
        assert queries, f"{dataset_path} should contain at least one query"


def _write_dataset(tmp_path: Path, payload: list[dict[str, object]]) -> Path:
    dataset_path = tmp_path / "queries.json"
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")
    return dataset_path
