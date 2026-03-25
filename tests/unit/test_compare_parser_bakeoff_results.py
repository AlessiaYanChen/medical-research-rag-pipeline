from __future__ import annotations

from experiments.compare_parser_bakeoff_results import compare_query_evaluations


def test_compare_query_evaluations_reports_doc_and_noise_regressions() -> None:
    baseline_payload = {
        "dataset": "data/eval/ood_adversarial_queries.json",
        "collection": "marker",
        "queries": [
            {
                "query_id": "O04",
                "query": "Which indexed paper would I cite...",
                "evaluation": {
                    "expected_doc_hit": True,
                    "expected_header_hit": True,
                    "top1_expected_doc_hit": True,
                    "top1_expected_header_hit": True,
                    "doc_precision": 1.0,
                    "header_precision": 1.0,
                    "table_hits": 0,
                    "citation_noise_hits": 0,
                    "result_docs": ["RAPID"],
                    "result_headers": ["Discussion"],
                },
            }
        ],
    }
    candidate_payload = {
        "dataset": "data/eval/ood_adversarial_queries.json",
        "collection": "docling",
        "queries": [
            {
                "query_id": "O04",
                "query": "Which indexed paper would I cite...",
                "evaluation": {
                    "expected_doc_hit": False,
                    "expected_header_hit": True,
                    "top1_expected_doc_hit": False,
                    "top1_expected_header_hit": True,
                    "doc_precision": 0.0,
                    "header_precision": 1.0,
                    "table_hits": 0,
                    "citation_noise_hits": 1,
                    "result_docs": ["Single site RCT"],
                    "result_headers": ["DISCUSSION"],
                },
            }
        ],
    }

    comparison = compare_query_evaluations(baseline_payload, candidate_payload)

    assert comparison["regression_count"] == 1
    assert comparison["regressed_query_ids"] == ["O04"]
    assert comparison["regressions"][0]["regression_fields"] == [
        "expected_doc_hit",
        "top1_expected_doc_hit",
        "doc_precision",
        "citation_noise_hits",
    ]


def test_compare_query_evaluations_reports_table_hit_regression() -> None:
    baseline_payload = {
        "dataset": "data/eval/sample_queries.json",
        "collection": "marker",
        "queries": [
            {
                "query_id": "Q05",
                "query": "What sensitivity or specificity findings are reported...",
                "evaluation": {
                    "expected_doc_hit": True,
                    "expected_header_hit": True,
                    "top1_expected_doc_hit": True,
                    "top1_expected_header_hit": True,
                    "doc_precision": 1.0,
                    "header_precision": 1.0,
                    "table_hits": 1,
                    "citation_noise_hits": 0,
                    "result_docs": ["Culture-Free Lipidomics-Based Screening Test"],
                    "result_headers": ["Results"],
                },
            }
        ],
    }
    candidate_payload = {
        "dataset": "data/eval/sample_queries.json",
        "collection": "docling",
        "queries": [
            {
                "query_id": "Q05",
                "query": "What sensitivity or specificity findings are reported...",
                "evaluation": {
                    "expected_doc_hit": True,
                    "expected_header_hit": True,
                    "top1_expected_doc_hit": True,
                    "top1_expected_header_hit": True,
                    "doc_precision": 1.0,
                    "header_precision": 1.0,
                    "table_hits": 0,
                    "citation_noise_hits": 0,
                    "result_docs": ["Culture-Free Lipidomics-Based Screening Test"],
                    "result_headers": ["Results"],
                },
            }
        ],
    }

    comparison = compare_query_evaluations(baseline_payload, candidate_payload)

    assert comparison["regressed_query_ids"] == ["Q05"]
    assert comparison["regressions"][0]["regression_fields"] == ["table_hits"]
