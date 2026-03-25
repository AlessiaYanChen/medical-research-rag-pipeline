from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_project_root_on_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two parser bakeoff result JSON files and list query-level regressions."
    )
    parser.add_argument("--baseline", required=True, help="Baseline result JSON path, e.g. Marker output.")
    parser.add_argument("--candidate", required=True, help="Candidate result JSON path, e.g. Docling output.")
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write the comparison payload as JSON.",
    )
    return parser.parse_args()


def load_result_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result payload is not a JSON object: {path}")
    return payload


def compare_query_evaluations(
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
) -> dict[str, Any]:
    baseline_queries = _queries_by_id(baseline_payload)
    candidate_queries = _queries_by_id(candidate_payload)
    all_query_ids = sorted(set(baseline_queries) | set(candidate_queries))

    regressions: list[dict[str, Any]] = []
    for query_id in all_query_ids:
        baseline_query = baseline_queries.get(query_id)
        candidate_query = candidate_queries.get(query_id)
        if baseline_query is None or candidate_query is None:
            regressions.append(
                {
                    "query_id": query_id,
                    "regression_fields": ["missing_query"],
                    "baseline_present": baseline_query is not None,
                    "candidate_present": candidate_query is not None,
                }
            )
            continue

        baseline_eval = baseline_query.get("evaluation", {})
        candidate_eval = candidate_query.get("evaluation", {})
        regression_fields = _regression_fields(baseline_eval, candidate_eval)
        if not regression_fields:
            continue
        regressions.append(
            {
                "query_id": query_id,
                "query": baseline_query.get("query", candidate_query.get("query", "")),
                "regression_fields": regression_fields,
                "baseline": {
                    "result_docs": baseline_eval.get("result_docs", []),
                    "result_headers": baseline_eval.get("result_headers", []),
                    "expected_doc_hit": baseline_eval.get("expected_doc_hit", False),
                    "expected_header_hit": baseline_eval.get("expected_header_hit", False),
                    "top1_expected_doc_hit": baseline_eval.get("top1_expected_doc_hit", False),
                    "top1_expected_header_hit": baseline_eval.get("top1_expected_header_hit", False),
                    "doc_precision": baseline_eval.get("doc_precision", 0.0),
                    "header_precision": baseline_eval.get("header_precision", 0.0),
                    "table_hits": baseline_eval.get("table_hits", 0),
                    "citation_noise_hits": baseline_eval.get("citation_noise_hits", 0),
                },
                "candidate": {
                    "result_docs": candidate_eval.get("result_docs", []),
                    "result_headers": candidate_eval.get("result_headers", []),
                    "expected_doc_hit": candidate_eval.get("expected_doc_hit", False),
                    "expected_header_hit": candidate_eval.get("expected_header_hit", False),
                    "top1_expected_doc_hit": candidate_eval.get("top1_expected_doc_hit", False),
                    "top1_expected_header_hit": candidate_eval.get("top1_expected_header_hit", False),
                    "doc_precision": candidate_eval.get("doc_precision", 0.0),
                    "header_precision": candidate_eval.get("header_precision", 0.0),
                    "table_hits": candidate_eval.get("table_hits", 0),
                    "citation_noise_hits": candidate_eval.get("citation_noise_hits", 0),
                },
            }
        )

    return {
        "dataset": baseline_payload.get("dataset", candidate_payload.get("dataset", "")),
        "baseline_collection": baseline_payload.get("collection", ""),
        "candidate_collection": candidate_payload.get("collection", ""),
        "regression_count": len(regressions),
        "regressed_query_ids": [item["query_id"] for item in regressions],
        "regressions": regressions,
    }


def _queries_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    queries = payload.get("queries", [])
    if not isinstance(queries, list):
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for item in queries:
        if not isinstance(item, dict):
            continue
        query_id = str(item.get("query_id", "")).strip()
        if query_id:
            indexed[query_id] = item
    return indexed


def _regression_fields(
    baseline_eval: dict[str, Any],
    candidate_eval: dict[str, Any],
) -> list[str]:
    regressions: list[str] = []
    boolean_fields = (
        "expected_doc_hit",
        "expected_header_hit",
        "top1_expected_doc_hit",
        "top1_expected_header_hit",
    )
    for field in boolean_fields:
        if bool(baseline_eval.get(field, False)) and not bool(candidate_eval.get(field, False)):
            regressions.append(field)

    numeric_lower_is_worse = (
        "doc_precision",
        "header_precision",
        "table_hits",
    )
    for field in numeric_lower_is_worse:
        if float(candidate_eval.get(field, 0)) < float(baseline_eval.get(field, 0)):
            regressions.append(field)

    numeric_higher_is_worse = ("citation_noise_hits",)
    for field in numeric_higher_is_worse:
        if int(candidate_eval.get(field, 0)) > int(baseline_eval.get(field, 0)):
            regressions.append(field)

    return regressions


def main() -> int:
    args = parse_args()
    baseline_payload = load_result_payload(args.baseline)
    candidate_payload = load_result_payload(args.candidate)
    comparison = compare_query_evaluations(baseline_payload, candidate_payload)

    if args.json_out.strip():
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print(f"Dataset: {comparison['dataset']}")
    print(f"Baseline: {comparison['baseline_collection']}")
    print(f"Candidate: {comparison['candidate_collection']}")
    print(f"Regression count: {comparison['regression_count']}")
    print(f"Regressed query IDs: {comparison['regressed_query_ids']}")
    for item in comparison["regressions"]:
        print(f"- {item['query_id']}: fields={item['regression_fields']}")
    if args.json_out.strip():
        print(f"JSON output: {Path(args.json_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
