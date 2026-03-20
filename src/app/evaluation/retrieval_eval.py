from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from src.app.services.retrieval_service import RetrievedChunk


@dataclass(frozen=True)
class EvaluationQuery:
    id: str
    query: str
    doc_id: str | None = None
    expected_docs: tuple[str, ...] = ()
    expected_headers: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    include_tables: bool | None = None
    notes: str = ""


def load_evaluation_queries(path: str | Path) -> list[EvaluationQuery]:
    dataset_path = Path(path)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Evaluation dataset must be a JSON array of query objects.")

    queries: list[EvaluationQuery] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Evaluation item {index} must be an object.")
        query = str(item.get("query", "")).strip()
        if not query:
            raise ValueError(f"Evaluation item {index} is missing a non-empty 'query'.")

        query_id = str(item.get("id", f"Q{index:02d}")).strip() or f"Q{index:02d}"
        doc_id = str(item.get("doc_id", "")).strip() or None
        expected_docs = tuple(_normalize_expected_list(item.get("expected_docs")))
        expected_headers = tuple(_normalize_expected_list(item.get("expected_headers")))
        labels = tuple(_normalize_expected_list(item.get("labels")))
        include_tables = _normalize_optional_bool(item.get("include_tables"))
        notes = str(item.get("notes", "")).strip()
        queries.append(
            EvaluationQuery(
                id=query_id,
                query=query,
                doc_id=doc_id,
                expected_docs=expected_docs,
                expected_headers=expected_headers,
                labels=labels,
                include_tables=include_tables,
                notes=notes,
            )
        )
    return queries


def evaluate_retrieval_results(query: EvaluationQuery, chunks: list[RetrievedChunk]) -> dict[str, Any]:
    result_docs = [chunk.doc_id for chunk in chunks]
    result_headers = [chunk.source for chunk in chunks]
    result_contents = [chunk.content for chunk in chunks]
    expected_doc_set = {doc.lower() for doc in query.expected_docs}
    expected_header_set = {_normalize_header(expected) for expected in query.expected_headers}

    expected_doc_hit = True if not query.expected_docs else any(doc.lower() in expected_doc_set for doc in result_docs)
    expected_header_hit = True if not query.expected_headers else any(
        _normalize_header(header) in expected_header_set
        for header in result_headers
    )
    top1_expected_doc_hit = True if not query.expected_docs else bool(result_docs and result_docs[0].lower() in expected_doc_set)
    top1_expected_header_hit = True if not query.expected_headers else bool(
        result_headers and _normalize_header(result_headers[0]) in expected_header_set
    )
    doc_precision = _safe_ratio(
        sum(1 for doc in result_docs if doc.lower() in expected_doc_set),
        len(result_docs),
    ) if query.expected_docs else None
    header_precision = _safe_ratio(
        sum(1 for header in result_headers if _normalize_header(header) in expected_header_set),
        len(result_headers),
    ) if query.expected_headers else None

    table_hits = sum(1 for chunk in chunks if chunk.chunk_type == "table" or chunk.content_role == "table")
    citation_noise_hits = sum(1 for content in result_contents if _looks_citation_heavy(content))
    duplicate_hits = _count_near_duplicate_contents(result_contents)
    non_structural_headers = [header for header in result_headers if _is_non_structural_header(header)]

    return {
        "query_id": query.id,
        "query": query.query,
        "doc_filter": query.doc_id or "",
        "expected_docs": list(query.expected_docs),
        "expected_headers": list(query.expected_headers),
        "labels": list(query.labels),
        "include_tables": query.include_tables,
        "notes": query.notes,
        "result_count": len(chunks),
        "result_docs": result_docs,
        "result_headers": result_headers,
        "expected_doc_hit": expected_doc_hit,
        "expected_header_hit": expected_header_hit,
        "top1_expected_doc_hit": top1_expected_doc_hit,
        "top1_expected_header_hit": top1_expected_header_hit,
        "doc_precision": doc_precision,
        "header_precision": header_precision,
        "table_hits": table_hits,
        "citation_noise_hits": citation_noise_hits,
        "duplicate_hits": duplicate_hits,
        "non_structural_header_hits": len(non_structural_headers),
        "non_structural_headers": non_structural_headers,
        "distinct_doc_count": len({doc.lower() for doc in result_docs if doc.strip()}),
        "distinct_header_count": len({_normalize_header(header) for header in result_headers if header.strip()}),
    }


def build_summary(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    total_queries = len(evaluations)
    return {
        "queries_total": total_queries,
        "expected_doc_hit_rate": _safe_ratio(sum(1 for item in evaluations if item["expected_doc_hit"]), total_queries),
        "expected_header_hit_rate": _safe_ratio(sum(1 for item in evaluations if item["expected_header_hit"]), total_queries),
        "top1_expected_doc_hit_rate": _safe_ratio(sum(1 for item in evaluations if item["top1_expected_doc_hit"]), total_queries),
        "top1_expected_header_hit_rate": _safe_ratio(sum(1 for item in evaluations if item["top1_expected_header_hit"]), total_queries),
        "average_doc_precision": _average_optional(item["doc_precision"] for item in evaluations),
        "average_header_precision": _average_optional(item["header_precision"] for item in evaluations),
        "queries_with_citation_noise": sum(1 for item in evaluations if item["citation_noise_hits"] > 0),
        "queries_with_table_hits": sum(1 for item in evaluations if item["table_hits"] > 0),
        "queries_with_duplicate_hits": sum(1 for item in evaluations if item["duplicate_hits"] > 0),
        "queries_with_non_structural_headers": sum(1 for item in evaluations if item["non_structural_header_hits"] > 0),
        "cross_document_queries": sum(1 for item in evaluations if not item["doc_filter"]),
        "cross_document_average_doc_precision": _average_optional(
            item["doc_precision"] for item in evaluations if not item["doc_filter"]
        ),
        "total_table_hits": sum(int(item["table_hits"]) for item in evaluations),
        "total_citation_noise_hits": sum(int(item["citation_noise_hits"]) for item in evaluations),
        "total_duplicate_hits": sum(int(item["duplicate_hits"]) for item in evaluations),
        "total_non_structural_header_hits": sum(int(item["non_structural_header_hits"]) for item in evaluations),
    }


def _normalize_expected_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Expected list fields must be arrays of strings.")
    return [str(item).strip() for item in value if str(item).strip()]


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _average_optional(values: Any) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return round(sum(filtered) / len(filtered), 4)


def _normalize_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError("Optional boolean fields must be true, false, or omitted.")


def _looks_citation_heavy(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    score = 0
    patterns = (
        r"\bet al\.",
        r"\bdoi[:/]",
        r"\bpmid\b",
        r"\bissn\b",
        r"\bhttps?://",
        r"\[[0-9,\s]+\]",
    )
    for pattern in patterns:
        matches = re.findall(pattern, normalized)
        if pattern == r"\[[0-9,\s]+\]":
            score += len(matches)
        elif matches:
            score += 1
    return score >= 2


def _count_near_duplicate_contents(contents: list[str]) -> int:
    duplicate_count = 0
    selected_tokens: list[set[str]] = []
    for content in contents:
        tokens = _tokenize(content)
        if not tokens:
            continue
        if any(_token_overlap(tokens, existing) > 0.9 for existing in selected_tokens):
            duplicate_count += 1
            continue
        selected_tokens.append(tokens)
    return duplicate_count


def _token_overlap(first: set[str], second: set[str]) -> float:
    return len(first & second) / max(1, min(len(first), len(second)))


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _normalize_header(header: str) -> str:
    return re.sub(r"\s+", " ", header.strip().lower())


def _is_non_structural_header(header: str) -> bool:
    normalized = _normalize_header(header)
    if not normalized:
        return True

    structural_tokens = (
        "document metadata/abstract",
        "abstract",
        "introduction",
        "background",
        "methods",
        "materials",
        "results",
        "discussion",
        "conclusion",
        "summary",
        "references",
        "bibliography",
        "acknowledg",
        "funding",
        "conflict",
        "author contribution",
        "data availability",
    )
    return not any(token in normalized for token in structural_tokens)
