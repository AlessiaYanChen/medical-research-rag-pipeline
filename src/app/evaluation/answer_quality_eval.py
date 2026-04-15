from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import re
import unicodedata

from src.app.services.reasoning_service import ConfidenceLevel, ResearchAnswer


_CONFIDENCE_ORDER: dict[ConfidenceLevel, int] = {
    ConfidenceLevel.INSUFFICIENT: 0,
    ConfidenceLevel.LOW: 1,
    ConfidenceLevel.MEDIUM: 2,
    ConfidenceLevel.HIGH: 3,
}


@dataclass(frozen=True)
class AnswerQualityQuery:
    id: str
    query: str
    expected_abstain: bool = False
    expected_confidence_min: ConfidenceLevel | None = None
    expected_doc_ids: tuple[str, ...] = ()
    notes: str = ""


def load_answer_quality_queries(path: str | Path) -> list[AnswerQualityQuery]:
    dataset_path = Path(path)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Answer quality dataset must be a JSON array.")

    queries: list[AnswerQualityQuery] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} must be an object.")
        query_text = str(item.get("query", "")).strip()
        if not query_text:
            raise ValueError(f"Item {index} is missing a non-empty 'query'.")

        query_id = str(item.get("id", f"AQ{index:02d}")).strip() or f"AQ{index:02d}"
        expected_abstain = bool(item.get("expected_abstain", False))

        raw_min = item.get("expected_confidence_min")
        expected_confidence_min: ConfidenceLevel | None = None
        if raw_min is not None:
            try:
                expected_confidence_min = ConfidenceLevel(str(raw_min).upper())
            except ValueError:
                raise ValueError(
                    f"Item {index}: unknown confidence level '{raw_min}'. "
                    f"Valid values: {[lvl.value for lvl in ConfidenceLevel]}"
                )

        raw_docs = item.get("expected_doc_ids")
        expected_doc_ids: tuple[str, ...] = ()
        if raw_docs is not None:
            if not isinstance(raw_docs, list):
                raise ValueError(f"Item {index}: 'expected_doc_ids' must be an array.")
            expected_doc_ids = tuple(str(d).strip() for d in raw_docs if str(d).strip())

        notes = str(item.get("notes", "")).strip()
        queries.append(AnswerQualityQuery(
            id=query_id,
            query=query_text,
            expected_abstain=expected_abstain,
            expected_confidence_min=expected_confidence_min,
            expected_doc_ids=expected_doc_ids,
            notes=notes,
        ))
    return queries


def evaluate_answer_quality(query: AnswerQualityQuery, answer: ResearchAnswer) -> dict[str, Any]:
    abstained = "insufficient evidence" in answer.insight.lower()
    abstain_correct = abstained == query.expected_abstain

    confidence_meets_minimum: bool | None = None
    if query.expected_confidence_min is not None:
        confidence_meets_minimum = (
            _CONFIDENCE_ORDER[answer.confidence] >= _CONFIDENCE_ORDER[query.expected_confidence_min]
        )

    evidence_basis_normalized = _normalize_match_text(answer.evidence_basis)
    expected_doc_ids_found = [
        doc_id for doc_id in query.expected_doc_ids
        if _normalize_match_text(doc_id) in evidence_basis_normalized
    ]
    doc_id_coverage: float | None = None
    if query.expected_doc_ids:
        doc_id_coverage = round(len(expected_doc_ids_found) / len(query.expected_doc_ids), 4)

    return {
        "query_id": query.id,
        "query": query.query,
        "notes": query.notes,
        "abstained": abstained,
        "expected_abstain": query.expected_abstain,
        "abstain_correct": abstain_correct,
        "has_insight": bool(answer.insight.strip()),
        "has_evidence_basis": bool(answer.evidence_basis.strip()),
        "confidence": answer.confidence.value,
        "expected_confidence_min": query.expected_confidence_min.value if query.expected_confidence_min else None,
        "confidence_meets_minimum": confidence_meets_minimum,
        "expected_doc_ids": list(query.expected_doc_ids),
        "expected_doc_ids_found": expected_doc_ids_found,
        "doc_id_coverage": doc_id_coverage,
        "citation_count": len(answer.citations),
        "distinct_cited_docs": len({c.doc_id for c in answer.citations}),
    }


def build_answer_quality_summary(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(evaluations)
    confidence_checks = [e for e in evaluations if e["confidence_meets_minimum"] is not None]
    doc_id_checks = [e for e in evaluations if e["doc_id_coverage"] is not None]

    return {
        "queries_total": total,
        "abstain_accuracy": _safe_ratio(
            sum(1 for e in evaluations if e["abstain_correct"]), total
        ),
        "has_insight_rate": _safe_ratio(
            sum(1 for e in evaluations if e["has_insight"]), total
        ),
        "has_evidence_basis_rate": _safe_ratio(
            sum(1 for e in evaluations if e["has_evidence_basis"]), total
        ),
        "confidence_meets_minimum_rate": _safe_ratio(
            sum(1 for e in confidence_checks if e["confidence_meets_minimum"]),
            len(confidence_checks),
        ) if confidence_checks else None,
        "average_doc_id_coverage": _average(
            e["doc_id_coverage"] for e in doc_id_checks
        ) if doc_id_checks else None,
        "average_citation_count": _average(e["citation_count"] for e in evaluations),
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _average(values: Any) -> float | None:
    items = [float(v) for v in values if v is not None]
    if not items:
        return None
    return round(sum(items) / len(items), 4)


def _normalize_match_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    normalized = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]+", "-", normalized)
    normalized = re.sub(r"[*_`]+", "", normalized)
    return " ".join(normalized.split())
