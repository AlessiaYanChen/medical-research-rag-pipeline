from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.app.evaluation.answer_quality_eval import (
    AnswerQualityQuery,
    build_answer_quality_summary,
    evaluate_answer_quality,
    load_answer_quality_queries,
)
from src.app.services.reasoning_service import ConfidenceLevel, ResearchAnswer
from src.app.services.retrieval_service import RetrievedChunk


def _make_answer(
    insight: str = "The study found significant improvement.",
    evidence_basis: str = "DOC-1, Results section.",
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    citations: list[RetrievedChunk] | None = None,
) -> ResearchAnswer:
    if citations is None:
        citations = [
            RetrievedChunk(
                source="Results", doc_id="DOC-1", content="text", chunk_type="text", content_role="child"
            ),
            RetrievedChunk(
                source="Methods", doc_id="DOC-1", content="text", chunk_type="text", content_role="child"
            ),
        ]
    return ResearchAnswer(
        insight=insight,
        evidence_basis=evidence_basis,
        citations=citations,
        confidence=confidence,
    )


def _make_query(**kwargs) -> AnswerQualityQuery:
    defaults = {"id": "AQ01", "query": "What is the effect?"}
    return AnswerQualityQuery(**{**defaults, **kwargs})


# --- evaluate_answer_quality ---

def test_evaluate_basic_fields_present() -> None:
    result = evaluate_answer_quality(_make_query(), _make_answer())
    assert result["query_id"] == "AQ01"
    assert result["has_insight"] is True
    assert result["has_evidence_basis"] is True
    assert result["confidence"] == "MEDIUM"
    assert result["citation_count"] == 2
    assert result["distinct_cited_docs"] == 1


def test_evaluate_abstain_correct_when_expected() -> None:
    query = _make_query(expected_abstain=True)
    answer = _make_answer(insight="Insufficient evidence.", citations=[])
    result = evaluate_answer_quality(query, answer)
    assert result["abstained"] is True
    assert result["abstain_correct"] is True


def test_evaluate_abstain_incorrect_when_not_expected() -> None:
    query = _make_query(expected_abstain=False)
    answer = _make_answer(insight="Insufficient evidence.", citations=[])
    result = evaluate_answer_quality(query, answer)
    assert result["abstained"] is True
    assert result["abstain_correct"] is False


def test_evaluate_abstain_detection_case_insensitive() -> None:
    answer = _make_answer(insight="INSUFFICIENT EVIDENCE found for this query.")
    result = evaluate_answer_quality(_make_query(), answer)
    assert result["abstained"] is True


def test_evaluate_no_abstain_when_substantive() -> None:
    answer = _make_answer(insight="LDL-C improved by 12%.")
    result = evaluate_answer_quality(_make_query(expected_abstain=False), answer)
    assert result["abstained"] is False
    assert result["abstain_correct"] is True


def test_evaluate_confidence_meets_minimum_true() -> None:
    query = _make_query(expected_confidence_min=ConfidenceLevel.LOW)
    answer = _make_answer(confidence=ConfidenceLevel.MEDIUM)
    result = evaluate_answer_quality(query, answer)
    assert result["confidence_meets_minimum"] is True


def test_evaluate_confidence_meets_minimum_false() -> None:
    query = _make_query(expected_confidence_min=ConfidenceLevel.HIGH)
    answer = _make_answer(confidence=ConfidenceLevel.MEDIUM)
    result = evaluate_answer_quality(query, answer)
    assert result["confidence_meets_minimum"] is False


def test_evaluate_confidence_meets_minimum_exact_match() -> None:
    query = _make_query(expected_confidence_min=ConfidenceLevel.MEDIUM)
    answer = _make_answer(confidence=ConfidenceLevel.MEDIUM)
    result = evaluate_answer_quality(query, answer)
    assert result["confidence_meets_minimum"] is True


def test_evaluate_confidence_meets_minimum_none_when_not_specified() -> None:
    result = evaluate_answer_quality(_make_query(), _make_answer())
    assert result["confidence_meets_minimum"] is None
    assert result["expected_confidence_min"] is None


def test_evaluate_doc_id_coverage_full() -> None:
    query = _make_query(expected_doc_ids=("DOC-1",))
    answer = _make_answer(evidence_basis="Supported by DOC-1, Results section.")
    result = evaluate_answer_quality(query, answer)
    assert result["doc_id_coverage"] == 1.0
    assert result["expected_doc_ids_found"] == ["DOC-1"]


def test_evaluate_doc_id_coverage_partial() -> None:
    query = _make_query(expected_doc_ids=("DOC-1", "DOC-2"))
    answer = _make_answer(evidence_basis="Only DOC-1 is cited here.")
    result = evaluate_answer_quality(query, answer)
    assert result["doc_id_coverage"] == 0.5
    assert result["expected_doc_ids_found"] == ["DOC-1"]


def test_evaluate_doc_id_coverage_none_when_not_specified() -> None:
    result = evaluate_answer_quality(_make_query(), _make_answer())
    assert result["doc_id_coverage"] is None
    assert result["expected_doc_ids"] == []


def test_evaluate_doc_id_coverage_case_insensitive() -> None:
    query = _make_query(expected_doc_ids=("DOC-1",))
    answer = _make_answer(evidence_basis="doc-1 results are strong.")
    result = evaluate_answer_quality(query, answer)
    assert result["doc_id_coverage"] == 1.0


# --- build_answer_quality_summary ---

def test_summary_abstain_accuracy() -> None:
    evals = [
        evaluate_answer_quality(_make_query(expected_abstain=False), _make_answer(insight="Substantive answer.")),
        evaluate_answer_quality(_make_query(expected_abstain=True), _make_answer(insight="Insufficient evidence.", citations=[])),
    ]
    summary = build_answer_quality_summary(evals)
    assert summary["abstain_accuracy"] == 1.0
    assert summary["queries_total"] == 2


def test_summary_has_insight_rate() -> None:
    evals = [
        evaluate_answer_quality(_make_query(), _make_answer(insight="")),
        evaluate_answer_quality(_make_query(), _make_answer(insight="Something.")),
    ]
    summary = build_answer_quality_summary(evals)
    assert summary["has_insight_rate"] == 0.5


def test_summary_confidence_meets_minimum_rate_none_when_no_queries_specify_minimum() -> None:
    evals = [evaluate_answer_quality(_make_query(), _make_answer())]
    summary = build_answer_quality_summary(evals)
    assert summary["confidence_meets_minimum_rate"] is None


def test_summary_confidence_meets_minimum_rate_computed() -> None:
    evals = [
        evaluate_answer_quality(
            _make_query(expected_confidence_min=ConfidenceLevel.LOW),
            _make_answer(confidence=ConfidenceLevel.HIGH),
        ),
        evaluate_answer_quality(
            _make_query(expected_confidence_min=ConfidenceLevel.HIGH),
            _make_answer(confidence=ConfidenceLevel.LOW),
        ),
    ]
    summary = build_answer_quality_summary(evals)
    assert summary["confidence_meets_minimum_rate"] == 0.5


def test_summary_average_citation_count() -> None:
    citations_2 = [
        RetrievedChunk(source="Results", doc_id="DOC-1", content="x", chunk_type="text", content_role="child"),
        RetrievedChunk(source="Methods", doc_id="DOC-1", content="x", chunk_type="text", content_role="child"),
    ]
    evals = [
        evaluate_answer_quality(_make_query(), _make_answer(citations=citations_2)),
        evaluate_answer_quality(_make_query(), _make_answer(citations=[])),
    ]
    summary = build_answer_quality_summary(evals)
    assert summary["average_citation_count"] == 1.0


# --- load_answer_quality_queries ---

def test_load_queries_valid() -> None:
    data = [
        {
            "id": "AQ01",
            "query": "What is the detection rate?",
            "expected_abstain": False,
            "expected_confidence_min": "MEDIUM",
            "expected_doc_ids": ["DOC-1"],
            "notes": "stable baseline query",
        }
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        tmp_path = Path(f.name)

    queries = load_answer_quality_queries(tmp_path)
    assert len(queries) == 1
    q = queries[0]
    assert q.id == "AQ01"
    assert q.expected_abstain is False
    assert q.expected_confidence_min == ConfidenceLevel.MEDIUM
    assert q.expected_doc_ids == ("DOC-1",)
    assert q.notes == "stable baseline query"


def test_load_queries_minimal_fields() -> None:
    data = [{"query": "Simple question?"}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        tmp_path = Path(f.name)

    queries = load_answer_quality_queries(tmp_path)
    assert len(queries) == 1
    assert queries[0].id == "AQ01"
    assert queries[0].expected_abstain is False
    assert queries[0].expected_confidence_min is None
    assert queries[0].expected_doc_ids == ()


def test_load_queries_rejects_missing_query_text() -> None:
    data = [{"id": "AQ01"}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        tmp_path = Path(f.name)

    with pytest.raises(ValueError, match="missing a non-empty 'query'"):
        load_answer_quality_queries(tmp_path)


def test_load_queries_rejects_invalid_confidence_level() -> None:
    data = [{"query": "Test?", "expected_confidence_min": "VERY_HIGH"}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        tmp_path = Path(f.name)

    with pytest.raises(ValueError, match="unknown confidence level"):
        load_answer_quality_queries(tmp_path)
