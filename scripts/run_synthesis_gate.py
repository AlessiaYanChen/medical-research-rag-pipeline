from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


MAX_ABSTAIN_ACCURACY_DROP = 0.10
MIN_ABSTAIN_ACCURACY = 0.80
MIN_HAS_INSIGHT_RATE = 1.0
MIN_CONFIDENCE_MEETS_MINIMUM_RATE = 0.75
MAX_FALSE_CONFIDENCE_ON_KNOWN_GAPS = 0

DEFAULT_OUTPUT_PATH = Path("data/eval/results/synthesis_gate_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare synthesis evaluation outputs against the Stage-1 baseline."
    )
    parser.add_argument(
        "--baseline-file",
        required=True,
        help="Baseline answer-quality evaluation JSON path.",
    )
    parser.add_argument(
        "--current-file",
        required=True,
        help="Current answer-quality evaluation JSON path.",
    )
    parser.add_argument(
        "--known-gaps-file",
        required=True,
        help="Known-gap evaluation JSON path.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSON report path.",
    )
    return parser.parse_args()


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def build_synthesis_gate_report(
    *,
    baseline_payload: dict[str, Any],
    current_payload: dict[str, Any],
    known_gaps_payload: dict[str, Any],
    baseline_file: Path,
    current_file: Path,
    known_gaps_file: Path,
) -> dict[str, Any]:
    baseline_summary = _summary_object(baseline_payload, label="baseline")
    current_summary = _summary_object(current_payload, label="current")
    known_gap_queries = _query_list(known_gaps_payload, label="known gaps")

    abstain_accuracy_baseline = _required_float(baseline_summary, "abstain_accuracy", label="baseline")
    abstain_accuracy_current = _required_float(current_summary, "abstain_accuracy", label="current")
    abstain_delta = round(abstain_accuracy_current - abstain_accuracy_baseline, 4)
    abstain_passed = (
        abstain_accuracy_current >= MIN_ABSTAIN_ACCURACY
        and abstain_delta >= -MAX_ABSTAIN_ACCURACY_DROP
    )

    has_insight_rate = _required_float(current_summary, "has_insight_rate", label="current")
    has_insight_passed = has_insight_rate >= MIN_HAS_INSIGHT_RATE

    confidence_baseline = _required_float(baseline_summary, "confidence_meets_minimum_rate", label="baseline")
    confidence_current = _required_float(current_summary, "confidence_meets_minimum_rate", label="current")
    confidence_delta = round(confidence_current - confidence_baseline, 4)
    confidence_passed = confidence_current >= MIN_CONFIDENCE_MEETS_MINIMUM_RATE

    false_confidence_query_ids = [
        str(item.get("query_id", "")).strip()
        for item in known_gap_queries
        if _query_confidence(item) == "HIGH"
    ]
    false_confidence_count = len(false_confidence_query_ids)
    false_confidence_passed = false_confidence_count <= MAX_FALSE_CONFIDENCE_ON_KNOWN_GAPS

    checks = {
        "abstain_accuracy": {
            "baseline": abstain_accuracy_baseline,
            "current": abstain_accuracy_current,
            "delta": abstain_delta,
            "passed": abstain_passed,
        },
        "has_insight_rate": {
            "current": has_insight_rate,
            "passed": has_insight_passed,
        },
        "confidence_meets_minimum_rate": {
            "baseline": confidence_baseline,
            "current": confidence_current,
            "delta": confidence_delta,
            "passed": confidence_passed,
        },
        "false_confidence_on_known_gaps": {
            "count": false_confidence_count,
            "query_ids": false_confidence_query_ids,
            "passed": false_confidence_passed,
        },
    }

    failures = [name for name, check in checks.items() if not bool(check["passed"])]
    return {
        "baseline_file": str(baseline_file),
        "current_file": str(current_file),
        "known_gaps_file": str(known_gaps_file),
        "gate_passed": not failures,
        "failures": failures,
        "checks": checks,
    }


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    baseline_file = Path(args.baseline_file)
    current_file = Path(args.current_file)
    known_gaps_file = Path(args.known_gaps_file)
    output_path = Path(args.output)

    try:
        baseline_payload = load_json_object(baseline_file, label="baseline file")
        current_payload = load_json_object(current_file, label="current file")
        known_gaps_payload = load_json_object(known_gaps_file, label="known gaps file")
        report = build_synthesis_gate_report(
            baseline_payload=baseline_payload,
            current_payload=current_payload,
            known_gaps_payload=known_gaps_payload,
            baseline_file=baseline_file,
            current_file=current_file,
            known_gaps_file=known_gaps_file,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    write_json(report, output_path)

    print(f"Baseline file: {baseline_file}")
    print(f"Current file: {current_file}")
    print(f"Known gaps file: {known_gaps_file}")
    print(f"Gate passed: {report['gate_passed']}")
    print(f"Output: {output_path}")
    return 0 if report["gate_passed"] else 1


def _summary_object(payload: dict[str, Any], *, label: str) -> dict[str, Any]:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{label} summary must be a JSON object")
    return summary


def _query_list(payload: dict[str, Any], *, label: str) -> list[dict[str, Any]]:
    queries = payload.get("queries")
    if not isinstance(queries, list):
        raise ValueError(f"{label} queries must be a JSON array")
    return [item for item in queries if isinstance(item, dict)]


def _required_float(summary: dict[str, Any], key: str, *, label: str) -> float:
    value = summary.get(key)
    if value is None:
        raise ValueError(f"{label} summary is missing '{key}'")
    return float(value)


def _query_confidence(query_payload: dict[str, Any]) -> str:
    evaluation = query_payload.get("evaluation")
    if isinstance(evaluation, dict):
        confidence = evaluation.get("confidence")
        if confidence is not None:
            return str(confidence).strip().upper()
    answer = query_payload.get("answer")
    if isinstance(answer, dict):
        confidence = answer.get("confidence")
        if confidence is not None:
            return str(confidence).strip().upper()
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
