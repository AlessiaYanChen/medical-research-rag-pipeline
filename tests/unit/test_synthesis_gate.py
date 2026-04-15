from __future__ import annotations

import json
from pathlib import Path
import tempfile

from scripts.run_synthesis_gate import (
    MAX_ABSTAIN_ACCURACY_DROP,
    MIN_ABSTAIN_ACCURACY,
    MIN_CONFIDENCE_MEETS_MINIMUM_RATE,
    build_synthesis_gate_report,
    main,
)


def _payload(
    *,
    abstain_accuracy: float = 0.85,
    has_insight_rate: float = 1.0,
    confidence_meets_minimum_rate: float = 0.80,
    queries: list[dict] | None = None,
) -> dict:
    return {
        "summary": {
            "abstain_accuracy": abstain_accuracy,
            "has_insight_rate": has_insight_rate,
            "confidence_meets_minimum_rate": confidence_meets_minimum_rate,
        },
        "queries": queries or [],
    }


def _known_gap_query(query_id: str, confidence: str) -> dict:
    return {
        "query_id": query_id,
        "evaluation": {
            "confidence": confidence,
        },
    }


def test_gate_passes_when_all_metrics_meet_thresholds() -> None:
    report = build_synthesis_gate_report(
        baseline_payload=_payload(abstain_accuracy=0.85, confidence_meets_minimum_rate=0.80),
        current_payload=_payload(abstain_accuracy=0.84, confidence_meets_minimum_rate=0.78),
        known_gaps_payload=_payload(queries=[_known_gap_query("K01", "INSUFFICIENT")]),
        baseline_file=Path("baseline.json"),
        current_file=Path("current.json"),
        known_gaps_file=Path("known_gaps.json"),
    )

    assert report["gate_passed"] is True
    assert report["failures"] == []
    assert report["checks"]["abstain_accuracy"]["passed"] is True
    assert report["checks"]["has_insight_rate"]["passed"] is True
    assert report["checks"]["confidence_meets_minimum_rate"]["passed"] is True
    assert report["checks"]["false_confidence_on_known_gaps"]["passed"] is True


def test_gate_fails_when_abstain_accuracy_drop_exceeds_threshold() -> None:
    report = build_synthesis_gate_report(
        baseline_payload=_payload(abstain_accuracy=0.95),
        current_payload=_payload(abstain_accuracy=0.95 - MAX_ABSTAIN_ACCURACY_DROP - 0.01),
        known_gaps_payload=_payload(),
        baseline_file=Path("baseline.json"),
        current_file=Path("current.json"),
        known_gaps_file=Path("known_gaps.json"),
    )

    assert report["gate_passed"] is False
    assert "abstain_accuracy" in report["failures"]


def test_gate_fails_when_abstain_accuracy_below_absolute_floor() -> None:
    report = build_synthesis_gate_report(
        baseline_payload=_payload(abstain_accuracy=MIN_ABSTAIN_ACCURACY + 0.01),
        current_payload=_payload(abstain_accuracy=MIN_ABSTAIN_ACCURACY - 0.01),
        known_gaps_payload=_payload(),
        baseline_file=Path("baseline.json"),
        current_file=Path("current.json"),
        known_gaps_file=Path("known_gaps.json"),
    )

    assert report["gate_passed"] is False
    assert "abstain_accuracy" in report["failures"]


def test_gate_fails_when_has_insight_rate_below_threshold() -> None:
    report = build_synthesis_gate_report(
        baseline_payload=_payload(),
        current_payload=_payload(has_insight_rate=0.99),
        known_gaps_payload=_payload(),
        baseline_file=Path("baseline.json"),
        current_file=Path("current.json"),
        known_gaps_file=Path("known_gaps.json"),
    )

    assert report["gate_passed"] is False
    assert "has_insight_rate" in report["failures"]


def test_gate_fails_when_confidence_rate_below_threshold() -> None:
    report = build_synthesis_gate_report(
        baseline_payload=_payload(confidence_meets_minimum_rate=0.82),
        current_payload=_payload(confidence_meets_minimum_rate=MIN_CONFIDENCE_MEETS_MINIMUM_RATE - 0.01),
        known_gaps_payload=_payload(),
        baseline_file=Path("baseline.json"),
        current_file=Path("current.json"),
        known_gaps_file=Path("known_gaps.json"),
    )

    assert report["gate_passed"] is False
    assert "confidence_meets_minimum_rate" in report["failures"]


def test_gate_fails_when_known_gap_has_high_confidence() -> None:
    report = build_synthesis_gate_report(
        baseline_payload=_payload(),
        current_payload=_payload(),
        known_gaps_payload=_payload(queries=[_known_gap_query("K06", "HIGH")]),
        baseline_file=Path("baseline.json"),
        current_file=Path("current.json"),
        known_gaps_file=Path("known_gaps.json"),
    )

    assert report["gate_passed"] is False
    assert "false_confidence_on_known_gaps" in report["failures"]
    assert report["checks"]["false_confidence_on_known_gaps"]["query_ids"] == ["K06"]


def test_gate_failures_list_reflects_multiple_failures() -> None:
    report = build_synthesis_gate_report(
        baseline_payload=_payload(abstain_accuracy=0.95, confidence_meets_minimum_rate=0.82),
        current_payload=_payload(
            abstain_accuracy=0.70,
            has_insight_rate=0.90,
            confidence_meets_minimum_rate=0.70,
        ),
        known_gaps_payload=_payload(queries=[_known_gap_query("K06", "HIGH")]),
        baseline_file=Path("baseline.json"),
        current_file=Path("current.json"),
        known_gaps_file=Path("known_gaps.json"),
    )

    assert report["gate_passed"] is False
    assert report["failures"] == [
        "abstain_accuracy",
        "has_insight_rate",
        "confidence_meets_minimum_rate",
        "false_confidence_on_known_gaps",
    ]


def test_main_returns_zero_when_gate_passes(monkeypatch, capsys) -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
        tmp_path = Path(tmp_dir)
        baseline_path = tmp_path / "baseline.json"
        current_path = tmp_path / "current.json"
        known_gaps_path = tmp_path / "known_gaps.json"
        output_path = tmp_path / "report.json"
        baseline_path.write_text(json.dumps(_payload(abstain_accuracy=0.85, confidence_meets_minimum_rate=0.80)), encoding="utf-8")
        current_path.write_text(json.dumps(_payload(abstain_accuracy=0.84, confidence_meets_minimum_rate=0.78)), encoding="utf-8")
        known_gaps_path.write_text(json.dumps(_payload(queries=[_known_gap_query("K01", "LOW")])), encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_synthesis_gate.py",
                "--baseline-file",
                str(baseline_path),
                "--current-file",
                str(current_path),
                "--known-gaps-file",
                str(known_gaps_path),
                "--output",
                str(output_path),
            ],
        )

        result = main()

        assert result == 0
        assert output_path.exists()
        assert "Gate passed: True" in capsys.readouterr().out


def test_main_returns_one_when_gate_fails(monkeypatch, capsys) -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
        tmp_path = Path(tmp_dir)
        baseline_path = tmp_path / "baseline.json"
        current_path = tmp_path / "current.json"
        known_gaps_path = tmp_path / "known_gaps.json"
        output_path = tmp_path / "report.json"
        baseline_path.write_text(json.dumps(_payload()), encoding="utf-8")
        current_path.write_text(json.dumps(_payload(has_insight_rate=0.5)), encoding="utf-8")
        known_gaps_path.write_text(json.dumps(_payload()), encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_synthesis_gate.py",
                "--baseline-file",
                str(baseline_path),
                "--current-file",
                str(current_path),
                "--known-gaps-file",
                str(known_gaps_path),
                "--output",
                str(output_path),
            ],
        )

        result = main()

        assert result == 1
        assert output_path.exists()
        assert "Gate passed: False" in capsys.readouterr().out


def test_main_returns_one_with_clear_error_when_input_missing(monkeypatch, capsys) -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
        tmp_path = Path(tmp_dir)
        current_path = tmp_path / "current.json"
        known_gaps_path = tmp_path / "known_gaps.json"
        current_path.write_text(json.dumps(_payload()), encoding="utf-8")
        known_gaps_path.write_text(json.dumps(_payload()), encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_synthesis_gate.py",
                "--baseline-file",
                str(tmp_path / "missing_baseline.json"),
                "--current-file",
                str(current_path),
                "--known-gaps-file",
                str(known_gaps_path),
            ],
        )

        result = main()

        assert result == 1
        assert "ERROR: baseline file not found" in capsys.readouterr().out
