from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_collection_state import load_manifest_payload, should_fail_audit


def test_should_fail_audit_returns_false_for_clean_state() -> None:
    assert should_fail_audit(
        manifest_version_issues=[],
        issue_count=0,
        cleanup_plan_count=0,
    ) is False


def test_should_fail_audit_returns_true_for_manifest_version_issues() -> None:
    assert should_fail_audit(
        manifest_version_issues=["Manifest collection mismatch"],
        issue_count=0,
        cleanup_plan_count=0,
    ) is True


def test_should_fail_audit_returns_true_for_reconciliation_issues() -> None:
    assert should_fail_audit(
        manifest_version_issues=[],
        issue_count=1,
        cleanup_plan_count=0,
    ) is True


def test_should_fail_audit_returns_true_for_cleanup_plan_entries() -> None:
    assert should_fail_audit(
        manifest_version_issues=[],
        issue_count=0,
        cleanup_plan_count=1,
    ) is True


def test_load_manifest_payload_reads_object_payload(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"collection": "medical_research_chunks_v1"}), encoding="utf-8")

    payload = load_manifest_payload(manifest_path)

    assert payload == {"collection": "medical_research_chunks_v1"}


def test_load_manifest_payload_rejects_invalid_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match=r"manifest is not valid JSON: .*line 1, column 2"):
        load_manifest_payload(manifest_path)


def test_load_manifest_payload_rejects_non_object_payload(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest is not a JSON object"):
        load_manifest_payload(manifest_path)
