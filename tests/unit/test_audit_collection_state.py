from __future__ import annotations

from scripts.audit_collection_state import should_fail_audit


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
