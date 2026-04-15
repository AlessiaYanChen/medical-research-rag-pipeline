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

from src.app.ingestion.registry_utils import default_manifest_path_for_collection

DEFAULT_RESULTS_DIR = Path("data/eval/results")
DEFAULT_EVAL_PATHS = {
    "stable": DEFAULT_RESULTS_DIR / "retrieval_eval_sample.json",
    "expanded": DEFAULT_RESULTS_DIR / "retrieval_eval_expanded.json",
    "ood": DEFAULT_RESULTS_DIR / "ood_retrieval_eval.json",
    "runtime": DEFAULT_RESULTS_DIR / "retrieval_eval_runtime_queries.json",
}
SUMMARY_METRICS = (
    "expected_doc_hit_rate",
    "expected_header_hit_rate",
    "top1_expected_doc_hit_rate",
    "top1_expected_header_hit_rate",
    "average_doc_precision",
    "average_header_precision",
    "cross_document_average_doc_precision",
)
PASS_STATUS = "pass"
FAIL_STATUS = "fail"
REVIEW_REQUIRED_STATUS = "review_required"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a medium-scale rollout stage report from rebuild, audit, evaluation, and manual spot-check artifacts."
    )
    parser.add_argument(
        "--collection",
        default="medical_research_chunks_docling_v1",
        help="Collection name being evaluated for rollout.",
    )
    parser.add_argument(
        "--stage-label",
        required=True,
        help="Short rollout stage label, for example stage-1-20-pdfs.",
    )
    parser.add_argument(
        "--target-pdf-count",
        type=int,
        default=0,
        help="Optional expected document count for this rollout stage.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional manifest path override. Defaults to data/ingestion_manifests/<collection>_rebuild_manifest.json.",
    )
    parser.add_argument(
        "--rebuild-failures",
        default="",
        help="Optional rebuild failure report path. Defaults to data/eval/results/rebuild_failures_<collection>.json.",
    )
    parser.add_argument(
        "--audit-json",
        default="",
        help="Optional audit JSON path. Defaults to data/eval/results/collection_audit_<collection>.json.",
    )
    parser.add_argument(
        "--stable-eval-json",
        default=str(DEFAULT_EVAL_PATHS["stable"]),
        help="Stable benchmark evaluation JSON path.",
    )
    parser.add_argument(
        "--expanded-eval-json",
        default=str(DEFAULT_EVAL_PATHS["expanded"]),
        help="Expanded benchmark evaluation JSON path.",
    )
    parser.add_argument(
        "--ood-eval-json",
        default=str(DEFAULT_EVAL_PATHS["ood"]),
        help="OOD/adversarial benchmark evaluation JSON path.",
    )
    parser.add_argument(
        "--runtime-eval-json",
        default=str(DEFAULT_EVAL_PATHS["runtime"]),
        help="Runtime benchmark evaluation JSON path.",
    )
    parser.add_argument(
        "--baseline-stable-eval-json",
        default="",
        help="Optional baseline stable evaluation JSON path for comparison.",
    )
    parser.add_argument(
        "--baseline-expanded-eval-json",
        default="",
        help="Optional baseline expanded evaluation JSON path for comparison.",
    )
    parser.add_argument(
        "--baseline-ood-eval-json",
        default="",
        help="Optional baseline OOD evaluation JSON path for comparison.",
    )
    parser.add_argument(
        "--baseline-runtime-eval-json",
        default="",
        help="Optional baseline runtime evaluation JSON path for comparison.",
    )
    parser.add_argument(
        "--max-metric-drop",
        type=float,
        default=0.02,
        help="Maximum allowed drop per retrieval summary metric when a baseline evaluation is provided.",
    )
    parser.add_argument(
        "--manual-spot-checks",
        default="",
        help="Optional manual spot-check JSON path. When omitted or missing, the rollout report stays incomplete.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional JSON output path. Defaults to data/eval/results/rollout_report_<collection>.json.",
    )
    parser.add_argument(
        "--md-out",
        default="",
        help="Optional Markdown output path. Defaults to data/eval/results/rollout_report_<collection>.md.",
    )
    parser.add_argument(
        "--promotion-alias",
        default="",
        help="Optional stable alias name to advertise as the rollout promotion target when the report passes.",
    )
    parser.add_argument(
        "--promotion-source-manifest",
        default="",
        help="Optional manifest path override to include in the generated promotion command.",
    )
    parser.add_argument(
        "--promotion-registry",
        default="data/kb_registry.json",
        help="Registry path to include in the generated promotion command.",
    )
    parser.add_argument(
        "--promotion-json-out",
        default="",
        help="Optional JSON path to include in the generated promotion command output.",
    )
    return parser.parse_args()


def resolve_manifest_path(collection: str, override: str) -> Path:
    if override.strip():
        return Path(override)
    return default_manifest_path_for_collection(collection)


def resolve_rebuild_failure_path(collection: str, override: str) -> Path:
    if override.strip():
        return Path(override)
    return DEFAULT_RESULTS_DIR / f"rebuild_failures_{collection}.json"


def resolve_audit_path(collection: str, override: str) -> Path:
    if override.strip():
        return Path(override)
    return DEFAULT_RESULTS_DIR / f"collection_audit_{collection}.json"


def resolve_json_output_path(collection: str, override: str) -> Path:
    if override.strip():
        return Path(override)
    return DEFAULT_RESULTS_DIR / f"rollout_report_{collection}.json"


def resolve_markdown_output_path(collection: str, override: str) -> Path:
    if override.strip():
        return Path(override)
    return DEFAULT_RESULTS_DIR / f"rollout_report_{collection}.md"


def load_json_object(path: Path, *, label: str, required: bool) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise ValueError(f"{label} not found: {path}")
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return loaded


def load_manual_spot_checks(path: Path) -> list[dict[str, Any]]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        loaded = loaded.get("checks", [])
    if not isinstance(loaded, list):
        raise ValueError(f"manual spot checks must be a JSON array or an object with a 'checks' array: {path}")

    checks: list[dict[str, Any]] = []
    for index, item in enumerate(loaded, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"manual spot check {index} must be a JSON object: {path}")
        query = str(item.get("query", "")).strip()
        status = str(item.get("status", "")).strip().lower()
        if not query:
            raise ValueError(f"manual spot check {index} is missing 'query': {path}")
        if status not in {PASS_STATUS, FAIL_STATUS}:
            raise ValueError(f"manual spot check {index} must use status 'pass' or 'fail': {path}")
        checks.append(
            {
                "query": query,
                "status": status,
                "observed": str(item.get("observed", "")).strip(),
                "expected": str(item.get("expected", "")).strip(),
                "repeated": bool(item.get("repeated", False)),
            }
        )
    return checks


def build_rebuild_gate(
    *,
    collection: str,
    manifest_payload: dict[str, Any] | None,
    manifest_path: Path,
    failure_payload: dict[str, Any] | None,
    failure_path: Path,
    target_pdf_count: int,
) -> dict[str, Any]:
    issues: list[str] = []
    doc_count = 0
    chunk_count = 0
    parser_name = ""
    if manifest_payload is None:
        issues.append(f"Missing rebuild manifest: {manifest_path}")
    else:
        doc_count = int(manifest_payload.get("doc_count", 0) or 0)
        chunk_count = int(manifest_payload.get("chunk_count", 0) or 0)
        parser_name = str(manifest_payload.get("parser", "")).strip()
        manifest_collection = str(manifest_payload.get("collection", "")).strip()
        if manifest_collection and manifest_collection != collection:
            issues.append(f"Manifest collection mismatch: expected {collection}, got {manifest_collection}")
        if target_pdf_count > 0 and doc_count != target_pdf_count:
            issues.append(f"Target PDF count mismatch: expected {target_pdf_count}, got {doc_count}")

    failure_count = 0
    failures: list[dict[str, Any]] = []
    if failure_payload is not None:
        failure_count = int(failure_payload.get("failure_count", 0) or 0)
        raw_failures = failure_payload.get("failures", [])
        if isinstance(raw_failures, list):
            failures = [item for item in raw_failures if isinstance(item, dict)]
        if failure_count > 0:
            issues.append(f"Rebuild failure report contains {failure_count} failed PDF(s): {failure_path}")

    status = PASS_STATUS if not issues else FAIL_STATUS
    return {
        "status": status,
        "manifest_path": str(manifest_path),
        "failure_report_path": str(failure_path),
        "doc_count": doc_count,
        "chunk_count": chunk_count,
        "parser": parser_name,
        "failure_count": failure_count,
        "failures": failures,
        "issues": issues,
    }


def build_audit_gate(audit_payload: dict[str, Any] | None, audit_path: Path) -> dict[str, Any]:
    if audit_payload is None:
        return {
            "status": FAIL_STATUS,
            "audit_path": str(audit_path),
            "issue_count": 0,
            "cleanup_plan_count": 0,
            "manifest_version_issues": [],
            "issues": [f"Missing audit report: {audit_path}"],
        }

    manifest_version_issues = [
        str(item).strip()
        for item in audit_payload.get("manifest_version_issues", [])
        if str(item).strip()
    ]
    issue_count = int(audit_payload.get("issue_count", 0) or 0)
    cleanup_plan_count = int(audit_payload.get("cleanup_plan_count", 0) or 0)
    issues: list[str] = []
    if manifest_version_issues:
        issues.append(f"Manifest version issues: {len(manifest_version_issues)}")
    if issue_count > 0:
        issues.append(f"Audit reconciliation issues: {issue_count}")
    if cleanup_plan_count > 0:
        issues.append(f"Duplicate cleanup plan entries: {cleanup_plan_count}")

    return {
        "status": PASS_STATUS if not issues else FAIL_STATUS,
        "audit_path": str(audit_path),
        "issue_count": issue_count,
        "cleanup_plan_count": cleanup_plan_count,
        "manifest_version_issues": manifest_version_issues,
        "issues": issues,
    }


def build_eval_gate(
    *,
    label: str,
    candidate_payload: dict[str, Any] | None,
    candidate_path: Path,
    baseline_payload: dict[str, Any] | None,
    baseline_path: Path | None,
    max_metric_drop: float,
) -> dict[str, Any]:
    if candidate_payload is None:
        return {
            "status": FAIL_STATUS,
            "label": label,
            "candidate_path": str(candidate_path),
            "baseline_path": str(baseline_path) if baseline_path else "",
            "comparisons": [],
            "issues": [f"Missing {label} evaluation report: {candidate_path}"],
        }

    candidate_summary = candidate_payload.get("summary", {})
    if not isinstance(candidate_summary, dict):
        raise ValueError(f"{label} evaluation summary must be a JSON object: {candidate_path}")

    if baseline_payload is None:
        return {
            "status": REVIEW_REQUIRED_STATUS,
            "label": label,
            "candidate_path": str(candidate_path),
            "baseline_path": str(baseline_path) if baseline_path else "",
            "candidate_summary": candidate_summary,
            "comparisons": [],
            "issues": [f"No baseline provided for {label} evaluation; review candidate metrics manually."],
        }

    baseline_summary = baseline_payload.get("summary", {})
    if not isinstance(baseline_summary, dict):
        raise ValueError(f"{label} baseline summary must be a JSON object: {baseline_path}")

    comparisons = []
    issues: list[str] = []
    for metric_name in SUMMARY_METRICS:
        candidate_value = _optional_float(candidate_summary.get(metric_name))
        baseline_value = _optional_float(baseline_summary.get(metric_name))
        if candidate_value is None or baseline_value is None:
            issues.append(f"{label} metric missing from candidate or baseline summary: {metric_name}")
            continue
        delta = round(candidate_value - baseline_value, 4)
        passed = delta >= -max_metric_drop
        comparisons.append(
            {
                "metric": metric_name,
                "candidate": candidate_value,
                "baseline": baseline_value,
                "delta": delta,
                "max_drop": max_metric_drop,
                "passed": passed,
            }
        )
        if not passed:
            issues.append(
                f"{label} metric {metric_name} regressed by {abs(delta):.4f}, exceeding allowed drop {max_metric_drop:.4f}"
            )

    return {
        "status": PASS_STATUS if not issues else FAIL_STATUS,
        "label": label,
        "candidate_path": str(candidate_path),
        "baseline_path": str(baseline_path) if baseline_path else "",
        "candidate_summary": candidate_summary,
        "baseline_summary": baseline_summary,
        "comparisons": comparisons,
        "issues": issues,
    }


def build_manual_spot_check_gate(manual_path: Path | None) -> dict[str, Any]:
    if manual_path is None:
        return {
            "status": FAIL_STATUS,
            "path": "",
            "checks_total": 0,
            "pass_count": 0,
            "fail_count": 0,
            "checks": [],
            "issues": ["Manual spot-check report not provided."],
        }
    if not manual_path.exists():
        return {
            "status": FAIL_STATUS,
            "path": str(manual_path),
            "checks_total": 0,
            "pass_count": 0,
            "fail_count": 0,
            "checks": [],
            "issues": [f"Manual spot-check report not found: {manual_path}"],
        }

    checks = load_manual_spot_checks(manual_path)
    pass_count = sum(check["status"] == PASS_STATUS for check in checks)
    fail_count = sum(check["status"] == FAIL_STATUS for check in checks)
    issues: list[str] = []
    if not checks:
        issues.append("Manual spot-check report is empty.")
    if fail_count > 0:
        issues.append(f"Manual spot checks contain {fail_count} failed query review(s).")

    return {
        "status": PASS_STATUS if not issues else FAIL_STATUS,
        "path": str(manual_path),
        "checks_total": len(checks),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "checks": checks,
        "issues": issues,
    }


def determine_overall_status(gates: list[dict[str, Any]]) -> str:
    statuses = [str(gate.get("status", "")).strip().lower() for gate in gates]
    if any(status == FAIL_STATUS for status in statuses):
        return FAIL_STATUS
    if any(status == REVIEW_REQUIRED_STATUS for status in statuses):
        return REVIEW_REQUIRED_STATUS
    return PASS_STATUS


def build_promotion_command(
    *,
    overall_status: str,
    source_collection: str,
    promotion_alias: str,
    qdrant_url: str,
    promotion_source_manifest: str,
    promotion_registry: str,
    promotion_json_out: str,
) -> str:
    if overall_status != PASS_STATUS or not promotion_alias.strip():
        return ""

    parts = [
        r".\.venv\Scripts\python.exe",
        "scripts/promote_collection_alias.py",
        "--source-collection",
        source_collection,
        "--alias",
        promotion_alias.strip(),
        "--qdrant-url",
        qdrant_url,
        "--registry",
        promotion_registry,
    ]
    if promotion_source_manifest.strip():
        parts.extend(["--source-manifest", promotion_source_manifest.strip()])
    if promotion_json_out.strip():
        parts.extend(["--json-out", promotion_json_out.strip()])
    return " ".join(parts)


def build_rollout_report(
    *,
    collection: str,
    stage_label: str,
    target_pdf_count: int,
    max_metric_drop: float,
    manifest_payload: dict[str, Any] | None,
    manifest_path: Path,
    rebuild_failure_payload: dict[str, Any] | None,
    rebuild_failure_path: Path,
    audit_payload: dict[str, Any] | None,
    audit_path: Path,
    eval_inputs: list[dict[str, Any]],
    manual_spot_check_path: Path | None,
    promotion_alias: str = "",
    qdrant_url: str = "http://localhost:6333",
    promotion_source_manifest: str = "",
    promotion_registry: str = "data/kb_registry.json",
    promotion_json_out: str = "",
) -> dict[str, Any]:
    rebuild_gate = build_rebuild_gate(
        collection=collection,
        manifest_payload=manifest_payload,
        manifest_path=manifest_path,
        failure_payload=rebuild_failure_payload,
        failure_path=rebuild_failure_path,
        target_pdf_count=target_pdf_count,
    )
    audit_gate = build_audit_gate(audit_payload, audit_path)
    eval_gates = [
        build_eval_gate(
            label=str(item["label"]),
            candidate_payload=item.get("candidate_payload"),
            candidate_path=item["candidate_path"],
            baseline_payload=item.get("baseline_payload"),
            baseline_path=item.get("baseline_path"),
            max_metric_drop=max_metric_drop,
        )
        for item in eval_inputs
    ]
    manual_gate = build_manual_spot_check_gate(manual_spot_check_path)
    all_gates = [rebuild_gate, audit_gate, *eval_gates, manual_gate]
    overall_status = determine_overall_status(all_gates)
    promotion_command = build_promotion_command(
        overall_status=overall_status,
        source_collection=collection,
        promotion_alias=promotion_alias,
        qdrant_url=qdrant_url,
        promotion_source_manifest=promotion_source_manifest,
        promotion_registry=promotion_registry,
        promotion_json_out=promotion_json_out,
    )
    return {
        "collection": collection,
        "stage_label": stage_label,
        "target_pdf_count": target_pdf_count,
        "max_metric_drop": max_metric_drop,
        "overall_status": overall_status,
        "promotion_alias": promotion_alias.strip(),
        "promotion_command": promotion_command,
        "rebuild_gate": rebuild_gate,
        "audit_gate": audit_gate,
        "evaluation_gates": eval_gates,
        "manual_spot_check_gate": manual_gate,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# Rollout Report: {report['stage_label']}",
        "",
        f"- Collection: `{report['collection']}`",
        f"- Overall status: `{report['overall_status']}`",
    ]
    target_pdf_count = int(report.get("target_pdf_count", 0) or 0)
    if target_pdf_count > 0:
        lines.append(f"- Target PDF count: `{target_pdf_count}`")
    promotion_alias = str(report.get("promotion_alias", "")).strip()
    promotion_command = str(report.get("promotion_command", "")).strip()
    if promotion_alias:
        lines.append(f"- Promotion alias: `{promotion_alias}`")
    lines.extend(
        [
            "",
            "## Gates",
            "",
            _render_gate_line("Rebuild", report["rebuild_gate"]),
            _render_gate_line("Audit", report["audit_gate"]),
        ]
    )
    for eval_gate in report["evaluation_gates"]:
        lines.append(_render_gate_line(f"{eval_gate['label'].title()} eval", eval_gate))
    lines.append(_render_gate_line("Manual spot checks", report["manual_spot_check_gate"]))

    rebuild_gate = report["rebuild_gate"]
    lines.extend(
        [
            "",
            "## Rebuild",
            "",
            f"- Manifest: `{rebuild_gate['manifest_path']}`",
            f"- Parser: `{rebuild_gate.get('parser', '') or 'unknown'}`",
            f"- Documents ingested: `{rebuild_gate['doc_count']}`",
            f"- Chunks stored: `{rebuild_gate['chunk_count']}`",
            f"- Rebuild failures: `{rebuild_gate['failure_count']}`",
        ]
    )
    lines.extend(_render_issue_lines(rebuild_gate.get("issues", [])))

    audit_gate = report["audit_gate"]
    lines.extend(
        [
            "",
            "## Audit",
            "",
            f"- Audit report: `{audit_gate['audit_path']}`",
            f"- Reconciliation issues: `{audit_gate['issue_count']}`",
            f"- Cleanup plan entries: `{audit_gate['cleanup_plan_count']}`",
            f"- Manifest version issues: `{len(audit_gate['manifest_version_issues'])}`",
        ]
    )
    lines.extend(_render_issue_lines(audit_gate.get("issues", [])))

    lines.extend(
        [
            "",
            "## Evaluation",
            "",
        ]
    )
    for gate in report["evaluation_gates"]:
        lines.append(f"### {gate['label'].title()}")
        lines.append("")
        lines.append(f"- Status: `{gate['status']}`")
        lines.append(f"- Candidate report: `{gate['candidate_path']}`")
        if gate.get("baseline_path"):
            lines.append(f"- Baseline report: `{gate['baseline_path']}`")
        candidate_summary = gate.get("candidate_summary", {})
        if isinstance(candidate_summary, dict) and candidate_summary:
            lines.append(f"- Expected doc hit rate: `{candidate_summary.get('expected_doc_hit_rate')}`")
            lines.append(f"- Expected header hit rate: `{candidate_summary.get('expected_header_hit_rate')}`")
            lines.append(f"- Top-1 expected doc hit rate: `{candidate_summary.get('top1_expected_doc_hit_rate')}`")
            lines.append(f"- Top-1 expected header hit rate: `{candidate_summary.get('top1_expected_header_hit_rate')}`")
            lines.append(f"- Average doc precision: `{candidate_summary.get('average_doc_precision')}`")
            lines.append(f"- Cross-document average doc precision: `{candidate_summary.get('cross_document_average_doc_precision')}`")
        for comparison in gate.get("comparisons", []):
            lines.append(
                f"- Metric `{comparison['metric']}`: candidate `{comparison['candidate']}` "
                f"baseline `{comparison['baseline']}` delta `{comparison['delta']}`"
            )
        lines.extend(_render_issue_lines(gate.get("issues", [])))
        lines.append("")

    manual_gate = report["manual_spot_check_gate"]
    lines.extend(
        [
            "## Manual Spot Checks",
            "",
            f"- Status: `{manual_gate['status']}`",
            f"- Report: `{manual_gate.get('path', '')}`",
            f"- Checks total: `{manual_gate['checks_total']}`",
            f"- Passed: `{manual_gate['pass_count']}`",
            f"- Failed: `{manual_gate['fail_count']}`",
        ]
    )
    lines.extend(_render_issue_lines(manual_gate.get("issues", [])))
    for check in manual_gate.get("checks", []):
        repeated = "yes" if check.get("repeated") else "no"
        lines.append(
            f"- `{check['status']}` query: {check['query']} | repeated: `{repeated}`"
        )
    if promotion_alias:
        lines.extend(
            [
                "",
                "## Promotion",
                "",
                f"- Promotion alias: `{promotion_alias}`",
            ]
        )
        if promotion_command:
            lines.extend(
                [
                    "- Promotion command:",
                    "```powershell",
                    promotion_command,
                    "```",
                ]
            )
        else:
            lines.append(
                f"- Promotion command withheld because overall status is `{report['overall_status']}`."
            )
    lines.append("")
    return "\n".join(lines)


def _render_gate_line(label: str, gate: dict[str, Any]) -> str:
    issue_count = len(gate.get("issues", []))
    return f"- {label}: `{gate['status']}` ({issue_count} issue{'s' if issue_count != 1 else ''})"


def _render_issue_lines(issues: list[str]) -> list[str]:
    if not issues:
        return []
    return [f"- Issue: {issue}" for issue in issues]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_path = resolve_manifest_path(args.collection, args.manifest)
    rebuild_failure_path = resolve_rebuild_failure_path(args.collection, args.rebuild_failures)
    audit_path = resolve_audit_path(args.collection, args.audit_json)
    json_out = resolve_json_output_path(args.collection, args.json_out)
    md_out = resolve_markdown_output_path(args.collection, args.md_out)

    manifest_payload = load_json_object(manifest_path, label="manifest", required=False)
    rebuild_failure_payload = load_json_object(rebuild_failure_path, label="rebuild failure report", required=False)
    audit_payload = load_json_object(audit_path, label="audit report", required=False)

    eval_inputs = []
    for label, candidate_arg, baseline_arg in (
        ("stable", args.stable_eval_json, args.baseline_stable_eval_json),
        ("expanded", args.expanded_eval_json, args.baseline_expanded_eval_json),
        ("ood", args.ood_eval_json, args.baseline_ood_eval_json),
        ("runtime", args.runtime_eval_json, args.baseline_runtime_eval_json),
    ):
        candidate_path = Path(candidate_arg)
        baseline_path = Path(baseline_arg) if baseline_arg.strip() else None
        eval_inputs.append(
            {
                "label": label,
                "candidate_path": candidate_path,
                "candidate_payload": load_json_object(candidate_path, label=f"{label} evaluation report", required=False),
                "baseline_path": baseline_path,
                "baseline_payload": (
                    load_json_object(baseline_path, label=f"{label} baseline evaluation report", required=True)
                    if baseline_path is not None
                    else None
                ),
            }
        )

    manual_path = Path(args.manual_spot_checks) if args.manual_spot_checks.strip() else None
    report = build_rollout_report(
        collection=args.collection,
        stage_label=args.stage_label,
        target_pdf_count=args.target_pdf_count,
        max_metric_drop=args.max_metric_drop,
        manifest_payload=manifest_payload,
        manifest_path=manifest_path,
        rebuild_failure_payload=rebuild_failure_payload,
        rebuild_failure_path=rebuild_failure_path,
        audit_payload=audit_payload,
        audit_path=audit_path,
        eval_inputs=eval_inputs,
        manual_spot_check_path=manual_path,
        promotion_alias=args.promotion_alias,
        qdrant_url="http://localhost:6333",
        promotion_source_manifest=args.promotion_source_manifest,
        promotion_registry=args.promotion_registry,
        promotion_json_out=args.promotion_json_out,
    )
    markdown = render_markdown_report(report)
    write_json(report, json_out)
    write_markdown(markdown, md_out)

    print(f"Stage: {args.stage_label}")
    print(f"Collection: {args.collection}")
    print(f"Overall status: {report['overall_status']}")
    print(f"JSON output: {json_out}")
    print(f"Markdown output: {md_out}")
    return 0 if report["overall_status"] == PASS_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
