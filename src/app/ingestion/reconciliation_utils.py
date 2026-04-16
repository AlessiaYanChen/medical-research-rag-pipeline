from __future__ import annotations

from collections import Counter
import statistics
from pathlib import Path
from typing import Any


def build_qdrant_doc_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        doc_id = str(record.get("doc_id", "")).strip()
        if not doc_id:
            continue
        chunk_type = str(record.get("chunk_type", "")).strip().lower()
        entry = summary.setdefault(
            doc_id,
            {
                "chunks": 0,
                "text_chunks": 0,
                "table_chunks": 0,
            },
        )
        entry["chunks"] += 1
        if chunk_type == "table":
            entry["table_chunks"] += 1
        else:
            entry["text_chunks"] += 1
    return summary


def build_manifest_doc_summary(manifest_payload: dict[str, Any]) -> dict[str, dict[str, int]]:
    docs = manifest_payload.get("docs", [])
    if not isinstance(docs, list):
        return {}

    summary: dict[str, dict[str, int]] = {}
    for item in docs:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id", "")).strip()
        if not doc_id:
            continue
        summary[doc_id] = {
            "chunks": int(item.get("chunk_count", 0)),
            "text_chunks": int(item.get("text_chunk_count", 0)),
            "table_chunks": int(item.get("table_chunk_count", 0)),
        }
    return summary


def build_registry_doc_summary(collection_entry: dict[str, Any]) -> dict[str, dict[str, int]]:
    docs = collection_entry.get("docs", collection_entry)
    if not isinstance(docs, dict):
        return {}

    summary: dict[str, dict[str, int]] = {}
    for doc_id, item in docs.items():
        if not isinstance(item, dict):
            continue
        normalized_doc_id = str(item.get("doc_id", doc_id)).strip()
        if not normalized_doc_id:
            continue
        summary[normalized_doc_id] = {
            "chunks": int(item.get("chunks", 0)),
            "text_chunks": int(item.get("text_chunks", 0)),
            "table_chunks": int(item.get("table_chunks", 0)),
        }
    return summary


def build_qdrant_doc_identities(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    identities: dict[str, dict[str, str]] = {}
    for record in records:
        doc_id = str(record.get("doc_id", "")).strip()
        if not doc_id:
            continue
        payload = record.get("payload", {})
        payload_dict = payload if isinstance(payload, dict) else {}
        entry = identities.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "source_file": "",
                "local_file": "",
                "source_sha256": "",
            },
        )
        source_file = str(payload_dict.get("source_file", "")).strip()
        local_file = str(payload_dict.get("local_file", "")).strip()
        source_sha256 = str(payload_dict.get("source_sha256", "")).strip()
        if source_file and not entry["source_file"]:
            entry["source_file"] = source_file
        if local_file and not entry["local_file"]:
            entry["local_file"] = local_file
        if source_sha256 and not entry["source_sha256"]:
            entry["source_sha256"] = source_sha256
    return sorted(identities.values(), key=lambda item: item["doc_id"].lower())


def build_manifest_doc_identities(manifest_payload: dict[str, Any]) -> list[dict[str, str]]:
    docs = manifest_payload.get("docs", [])
    if not isinstance(docs, list):
        return []

    identities: list[dict[str, str]] = []
    for item in docs:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id", "")).strip()
        if not doc_id:
            continue
        identities.append(
            {
                "doc_id": doc_id,
                "source_file": str(item.get("source_file", "")).strip(),
                "local_file": str(item.get("local_file", "")).strip(),
                "source_sha256": str(item.get("source_sha256", "")).strip(),
            }
        )
    return identities


def build_registry_doc_identities(collection_entry: dict[str, Any]) -> list[dict[str, str]]:
    docs = collection_entry.get("docs", collection_entry)
    if not isinstance(docs, dict):
        return []

    identities: list[dict[str, str]] = []
    for doc_id, item in docs.items():
        if not isinstance(item, dict):
            continue
        normalized_doc_id = str(item.get("doc_id", doc_id)).strip()
        if not normalized_doc_id:
            continue
        identities.append(
            {
                "doc_id": normalized_doc_id,
                "source_file": str(item.get("source_file", "")).strip(),
                "local_file": str(item.get("pdf_path", item.get("local_file", ""))).strip(),
                "source_sha256": str(item.get("source_sha256", "")).strip(),
            }
        )
    return sorted(identities, key=lambda item: item["doc_id"].lower())


def find_duplicate_identity_issues(
    entries: list[dict[str, Any]],
    *,
    source_name: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    issues.extend(_find_duplicate_identity_issues_for_field(entries, source_name=source_name, field="doc_id"))
    issues.extend(_find_duplicate_identity_issues_for_field(entries, source_name=source_name, field="source_file"))
    issues.extend(_find_duplicate_identity_issues_for_field(entries, source_name=source_name, field="local_file"))
    issues.extend(_find_duplicate_identity_issues_for_field(entries, source_name=source_name, field="source_sha256"))
    return issues


def build_duplicate_cleanup_plan(
    *,
    qdrant_identities: list[dict[str, Any]],
    manifest_identities: list[dict[str, Any]],
    registry_identities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    combined_entries = [
        *(_annotate_source(qdrant_identities, source_name="qdrant")),
        *(_annotate_source(manifest_identities, source_name="manifest")),
        *(_annotate_source(registry_identities, source_name="registry")),
    ]

    plan: list[dict[str, Any]] = []
    for field in ("source_file", "local_file", "source_sha256"):
        grouped = _group_entries_by_identity_value(combined_entries, field=field)
        for normalized_value, entries in sorted(grouped.items()):
            unique_doc_ids = sorted({str(entry.get("doc_id", "")).strip() for entry in entries if str(entry.get("doc_id", "")).strip()})
            if len(unique_doc_ids) < 2:
                continue

            value = next(
                (str(entry.get(field, "")).strip() for entry in entries if str(entry.get(field, "")).strip()),
                normalized_value,
            )
            sources_by_doc_id: dict[str, set[str]] = {}
            for entry in entries:
                doc_id = str(entry.get("doc_id", "")).strip()
                source_name = str(entry.get("source", "")).strip()
                if not doc_id or not source_name:
                    continue
                sources_by_doc_id.setdefault(doc_id, set()).add(source_name)

            keep_doc_id = _choose_canonical_doc_id(sources_by_doc_id)
            if keep_doc_id is None:
                plan.append(
                    {
                        "action": "manual_review",
                        "reason": "no canonical doc_id established across sources",
                        "field": field,
                        "value": value,
                        "doc_ids": unique_doc_ids,
                        "sources_by_doc_id": {
                            doc_id: sorted(source_names)
                            for doc_id, source_names in sorted(sources_by_doc_id.items())
                        },
                    }
                )
                continue

            plan.append(
                {
                    "action": "drop_duplicate_doc_ids",
                    "reason": "canonical doc_id supported by the broadest source coverage",
                    "field": field,
                    "value": value,
                    "keep_doc_id": keep_doc_id,
                    "drop_doc_ids": [doc_id for doc_id in unique_doc_ids if doc_id != keep_doc_id],
                    "sources_by_doc_id": {
                        doc_id: sorted(source_names)
                        for doc_id, source_names in sorted(sources_by_doc_id.items())
                    },
                }
            )

    return plan


def reconcile_collection_state(
    *,
    qdrant_docs: dict[str, dict[str, int]],
    manifest_docs: dict[str, dict[str, int]],
    registry_docs: dict[str, dict[str, int]],
    qdrant_identities: list[dict[str, Any]] | None = None,
    manifest_identities: list[dict[str, Any]] | None = None,
    registry_identities: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    all_doc_ids = sorted(set(qdrant_docs) | set(manifest_docs) | set(registry_docs))
    issues: list[dict[str, Any]] = []

    for doc_id in all_doc_ids:
        qdrant = qdrant_docs.get(doc_id)
        manifest = manifest_docs.get(doc_id)
        registry = registry_docs.get(doc_id)

        if qdrant is None or manifest is None or registry is None:
            issues.append(
                {
                    "doc_id": doc_id,
                    "issue_type": "missing_doc",
                    "present_in": [
                        name
                        for name, source in (
                            ("qdrant", qdrant),
                            ("manifest", manifest),
                            ("registry", registry),
                        )
                        if source is not None
                    ],
                }
            )
            continue

        mismatched_fields = [
            field
            for field in ("chunks", "text_chunks", "table_chunks")
            if len({qdrant[field], manifest[field], registry[field]}) > 1
        ]
        if mismatched_fields:
            issues.append(
                {
                    "doc_id": doc_id,
                    "issue_type": "count_mismatch",
                    "fields": mismatched_fields,
                    "qdrant": qdrant,
                    "manifest": manifest,
                    "registry": registry,
                }
            )

    for source_name, entries in (
        ("qdrant", qdrant_identities or []),
        ("manifest", manifest_identities or []),
        ("registry", registry_identities or []),
    ):
        issues.extend(find_duplicate_identity_issues(entries, source_name=source_name))

    issues.extend(
        find_doc_metadata_mismatch_issues(
            qdrant_identities=qdrant_identities or [],
            manifest_identities=manifest_identities or [],
            registry_identities=registry_identities or [],
        )
    )

    for source_name, summary in (
        ("qdrant", qdrant_docs),
        ("manifest", manifest_docs),
        ("registry", registry_docs),
    ):
        issues.extend(find_chunk_count_sanity_issues(summary, source_name=source_name))

    issue_counter = Counter(issue["issue_type"] for issue in issues)
    repair_plan = build_reconciliation_repair_plan(issues)
    return {
        "docs_total": len(all_doc_ids),
        "issue_count": len(issues),
        "issue_types": dict(issue_counter),
        "issues": issues,
        "repair_plan_count": len(repair_plan),
        "repair_plan": repair_plan,
    }


def find_chunk_count_sanity_issues(
    doc_summary: dict[str, dict[str, int]],
    *,
    source_name: str,
) -> list[dict[str, Any]]:
    if not doc_summary:
        return []

    issues: list[dict[str, Any]] = []
    chunk_counts = [
        int(summary.get("chunks", 0))
        for summary in doc_summary.values()
        if isinstance(summary, dict)
    ]
    positive_chunk_counts = [count for count in chunk_counts if count > 0]
    median_chunk_count = statistics.median(positive_chunk_counts) if positive_chunk_counts else 0.0
    low_outlier_threshold = max(1, int(median_chunk_count * 0.2)) if median_chunk_count else 0
    high_outlier_threshold = int(median_chunk_count * 5) if median_chunk_count else 0
    enable_outlier_checks = len(positive_chunk_counts) >= 5 and median_chunk_count > 0

    for doc_id, summary in sorted(doc_summary.items()):
        if not isinstance(summary, dict):
            continue
        chunks = int(summary.get("chunks", 0))
        text_chunks = int(summary.get("text_chunks", 0))
        table_chunks = int(summary.get("table_chunks", 0))
        checks: list[str] = []

        if chunks <= 0:
            checks.append("no_chunks")
        if text_chunks < 0 or table_chunks < 0:
            checks.append("negative_chunk_count")
        if text_chunks + table_chunks != chunks:
            checks.append("count_breakdown_mismatch")
        if chunks > 0 and text_chunks <= 0:
            checks.append("no_text_chunks")
        if enable_outlier_checks and chunks > 0:
            if chunks <= low_outlier_threshold:
                checks.append("unusually_low_chunk_count")
            if chunks >= high_outlier_threshold:
                checks.append("unusually_high_chunk_count")

        if checks:
            issues.append(
                {
                    "issue_type": "chunk_count_sanity",
                    "source": source_name,
                    "doc_id": doc_id,
                    "checks": checks,
                    "summary": {
                        "chunks": chunks,
                        "text_chunks": text_chunks,
                        "table_chunks": table_chunks,
                    },
                    "median_chunk_count": median_chunk_count,
                    "low_outlier_threshold": low_outlier_threshold,
                    "high_outlier_threshold": high_outlier_threshold,
                }
            )

    return issues


def find_doc_metadata_mismatch_issues(
    *,
    qdrant_identities: list[dict[str, Any]],
    manifest_identities: list[dict[str, Any]],
    registry_identities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_source = {
        "qdrant": _index_identities_by_doc_id(qdrant_identities),
        "manifest": _index_identities_by_doc_id(manifest_identities),
        "registry": _index_identities_by_doc_id(registry_identities),
    }
    all_doc_ids = sorted(set().union(*[set(items.keys()) for items in by_source.values()]))
    issues: list[dict[str, Any]] = []

    for doc_id in all_doc_ids:
        present_sources = {
            source_name: items[doc_id]
            for source_name, items in by_source.items()
            if doc_id in items
        }
        if len(present_sources) < 2:
            continue

        mismatched_fields: list[str] = []
        field_values: dict[str, dict[str, str]] = {}
        for field in ("source_file", "local_file", "source_sha256"):
            source_values = {
                source_name: str(entry.get(field, "")).strip()
                for source_name, entry in present_sources.items()
            }
            non_empty_values = {value for value in source_values.values() if value}
            if len(non_empty_values) > 1:
                mismatched_fields.append(field)
                field_values[field] = source_values

        if mismatched_fields:
            issues.append(
                {
                    "issue_type": "metadata_mismatch",
                    "doc_id": doc_id,
                    "fields": mismatched_fields,
                    "sources": field_values,
                }
            )

    return issues


def build_reconciliation_repair_plan(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for issue in issues:
        issue_type = str(issue.get("issue_type", "")).strip()
        if issue_type == "missing_doc":
            present_in = [str(item).strip() for item in issue.get("present_in", []) if str(item).strip()]
            if "manifest" in present_in and "registry" not in present_in:
                plan.append(
                    {
                        "action": "sync_registry_from_manifest",
                        "doc_id": str(issue.get("doc_id", "")).strip(),
                        "reason": "manifest and registry are out of sync for an otherwise indexed document",
                    }
                )
        elif issue_type == "metadata_mismatch":
            plan.append(
                {
                    "action": "review_doc_metadata",
                    "doc_id": str(issue.get("doc_id", "")).strip(),
                    "fields": list(issue.get("fields", [])),
                    "reason": "doc metadata differs across sources and needs canonical reconciliation",
                }
            )
        elif issue_type == "count_mismatch":
            plan.append(
                {
                    "action": "review_doc_counts",
                    "doc_id": str(issue.get("doc_id", "")).strip(),
                    "fields": list(issue.get("fields", [])),
                    "reason": "chunk counts disagree across sources for the same doc_id",
                }
            )
        elif issue_type == "chunk_count_sanity":
            plan.append(
                {
                    "action": "inspect_parser_output",
                    "doc_id": str(issue.get("doc_id", "")).strip(),
                    "source": str(issue.get("source", "")).strip(),
                    "checks": list(issue.get("checks", [])),
                    "reason": "chunk profile looks structurally implausible for this document",
                }
            )
    return plan


def _find_duplicate_identity_issues_for_field(
    entries: list[dict[str, Any]],
    *,
    source_name: str,
    field: str,
) -> list[dict[str, Any]]:
    values_to_doc_ids: dict[str, set[str]] = {}
    values_to_display: dict[str, str] = {}

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        doc_id = str(entry.get("doc_id", "")).strip()
        value = str(entry.get(field, "")).strip()
        if not doc_id or not value:
            continue

        normalized_value = _normalize_identity_value(field, value)
        if not normalized_value:
            continue
        values_to_doc_ids.setdefault(normalized_value, set()).add(doc_id)
        values_to_display.setdefault(normalized_value, value)

    issues: list[dict[str, Any]] = []
    for normalized_value, doc_ids in sorted(values_to_doc_ids.items()):
        if len(doc_ids) < 2:
            continue
        issues.append(
            {
                "issue_type": "duplicate_identity",
                "source": source_name,
                "field": field,
                "value": values_to_display[normalized_value],
                "doc_ids": sorted(doc_ids),
            }
        )
    return issues


def _annotate_source(entries: list[dict[str, Any]], *, source_name: str) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        annotated.append({**entry, "source": source_name})
    return annotated


def _index_identities_by_doc_id(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        doc_id = str(entry.get("doc_id", "")).strip()
        if not doc_id or doc_id in indexed:
            continue
        indexed[doc_id] = entry
    return indexed


def _group_entries_by_identity_value(
    entries: list[dict[str, Any]],
    *,
    field: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        value = str(entry.get(field, "")).strip()
        if not value:
            continue
        normalized_value = _normalize_identity_value(field, value)
        if not normalized_value:
            continue
        grouped.setdefault(normalized_value, []).append(entry)
    return grouped


def _choose_canonical_doc_id(sources_by_doc_id: dict[str, set[str]]) -> str | None:
    ranked = sorted(
        ((len(source_names), doc_id) for doc_id, source_names in sources_by_doc_id.items()),
        reverse=True,
    )
    if not ranked:
        return None

    top_score = ranked[0][0]
    top_doc_ids = [doc_id for score_count, doc_id in ranked if score_count == top_score]
    if len(top_doc_ids) != 1:
        return None
    return top_doc_ids[0]


def _normalize_identity_value(field: str, value: str) -> str:
    if field == "local_file":
        raw = str(value).strip()
        if not raw:
            return ""
        normalized = raw.replace("\\", "/")
        return str(Path(normalized)).replace("\\", "/").casefold()
    return " ".join(str(value).strip().split()).casefold()
