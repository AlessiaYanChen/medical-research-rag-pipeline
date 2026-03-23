from __future__ import annotations

from collections import Counter
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
            },
        )
        source_file = str(payload_dict.get("source_file", "")).strip()
        local_file = str(payload_dict.get("local_file", "")).strip()
        if source_file and not entry["source_file"]:
            entry["source_file"] = source_file
        if local_file and not entry["local_file"]:
            entry["local_file"] = local_file
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
    for field in ("source_file", "local_file"):
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

    issue_counter = Counter(issue["issue_type"] for issue in issues)
    return {
        "docs_total": len(all_doc_ids),
        "issue_count": len(issues),
        "issue_types": dict(issue_counter),
        "issues": issues,
    }


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
