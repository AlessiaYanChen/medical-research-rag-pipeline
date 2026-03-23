from __future__ import annotations

from collections import Counter
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


def reconcile_collection_state(
    *,
    qdrant_docs: dict[str, dict[str, int]],
    manifest_docs: dict[str, dict[str, int]],
    registry_docs: dict[str, dict[str, int]],
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

    issue_counter = Counter(issue["issue_type"] for issue in issues)
    return {
        "docs_total": len(all_doc_ids),
        "issue_count": len(issues),
        "issue_types": dict(issue_counter),
        "issues": issues,
    }
