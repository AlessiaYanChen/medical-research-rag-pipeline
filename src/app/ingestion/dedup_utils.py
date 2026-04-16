from __future__ import annotations

from pathlib import Path
from typing import Any


class DuplicateDocumentError(ValueError):
    """Raised when document metadata would create an ambiguous duplicate entry."""


def build_doc_identity(
    *,
    doc_id: str,
    source_file: str = "",
    local_file: str = "",
    source_sha256: str = "",
) -> dict[str, str]:
    return {
        "doc_id": str(doc_id).strip(),
        "source_file": str(source_file).strip(),
        "local_file": str(local_file).strip(),
        "source_sha256": str(source_sha256).strip(),
    }


def validate_unique_doc_identities(
    entries: list[dict[str, Any]],
    *,
    context: str,
    local_file_keys: tuple[str, ...] = ("local_file", "pdf_path"),
) -> None:
    seen_doc_ids: dict[str, str] = {}
    seen_source_files: dict[str, str] = {}
    seen_local_files: dict[str, str] = {}
    seen_source_hashes: dict[str, str] = {}

    for entry in _iter_doc_identities(entries, local_file_keys=local_file_keys):
        doc_id = entry["doc_id"]
        if not doc_id:
            continue

        normalized_doc_id = _normalize_identity_text(doc_id)
        if normalized_doc_id in seen_doc_ids:
            raise DuplicateDocumentError(
                f"{context}: duplicate doc_id '{doc_id}' for entries '{seen_doc_ids[normalized_doc_id]}' and '{doc_id}'."
            )
        seen_doc_ids[normalized_doc_id] = doc_id

        source_file = entry["source_file"]
        if source_file:
            normalized_source_file = _normalize_identity_text(source_file)
            if normalized_source_file in seen_source_files:
                raise DuplicateDocumentError(
                    f"{context}: source_file '{source_file}' is already registered to doc_id "
                    f"'{seen_source_files[normalized_source_file]}'."
                )
            seen_source_files[normalized_source_file] = doc_id

        local_file = entry["local_file"]
        if local_file:
            normalized_local_file = _normalize_identity_path(local_file)
            if normalized_local_file in seen_local_files:
                raise DuplicateDocumentError(
                    f"{context}: local_file '{local_file}' is already registered to doc_id "
                    f"'{seen_local_files[normalized_local_file]}'."
                )
            seen_local_files[normalized_local_file] = doc_id

        source_sha256 = entry["source_sha256"]
        if source_sha256:
            normalized_source_sha256 = _normalize_identity_text(source_sha256)
            if normalized_source_sha256 in seen_source_hashes:
                raise DuplicateDocumentError(
                    f"{context}: source_sha256 '{source_sha256}' is already registered to doc_id "
                    f"'{seen_source_hashes[normalized_source_sha256]}'."
                )
            seen_source_hashes[normalized_source_sha256] = doc_id


def ensure_doc_identity_is_available(
    *,
    doc_id: str,
    source_file: str = "",
    local_file: str = "",
    source_sha256: str = "",
    existing_entries: list[dict[str, Any]],
    context: str,
    allowed_doc_ids: set[str] | None = None,
    local_file_keys: tuple[str, ...] = ("local_file", "pdf_path"),
) -> None:
    allowed_normalized_doc_ids = {
        _normalize_identity_text(value)
        for value in (allowed_doc_ids or set())
        if str(value).strip()
    }
    normalized_doc_id = _normalize_identity_text(doc_id)
    normalized_source_file = _normalize_identity_text(source_file)
    normalized_local_file = _normalize_identity_path(local_file)
    normalized_source_sha256 = _normalize_identity_text(source_sha256)

    for entry in _iter_doc_identities(existing_entries, local_file_keys=local_file_keys):
        existing_doc_id = entry["doc_id"]
        if not existing_doc_id:
            continue

        normalized_existing_doc_id = _normalize_identity_text(existing_doc_id)
        if normalized_existing_doc_id in allowed_normalized_doc_ids:
            continue

        if normalized_existing_doc_id == normalized_doc_id:
            raise DuplicateDocumentError(
                f"{context}: doc_id '{doc_id}' already exists in doc_id '{existing_doc_id}'. "
                "Use the single-document repair flow instead of ingesting a second copy."
            )

        existing_source_file = entry["source_file"]
        if normalized_source_file and _normalize_identity_text(existing_source_file) == normalized_source_file:
            raise DuplicateDocumentError(
                f"{context}: source_file '{source_file}' is already registered to doc_id '{existing_doc_id}'."
            )

        existing_local_file = entry["local_file"]
        if normalized_local_file and _normalize_identity_path(existing_local_file) == normalized_local_file:
            raise DuplicateDocumentError(
                f"{context}: local_file '{local_file}' is already registered to doc_id '{existing_doc_id}'."
            )

        existing_source_sha256 = entry["source_sha256"]
        if normalized_source_sha256 and _normalize_identity_text(existing_source_sha256) == normalized_source_sha256:
            raise DuplicateDocumentError(
                f"{context}: source_sha256 '{source_sha256}' is already registered to doc_id '{existing_doc_id}'."
            )


def fetch_collection_doc_identities(
    client: Any,
    *,
    collection_name: str,
    batch_size: int = 256,
) -> list[dict[str, str]]:
    if not client.collection_exists(collection_name):
        return []

    identities: dict[str, dict[str, str]] = {}
    offset: Any = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=None,
            limit=max(1, batch_size),
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if not points:
            break

        for point in points:
            payload = dict(getattr(point, "payload", {}) or {})
            doc_id = str(payload.get("doc_id", "")).strip()
            if not doc_id:
                continue
            entry = identities.setdefault(doc_id, build_doc_identity(doc_id=doc_id))
            source_file = str(payload.get("source_file", "")).strip()
            local_file = str(payload.get("local_file", "")).strip()
            source_sha256 = str(payload.get("source_sha256", "")).strip()
            if source_file and not entry["source_file"]:
                entry["source_file"] = source_file
            if local_file and not entry["local_file"]:
                entry["local_file"] = local_file
            if source_sha256 and not entry["source_sha256"]:
                entry["source_sha256"] = source_sha256

        if next_offset is None:
            break
        offset = next_offset

    return sorted(identities.values(), key=lambda item: item["doc_id"].lower())


def _iter_doc_identities(
    entries: list[dict[str, Any]],
    *,
    local_file_keys: tuple[str, ...],
) -> list[dict[str, str]]:
    identities: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        local_file = ""
        for key in local_file_keys:
            value = str(entry.get(key, "")).strip()
            if value:
                local_file = value
                break
        identities.append(
            build_doc_identity(
                doc_id=str(entry.get("doc_id", "")).strip(),
                source_file=str(entry.get("source_file", "")).strip(),
                local_file=local_file,
                source_sha256=str(entry.get("source_sha256", "")).strip(),
            )
        )
    return identities


def _normalize_identity_text(value: str) -> str:
    return " ".join(str(value).strip().split()).casefold()


def _normalize_identity_path(value: str) -> str:
    raw = str(value).strip()
    if not raw:
        return ""
    normalized = raw.replace("\\", "/")
    return str(Path(normalized)).replace("\\", "/").casefold()
