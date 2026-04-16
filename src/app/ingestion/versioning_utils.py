from __future__ import annotations

from typing import Any


def resolve_chunker_version(payload: dict[str, Any]) -> str:
    return str(
        payload.get("chunker_version", payload.get("chunking_version", ""))
    ).strip()


def build_version_metadata(
    *,
    ingestion_version: str,
    chunker_version: str,
) -> dict[str, str]:
    normalized_ingestion_version = str(ingestion_version).strip()
    normalized_chunker_version = str(chunker_version).strip()
    return {
        "ingestion_version": normalized_ingestion_version,
        "chunker_version": normalized_chunker_version,
        # Keep the legacy field for backward compatibility with existing manifests/tests.
        "chunking_version": normalized_chunker_version,
    }


def validate_manifest_compatibility(
    manifest_payload: dict[str, Any],
    *,
    expected_collection: str,
    expected_ingestion_version: str,
    expected_chunking_version: str,
) -> list[str]:
    issues: list[str] = []

    manifest_collection = str(manifest_payload.get("collection", "")).strip()
    if manifest_collection and manifest_collection != expected_collection:
        issues.append(
            f"Manifest collection mismatch: expected '{expected_collection}', found '{manifest_collection}'."
        )

    manifest_ingestion_version = str(manifest_payload.get("ingestion_version", "")).strip()
    if manifest_ingestion_version and manifest_ingestion_version != expected_ingestion_version:
        issues.append(
            f"Manifest ingestion_version mismatch: expected '{expected_ingestion_version}', found '{manifest_ingestion_version}'."
        )

    manifest_chunking_version = resolve_chunker_version(manifest_payload)
    if manifest_chunking_version and manifest_chunking_version != expected_chunking_version:
        issues.append(
            f"Manifest chunker_version mismatch: expected '{expected_chunking_version}', found '{manifest_chunking_version}'."
        )

    return issues
