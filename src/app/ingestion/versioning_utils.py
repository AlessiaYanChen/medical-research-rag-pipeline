from __future__ import annotations

from typing import Any


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

    manifest_chunking_version = str(manifest_payload.get("chunking_version", "")).strip()
    if manifest_chunking_version and manifest_chunking_version != expected_chunking_version:
        issues.append(
            f"Manifest chunking_version mismatch: expected '{expected_chunking_version}', found '{manifest_chunking_version}'."
        )

    return issues
