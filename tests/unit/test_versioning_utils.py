from __future__ import annotations

from src.app.ingestion.versioning_utils import validate_manifest_compatibility


def test_validate_manifest_compatibility_accepts_matching_metadata() -> None:
    issues = validate_manifest_compatibility(
        {
            "collection": "medical_research_chunks_v1",
            "ingestion_version": "ingestion_v2",
            "chunker_version": "chunking_v2",
            "chunking_version": "chunking_v2",
        },
        expected_collection="medical_research_chunks_v1",
        expected_ingestion_version="ingestion_v2",
        expected_chunking_version="chunking_v2",
    )

    assert issues == []


def test_validate_manifest_compatibility_reports_collection_and_version_mismatches() -> None:
    issues = validate_manifest_compatibility(
        {
            "collection": "medical_research_chunks_old",
            "ingestion_version": "ingestion_v1",
            "chunker_version": "chunking_v1",
            "chunking_version": "chunking_v1",
        },
        expected_collection="medical_research_chunks_v1",
        expected_ingestion_version="ingestion_v2",
        expected_chunking_version="chunking_v2",
    )

    assert len(issues) == 3
    assert any("collection mismatch" in issue.lower() for issue in issues)
    assert any("ingestion_version mismatch" in issue.lower() for issue in issues)
    assert any("chunker_version mismatch" in issue.lower() for issue in issues)


def test_validate_manifest_compatibility_accepts_legacy_chunking_version_field() -> None:
    issues = validate_manifest_compatibility(
        {
            "collection": "medical_research_chunks_v1",
            "ingestion_version": "ingestion_v2",
            "chunking_version": "chunking_v2",
        },
        expected_collection="medical_research_chunks_v1",
        expected_ingestion_version="ingestion_v2",
        expected_chunking_version="chunking_v2",
    )

    assert issues == []
