from __future__ import annotations

import json
from pathlib import Path

from scripts.promote_collection_alias import (
    build_alias_manifest_payload,
    load_manifest_json_object,
    resolve_alias_manifest_path,
    resolve_source_manifest_path,
    write_alias_manifest,
)


def test_resolve_manifest_paths_use_expected_defaults() -> None:
    assert resolve_source_manifest_path("stage_v2_build", "") == Path(
        "data/ingestion_manifests/stage_v2_build_rebuild_manifest.json"
    )
    assert resolve_alias_manifest_path("medical_research_chunks_active", "") == Path(
        "data/ingestion_manifests/medical_research_chunks_active_rebuild_manifest.json"
    )


def test_load_manifest_json_object_reads_object_payload(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"collection": "stage_v2_build"}), encoding="utf-8")

    payload = load_manifest_json_object(manifest_path)

    assert payload == {"collection": "stage_v2_build"}


def test_build_alias_manifest_payload_rewrites_collection_name() -> None:
    payload = build_alias_manifest_payload(
        {
            "collection": "stage_v2_build",
            "doc_count": 20,
            "chunk_count": 400,
        },
        alias="medical_research_chunks_active",
    )

    assert payload["collection"] == "medical_research_chunks_active"
    assert payload["doc_count"] == 20


def test_write_alias_manifest_writes_json_payload(tmp_path: Path) -> None:
    output_path = tmp_path / "alias_manifest.json"

    written_path = write_alias_manifest(
        alias_manifest_path=output_path,
        payload={"collection": "medical_research_chunks_active", "doc_count": 20},
    )

    assert written_path == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "collection": "medical_research_chunks_active",
        "doc_count": 20,
    }
