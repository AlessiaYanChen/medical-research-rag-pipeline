from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_project_root_on_path()

from qdrant_client import QdrantClient  # noqa: E402

from src.app.ingestion.reconciliation_utils import (  # noqa: E402
    build_duplicate_cleanup_plan,
    build_manifest_doc_identities,
    build_manifest_doc_summary,
    build_qdrant_doc_identities,
    build_qdrant_doc_summary,
    build_registry_doc_identities,
    build_registry_doc_summary,
    reconcile_collection_state,
)
from src.app.ingestion.registry_utils import (  # noqa: E402
    default_manifest_path_for_collection,
    load_registry,
    save_registry,
    sync_collection_from_manifest,
)
from src.app.ingestion.versioning_utils import validate_manifest_compatibility  # noqa: E402
from src.app.tables.table_chunker import UnifiedChunker  # noqa: E402
from scripts.export_qdrant_chunks import fetch_points  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a collection across Qdrant, rebuild manifest, and local registry."
    )
    parser.add_argument(
        "--collection",
        default="medical_research_chunks_v1",
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant base URL.",
    )
    parser.add_argument(
        "--registry",
        default="data/kb_registry.json",
        help="Path to local registry JSON.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional manifest path override. Defaults to data/ingestion_manifests/<collection>_rebuild_manifest.json.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write the audit payload as JSON.",
    )
    parser.add_argument(
        "--cleanup-plan-out",
        default="",
        help="Optional path to write a non-destructive duplicate cleanup plan as JSON.",
    )
    parser.add_argument(
        "--sync-registry",
        action="store_true",
        help="Update the local registry from the rebuild manifest before reporting.",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Return exit code 1 when the audit finds any version issue, reconciliation issue, or cleanup-plan step.",
    )
    return parser.parse_args()


def should_fail_audit(
    *,
    manifest_version_issues: list[str],
    issue_count: int,
    cleanup_plan_count: int,
) -> bool:
    return bool(manifest_version_issues) or issue_count > 0 or cleanup_plan_count > 0


def load_manifest_payload(manifest_path: Path) -> dict[str, object]:
    try:
        loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"manifest is not valid JSON: {manifest_path} "
            f"(line {exc.lineno}, column {exc.colno}: {exc.msg})"
        ) from exc
    if not isinstance(loaded_manifest, dict):
        raise ValueError(f"manifest is not a JSON object: {manifest_path}")
    return loaded_manifest


def main() -> int:
    args = parse_args()
    client = QdrantClient(url=args.qdrant_url)
    if not client.collection_exists(args.collection):
        print(f"ERROR: collection not found: {args.collection}")
        return 1

    qdrant_records = fetch_points(
        client=client,
        collection_name=args.collection,
        doc_id=None,
        batch_size=256,
        limit=0,
        include_vectors=False,
    )
    qdrant_docs = build_qdrant_doc_summary(qdrant_records)
    qdrant_identities = build_qdrant_doc_identities(qdrant_records)

    manifest_path = Path(args.manifest) if args.manifest.strip() else default_manifest_path_for_collection(args.collection)
    manifest_payload: dict[str, object] = {}
    if manifest_path.exists():
        try:
            manifest_payload = load_manifest_payload(manifest_path)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 1
    manifest_docs = build_manifest_doc_summary(manifest_payload)
    manifest_identities = build_manifest_doc_identities(manifest_payload)
    manifest_version_issues = (
        validate_manifest_compatibility(
            manifest_payload,
            expected_collection=args.collection,
            expected_ingestion_version=UnifiedChunker.INGESTION_VERSION,
            expected_chunking_version=UnifiedChunker.CHUNKING_VERSION,
        )
        if manifest_payload
        else []
    )

    registry = load_registry(args.registry)
    if args.sync_registry:
        sync_collection_from_manifest(
            registry,
            collection_name=args.collection,
            manifest_path=manifest_path,
        )
        save_registry(args.registry, registry)
    collection_entry = registry.get("collections", {}).get(args.collection, {})
    if not isinstance(collection_entry, dict):
        collection_entry = {}
    registry_docs = build_registry_doc_summary(collection_entry)
    registry_identities = build_registry_doc_identities(collection_entry)

    audit = reconcile_collection_state(
        qdrant_docs=qdrant_docs,
        manifest_docs=manifest_docs,
        registry_docs=registry_docs,
        qdrant_identities=qdrant_identities,
        manifest_identities=manifest_identities,
        registry_identities=registry_identities,
    )
    cleanup_plan = build_duplicate_cleanup_plan(
        qdrant_identities=qdrant_identities,
        manifest_identities=manifest_identities,
        registry_identities=registry_identities,
    )
    payload = {
        "collection": args.collection,
        "qdrant_url": args.qdrant_url,
        "registry_path": str(args.registry),
        "manifest_path": str(manifest_path),
        "qdrant_doc_count": len(qdrant_docs),
        "manifest_doc_count": len(manifest_docs),
        "registry_doc_count": len(registry_docs),
        "manifest_version_issues": manifest_version_issues,
        "cleanup_plan_count": len(cleanup_plan),
        "cleanup_plan": cleanup_plan,
        **audit,
    }

    if args.json_out.strip():
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.cleanup_plan_out.strip():
        cleanup_output_path = Path(args.cleanup_plan_out)
        cleanup_output_path.parent.mkdir(parents=True, exist_ok=True)
        cleanup_output_path.write_text(json.dumps(cleanup_plan, indent=2), encoding="utf-8")

    print(f"Collection: {args.collection}")
    print(f"Qdrant docs: {len(qdrant_docs)}")
    print(f"Manifest docs: {len(manifest_docs)}")
    print(f"Registry docs: {len(registry_docs)}")
    print(f"Issues: {payload['issue_count']}")
    print(f"Cleanup plan entries: {len(cleanup_plan)}")
    if args.sync_registry:
        print("Registry sync: applied from manifest before audit")
    if manifest_version_issues:
        print(f"Manifest version issues: {len(manifest_version_issues)}")
        for issue in manifest_version_issues:
            print(f"- {issue}")
    if payload["issue_types"]:
        print(f"Issue types: {payload['issue_types']}")
    for issue in payload["issues"]:
        if issue["issue_type"] == "missing_doc":
            print(f"- missing_doc: {issue['doc_id']} present_in={issue['present_in']}")
        elif issue["issue_type"] == "duplicate_identity":
            print(
                f"- duplicate_identity: source={issue['source']} "
                f"field={issue['field']} value={issue['value']} doc_ids={issue['doc_ids']}"
            )
        elif issue["issue_type"] == "chunk_count_sanity":
            print(
                f"- chunk_count_sanity: source={issue['source']} doc_id={issue['doc_id']} "
                f"checks={issue['checks']} summary={issue['summary']}"
            )
        elif issue["issue_type"] == "metadata_mismatch":
            print(
                f"- metadata_mismatch: doc_id={issue['doc_id']} "
                f"fields={issue['fields']} sources={issue['sources']}"
            )
        else:
            print(
                f"- count_mismatch: {issue['doc_id']} "
                f"fields={issue['fields']} "
                f"qdrant={issue['qdrant']} manifest={issue['manifest']} registry={issue['registry']}"
            )
    for step in cleanup_plan:
        if step["action"] == "drop_duplicate_doc_ids":
            print(
                f"- cleanup_plan: keep={step['keep_doc_id']} drop={step['drop_doc_ids']} "
                f"field={step['field']} value={step['value']}"
            )
        else:
            print(
                f"- cleanup_plan: manual_review field={step['field']} "
                f"value={step['value']} doc_ids={step['doc_ids']}"
            )
    for step in payload.get("repair_plan", []):
        if step["action"] == "sync_registry_from_manifest":
            print(f"- repair_plan: sync_registry_from_manifest doc_id={step['doc_id']}")
        elif step["action"] == "review_doc_metadata":
            print(f"- repair_plan: review_doc_metadata doc_id={step['doc_id']} fields={step['fields']}")
        elif step["action"] == "review_doc_counts":
            print(f"- repair_plan: review_doc_counts doc_id={step['doc_id']} fields={step['fields']}")
        elif step["action"] == "inspect_parser_output":
            print(
                f"- repair_plan: inspect_parser_output doc_id={step['doc_id']} "
                f"source={step['source']} checks={step['checks']}"
            )
    if args.json_out.strip():
        print(f"JSON output: {Path(args.json_out)}")
    if args.cleanup_plan_out.strip():
        print(f"Cleanup plan output: {Path(args.cleanup_plan_out)}")
    if args.fail_on_issues and should_fail_audit(
        manifest_version_issues=manifest_version_issues,
        issue_count=int(payload["issue_count"]),
        cleanup_plan_count=len(cleanup_plan),
    ):
        print("ERROR: audit gate failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
