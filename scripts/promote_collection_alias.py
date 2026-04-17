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

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import (  # noqa: E402
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
)

from src.app.ingestion.registry_utils import (  # noqa: E402
    default_manifest_path_for_collection,
    load_registry,
    save_registry,
    sync_collection_from_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a staged Qdrant collection behind a stable alias and snapshot its manifest/registry state."
    )
    parser.add_argument("--source-collection", required=True, help="Existing staged Qdrant collection to promote.")
    parser.add_argument("--alias", required=True, help="Stable alias name to point at the staged collection.")
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant base URL.",
    )
    parser.add_argument(
        "--source-manifest",
        default="",
        help="Optional manifest path override for the staged collection.",
    )
    parser.add_argument(
        "--alias-manifest-out",
        default="",
        help="Optional path to write the alias-backed manifest snapshot.",
    )
    parser.add_argument(
        "--registry",
        default="data/kb_registry.json",
        help="Path to local registry JSON.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional JSON path to write the promotion summary.",
    )
    return parser.parse_args()


def resolve_source_manifest_path(source_collection: str, override: str) -> Path:
    if override.strip():
        return Path(override)
    return default_manifest_path_for_collection(source_collection)


def resolve_alias_manifest_path(alias: str, override: str) -> Path:
    if override.strip():
        return Path(override)
    return default_manifest_path_for_collection(alias)


def load_manifest_json_object(path: Path) -> dict[str, Any]:
    try:
        loaded_manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"manifest is not valid JSON: {path} "
            f"(line {exc.lineno}, column {exc.colno}: {exc.msg})"
        ) from exc
    if not isinstance(loaded_manifest, dict):
        raise ValueError(f"manifest is not a JSON object: {path}")
    return loaded_manifest


def build_alias_manifest_payload(source_manifest: dict[str, Any], *, alias: str) -> dict[str, Any]:
    payload = dict(source_manifest)
    payload["collection"] = alias
    return payload


def write_alias_manifest(*, alias_manifest_path: Path, payload: dict[str, Any]) -> Path:
    alias_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    alias_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return alias_manifest_path


def promote_alias(
    *,
    client: QdrantClient,
    source_collection: str,
    alias: str,
) -> None:
    existing_aliases = client.get_aliases().aliases
    actions: list[Any] = []
    if any(item.alias_name == alias for item in existing_aliases):
        actions.append(DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias)))
    actions.append(
        CreateAliasOperation(
            create_alias=CreateAlias(
                collection_name=source_collection,
                alias_name=alias,
            )
        )
    )
    client.update_collection_aliases(actions)


def main() -> int:
    args = parse_args()
    client = QdrantClient(url=args.qdrant_url)
    if not client.collection_exists(args.source_collection):
        print(f"ERROR: collection not found: {args.source_collection}")
        return 1

    source_manifest_path = resolve_source_manifest_path(args.source_collection, args.source_manifest)
    if not source_manifest_path.exists():
        print(f"ERROR: manifest not found: {source_manifest_path}")
        return 1
    try:
        source_manifest = load_manifest_json_object(source_manifest_path)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    alias_manifest_payload = build_alias_manifest_payload(source_manifest, alias=args.alias)
    alias_manifest_path = write_alias_manifest(
        alias_manifest_path=resolve_alias_manifest_path(args.alias, args.alias_manifest_out),
        payload=alias_manifest_payload,
    )

    promote_alias(
        client=client,
        source_collection=args.source_collection,
        alias=args.alias,
    )

    registry = load_registry(args.registry)
    sync_collection_from_manifest(
        registry,
        collection_name=args.alias,
        manifest_path=alias_manifest_path,
    )
    save_registry(args.registry, registry)

    payload = {
        "source_collection": args.source_collection,
        "alias": args.alias,
        "qdrant_url": args.qdrant_url,
        "source_manifest_path": str(source_manifest_path),
        "alias_manifest_path": str(alias_manifest_path),
        "registry_path": str(args.registry),
    }
    if args.json_out.strip():
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Promoted source collection: {args.source_collection}")
    print(f"Alias: {args.alias}")
    print(f"Alias manifest: {alias_manifest_path}")
    print(f"Registry updated: {args.registry}")
    if args.json_out.strip():
        print(f"JSON output: {Path(args.json_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
