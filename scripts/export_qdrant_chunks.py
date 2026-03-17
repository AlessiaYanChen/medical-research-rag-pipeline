from __future__ import annotations

import argparse
import csv
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export stored Qdrant points for validation as JSON or CSV."
    )
    parser.add_argument(
        "--collection",
        default="medical_research_chunks",
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant base URL.",
    )
    parser.add_argument(
        "--doc-id",
        help="Optional doc_id filter.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of points to fetch per scroll request.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of points to export. Use 0 for all points.",
    )
    parser.add_argument(
        "--with-vectors",
        action="store_true",
        help="Include vectors in the export.",
    )
    parser.add_argument(
        "--json-out",
        help="Path to write JSON output.",
    )
    parser.add_argument(
        "--csv-out",
        help="Path to write CSV output.",
    )
    return parser.parse_args()


def build_doc_filter(doc_id: str | None) -> Any:
    if not doc_id:
        return None
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore
    except Exception:  # noqa: BLE001
        return {
            "must": [
                {
                    "key": "doc_id",
                    "match": {"value": doc_id},
                }
            ]
        }
    return Filter(
        must=[
            FieldCondition(
                key="doc_id",
                match=MatchValue(value=doc_id),
            )
        ]
    )


def extract_vector(raw_vector: Any) -> list[float] | dict[str, list[float]] | None:
    if raw_vector is None:
        return None
    if isinstance(raw_vector, dict):
        return {
            str(name): [float(value) for value in values]
            for name, values in raw_vector.items()
        }
    return [float(value) for value in raw_vector]


def point_to_record(point: Any, include_vectors: bool) -> dict[str, Any]:
    payload = dict(getattr(point, "payload", {}) or {})
    record: dict[str, Any] = {
        "point_id": str(getattr(point, "id", "")),
        "chunk_id": str(payload.get("chunk_id", "")),
        "doc_id": str(payload.get("doc_id", "")),
        "chunk_type": str(payload.get("chunk_type", "")),
        "parent_header": str(payload.get("parent_header", "")),
        "page_number": payload.get("page_number"),
        "content": str(payload.get("content", "")),
        "payload": payload,
    }
    if include_vectors:
        record["vector"] = extract_vector(getattr(point, "vector", None))
    return record


def fetch_points(
    client: QdrantClient,
    collection_name: str,
    doc_id: str | None,
    batch_size: int,
    limit: int,
    include_vectors: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset: Any = None
    query_filter = build_doc_filter(doc_id)

    while True:
        page_limit = batch_size
        if limit > 0:
            remaining = limit - len(records)
            if remaining <= 0:
                break
            page_limit = min(page_limit, remaining)

        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=page_limit,
            with_payload=True,
            with_vectors=include_vectors,
            offset=offset,
        )
        if not points:
            break

        records.extend(point_to_record(point, include_vectors) for point in points)
        if next_offset is None:
            break
        offset = next_offset

    return records


def write_json(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def write_csv(records: list[dict[str, Any]], output_path: Path, include_vectors: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "point_id",
        "chunk_id",
        "doc_id",
        "chunk_type",
        "parent_header",
        "page_number",
        "content",
        "payload_json",
    ]
    if include_vectors:
        fieldnames.append("vector_json")

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "point_id": record["point_id"],
                "chunk_id": record["chunk_id"],
                "doc_id": record["doc_id"],
                "chunk_type": record["chunk_type"],
                "parent_header": record["parent_header"],
                "page_number": record["page_number"],
                "content": record["content"],
                "payload_json": json.dumps(record["payload"], ensure_ascii=True),
            }
            if include_vectors:
                row["vector_json"] = json.dumps(record.get("vector"), ensure_ascii=True)
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    if not args.json_out and not args.csv_out:
        print("ERROR: provide at least one output path via --json-out and/or --csv-out.")
        return 1

    client = QdrantClient(url=args.qdrant_url)
    if not client.collection_exists(args.collection):
        print(f"ERROR: collection not found: {args.collection}")
        return 1

    records = fetch_points(
        client=client,
        collection_name=args.collection,
        doc_id=args.doc_id,
        batch_size=max(1, args.batch_size),
        limit=max(0, args.limit),
        include_vectors=args.with_vectors,
    )

    if args.json_out:
        write_json(records, Path(args.json_out))
    if args.csv_out:
        write_csv(records, Path(args.csv_out), include_vectors=args.with_vectors)

    print(f"Collection: {args.collection}")
    print(f"Qdrant URL: {args.qdrant_url}")
    print(f"doc_id filter: {args.doc_id or '[none]'}")
    print(f"Records exported: {len(records)}")
    if args.json_out:
        print(f"JSON output: {Path(args.json_out)}")
    if args.csv_out:
        print(f"CSV output: {Path(args.csv_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
