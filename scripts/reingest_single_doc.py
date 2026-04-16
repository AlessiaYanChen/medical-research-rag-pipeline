from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_project_root_on_path()
load_dotenv()

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: E402

from src.app.adapters.embeddings.openai_embedding_adapter import OpenAIEmbeddingAdapter  # noqa: E402
from src.app.ingestion.dedup_utils import (  # noqa: E402
    ensure_doc_identity_is_available,
    fetch_collection_doc_identities,
    resolve_canonical_doc_id,
)
from src.app.ingestion.doc_id_utils import normalize_doc_id  # noqa: E402
from src.app.ingestion.file_identity_utils import compute_file_identity  # noqa: E402
from src.app.adapters.vectorstores.qdrant_repository import QdrantRepository  # noqa: E402
from src.app.ingestion.manifest_utils import build_manifest_doc_entry, upsert_manifest_doc_entry  # noqa: E402
from src.app.ingestion.parser_factory import DEFAULT_PARSER_NAME, PARSER_CHOICES, build_parser  # noqa: E402
from src.app.ingestion.runtime_utils import normalize_tables  # noqa: E402
from src.app.tables.table_chunker import UnifiedChunker  # noqa: E402
from src.app.ingestion.versioning_utils import validate_manifest_compatibility  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reparse and replace one document in an existing Qdrant collection."
    )
    parser.add_argument("--doc-id", required=True, help="Document ID stored in Qdrant.")
    parser.add_argument("--pdf", required=True, help="Path to the source PDF.")
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "medical_research_chunks_docling_v1"),
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--parser",
        choices=PARSER_CHOICES,
        default=DEFAULT_PARSER_NAME,
        help="Parser used during reingestion.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant base URL.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1800,
        help="Maximum characters per text chunk.",
    )
    parser.add_argument(
        "--overlap-paragraphs",
        type=int,
        default=1,
        help="Paragraph overlap for text chunking.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional rebuild manifest JSON to update after successful reingestion.",
    )
    parser.add_argument(
        "--failure-report-out",
        default="",
        help="Optional path to write a JSON report describing the failed reingestion attempt.",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["openai", "azure_openai"],
        default=os.getenv("EMBEDDING_PROVIDER", "openai"),
        help="Embedding provider.",
    )
    parser.add_argument(
        "--embedding-api-key",
        default=os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        help="Embedding API key.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        help="Embedding model or Azure deployment name.",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
        help="Embedding dimensions. Use 0 for provider default.",
    )
    parser.add_argument(
        "--embedding-azure-endpoint",
        default=os.getenv("EMBEDDING_AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT", "")),
        help="Azure OpenAI endpoint for embeddings.",
    )
    parser.add_argument(
        "--embedding-azure-api-version",
        default=os.getenv("EMBEDDING_AZURE_OPENAI_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")),
        help="Azure OpenAI API version for embeddings.",
    )
    return parser.parse_args()


def build_failure_record(
    *,
    pdf_path: Path,
    doc_id: str,
    collection: str,
    stage: str,
    error: Exception | str,
) -> dict[str, str]:
    error_message = str(error).strip() or (
        error.__class__.__name__ if isinstance(error, Exception) else "Unknown error"
    )
    return {
        "pdf_path": str(pdf_path),
        "doc_id": doc_id,
        "collection": collection,
        "stage": stage,
        "error": error_message,
    }


def resolve_failure_report_path(
    *,
    output_path: str,
    collection: str,
    doc_id: str,
) -> Path:
    if output_path.strip():
        return Path(output_path)
    safe_doc_id = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in doc_id).strip("_")
    safe_doc_id = safe_doc_id or "unknown_doc"
    return Path("data/eval/results") / f"reingest_failure_{collection}_{safe_doc_id}.json"


def load_manifest_json_object(manifest_path: Path) -> dict[str, Any]:
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


def write_failure_report(
    *,
    output_path: str | Path,
    failure: dict[str, str],
    manifest_path: str = "",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "doc_id": failure["doc_id"],
        "collection": failure["collection"],
        "pdf_path": failure["pdf_path"],
        "manifest_path": manifest_path,
        "failure": failure,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def backup_existing_doc_points(
    *,
    client: QdrantClient,
    collection_name: str,
    doc_id: str,
    batch_size: int = 256,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset: Any = None
    query_filter = Filter(
        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    )

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=batch_size,
            with_payload=True,
            with_vectors=True,
            offset=offset,
        )
        if not points:
            break

        for point in points:
            records.append(
                {
                    "point_id": str(getattr(point, "id", "")),
                    "payload": dict(getattr(point, "payload", {}) or {}),
                    "vector": getattr(point, "vector", None),
                }
            )

        if next_offset is None:
            break
        offset = next_offset

    return records


def restore_doc_points(
    *,
    client: QdrantClient,
    collection_name: str,
    records: list[dict[str, Any]],
) -> None:
    if not records:
        return

    try:
        from qdrant_client.models import PointStruct  # type: ignore
    except Exception:  # noqa: BLE001
        points = [
            {
                "id": record["point_id"],
                "payload": dict(record["payload"]),
                "vector": record["vector"],
            }
            for record in records
        ]
    else:
        points = [
            PointStruct(
                id=record["point_id"],
                payload=dict(record["payload"]),
                vector=record["vector"],
            )
            for record in records
        ]

    client.upsert(collection_name=collection_name, points=points, wait=True)


def report_failure(
    *,
    pdf_path: Path,
    doc_id: str,
    collection: str,
    stage: str,
    error: Exception | str,
    failure_report_out: str,
    manifest_path: str = "",
) -> int:
    failure = build_failure_record(
        pdf_path=pdf_path,
        doc_id=doc_id,
        collection=collection,
        stage=stage,
        error=error,
    )
    print(f"ERROR [{stage}]: {failure['error']}")
    written_path = write_failure_report(
        output_path=resolve_failure_report_path(
            output_path=failure_report_out,
            collection=collection,
            doc_id=doc_id,
        ),
        failure=failure,
        manifest_path=manifest_path,
    )
    print(f"Failure report: {written_path}")
    return 1


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf)
    try:
        normalized_doc_id = normalize_doc_id(args.doc_id)
    except ValueError as exc:
        return report_failure(
            pdf_path=pdf_path,
            doc_id=args.doc_id,
            collection=args.collection,
            stage="normalize_doc_id",
            error=exc,
            failure_report_out=args.failure_report_out,
            manifest_path=args.manifest,
        )
    if not pdf_path.exists():
        return report_failure(
            pdf_path=pdf_path,
            doc_id=normalized_doc_id,
            collection=args.collection,
            stage="validate_pdf_path",
            error=f"PDF not found: {pdf_path}",
            failure_report_out=args.failure_report_out,
            manifest_path=args.manifest,
        )
    if not args.embedding_api_key:
        return report_failure(
            pdf_path=pdf_path,
            doc_id=normalized_doc_id,
            collection=args.collection,
            stage="validate_embedding_config",
            error="embedding API key is required. Provide --embedding-api-key or set EMBEDDING_API_KEY.",
            failure_report_out=args.failure_report_out,
            manifest_path=args.manifest,
        )
    file_identity = compute_file_identity(pdf_path)
    normalized_source_sha256 = str(file_identity["source_sha256"])
    canonical_doc_id = normalized_doc_id

    manifest_entries: list[dict[str, Any]] = []
    if args.manifest.strip():
        manifest_path = Path(args.manifest)
        if manifest_path.exists():
            try:
                loaded_manifest = load_manifest_json_object(manifest_path)
            except ValueError as exc:
                return report_failure(
                    pdf_path=pdf_path,
                    doc_id=normalized_doc_id,
                    collection=args.collection,
                    stage="manifest_validation",
                    error=exc,
                    failure_report_out=args.failure_report_out,
                    manifest_path=args.manifest,
                )
            compatibility_issues = validate_manifest_compatibility(
                loaded_manifest,
                expected_collection=args.collection,
                expected_ingestion_version=UnifiedChunker.INGESTION_VERSION,
                expected_chunking_version=UnifiedChunker.CHUNKING_VERSION,
            )
            if compatibility_issues:
                return report_failure(
                    pdf_path=pdf_path,
                    doc_id=normalized_doc_id,
                    collection=args.collection,
                    stage="manifest_validation",
                    error="manifest compatibility check failed: " + " | ".join(compatibility_issues),
                    failure_report_out=args.failure_report_out,
                    manifest_path=args.manifest,
                )
            manifest_entries = list(loaded_manifest.get("docs", [])) if isinstance(loaded_manifest.get("docs", []), list) else []
            canonical_doc_id = resolve_canonical_doc_id(
                requested_doc_id=canonical_doc_id,
                source_sha256=normalized_source_sha256,
                existing_entries=manifest_entries,
            )
            try:
                ensure_doc_identity_is_available(
                    doc_id=canonical_doc_id,
                    source_file=pdf_path.name,
                    local_file=str(pdf_path),
                    source_sha256=normalized_source_sha256,
                    existing_entries=manifest_entries,
                    context=f"Manifest '{manifest_path}'",
                    allowed_doc_ids={canonical_doc_id},
                )
            except ValueError as exc:
                return report_failure(
                    pdf_path=pdf_path,
                    doc_id=canonical_doc_id,
                    collection=args.collection,
                    stage="manifest_validation",
                    error=exc,
                    failure_report_out=args.failure_report_out,
                    manifest_path=args.manifest,
                )

    embedding_fn = OpenAIEmbeddingAdapter(
        api_key=args.embedding_api_key,
        model=args.embedding_model,
        provider=args.embedding_provider,
        azure_endpoint=args.embedding_azure_endpoint or None,
        azure_api_version=args.embedding_azure_api_version or None,
        dimensions=None if args.embedding_dimensions == 0 else args.embedding_dimensions,
    )
    try:
        embedding_fn(["embedding auth preflight"])
    except Exception as exc:  # noqa: BLE001
        return report_failure(
            pdf_path=pdf_path,
            doc_id=normalized_doc_id,
            collection=args.collection,
            stage="embedding_preflight",
            error=exc,
            failure_report_out=args.failure_report_out,
            manifest_path=args.manifest,
        )

    client = QdrantClient(url=args.qdrant_url)
    collection_entries = fetch_collection_doc_identities(client, collection_name=args.collection)
    canonical_doc_id = resolve_canonical_doc_id(
        requested_doc_id=canonical_doc_id,
        source_sha256=normalized_source_sha256,
        existing_entries=collection_entries,
    )

    document_parser = build_parser(args.parser)
    chunker = UnifiedChunker(max_chars=args.max_chars, overlap_paragraphs=args.overlap_paragraphs)

    print(f"Parsing: {pdf_path}")
    try:
        parsed = document_parser.parse(pdf_path)
    except Exception as exc:  # noqa: BLE001
        return report_failure(
            pdf_path=pdf_path,
            doc_id=normalized_doc_id,
            collection=args.collection,
            stage="parse",
            error=exc,
            failure_report_out=args.failure_report_out,
            manifest_path=args.manifest,
        )
    try:
        normalized_tables = normalize_tables(parsed.tables, file_name=pdf_path.name)
        chunks = chunker.chunk_document(
            doc_id=canonical_doc_id,
            source_file=pdf_path.name,
            markdown_text=parsed.markdown_text,
            tables=normalized_tables,
            local_file=str(pdf_path),
            source_sha256=normalized_source_sha256,
            file_size_bytes=int(file_identity["file_size_bytes"]),
        )
    except Exception as exc:  # noqa: BLE001
        return report_failure(
            pdf_path=pdf_path,
            doc_id=canonical_doc_id,
            collection=args.collection,
            stage="chunk",
            error=exc,
            failure_report_out=args.failure_report_out,
            manifest_path=args.manifest,
        )

    repository = QdrantRepository(
        qdrant_client=client,
        collection_name=args.collection,
        embedding_fn=embedding_fn,
    )
    try:
        ensure_doc_identity_is_available(
            doc_id=canonical_doc_id,
            source_file=pdf_path.name,
            local_file=str(pdf_path),
            source_sha256=normalized_source_sha256,
            existing_entries=collection_entries,
            context=f"Qdrant collection '{args.collection}'",
            allowed_doc_ids={canonical_doc_id},
        )
    except ValueError as exc:
        return report_failure(
            pdf_path=pdf_path,
            doc_id=canonical_doc_id,
            collection=args.collection,
            stage="validate_collection_identity",
            error=exc,
            failure_report_out=args.failure_report_out,
            manifest_path=args.manifest,
        )

    backup_records = backup_existing_doc_points(
        client=client,
        collection_name=args.collection,
        doc_id=canonical_doc_id,
    )

    try:
        client.delete(
            collection_name=args.collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=canonical_doc_id))]
            ),
            wait=True,
        )
    except Exception as exc:  # noqa: BLE001
        return report_failure(
            pdf_path=pdf_path,
            doc_id=canonical_doc_id,
            collection=args.collection,
            stage="delete_old_doc",
            error=exc,
            failure_report_out=args.failure_report_out,
            manifest_path=args.manifest,
        )
    try:
        repository.upsert_chunks(chunks)
    except Exception as exc:  # noqa: BLE001
        rollback_error: Exception | None = None
        try:
            restore_doc_points(
                client=client,
                collection_name=args.collection,
                records=backup_records,
            )
        except Exception as restore_exc:  # noqa: BLE001
            rollback_error = restore_exc

        if rollback_error is not None:
            return report_failure(
                pdf_path=pdf_path,
                doc_id=canonical_doc_id,
                collection=args.collection,
                stage="rollback_old_doc",
                error=(
                    "replacement write failed and rollback restore also failed: "
                    f"upsert_new_doc={exc}; rollback_old_doc={rollback_error}"
                ),
                failure_report_out=args.failure_report_out,
                manifest_path=args.manifest,
            )
        return report_failure(
            pdf_path=pdf_path,
            doc_id=canonical_doc_id,
            collection=args.collection,
            stage="upsert_new_doc",
            error=exc,
            failure_report_out=args.failure_report_out,
            manifest_path=args.manifest,
        )

    text_chunks = sum(chunk.metadata.chunk_type == "text" for chunk in chunks)
    table_chunks = sum(chunk.metadata.chunk_type == "table" for chunk in chunks)
    if args.manifest.strip():
        try:
            manifest_entry = build_manifest_doc_entry(
                doc_id=normalized_doc_id,
                source_file=pdf_path.name,
                local_file=str(pdf_path),
                chunks=chunks,
                ingestion_version=UnifiedChunker.INGESTION_VERSION,
                chunking_version=UnifiedChunker.CHUNKING_VERSION,
                parser_name=args.parser,
                source_sha256=str(file_identity["source_sha256"]),
                file_size_bytes=int(file_identity["file_size_bytes"]),
            )
            upsert_manifest_doc_entry(
                manifest_path=args.manifest,
                collection=args.collection,
                doc_entry=manifest_entry,
                ingestion_version=UnifiedChunker.INGESTION_VERSION,
                chunking_version=UnifiedChunker.CHUNKING_VERSION,
                parser_name=args.parser,
            )
        except Exception as exc:  # noqa: BLE001
            return report_failure(
                pdf_path=pdf_path,
                doc_id=normalized_doc_id,
                collection=args.collection,
                stage="manifest_update",
                error=exc,
                failure_report_out=args.failure_report_out,
                manifest_path=args.manifest,
            )

    if canonical_doc_id != normalized_doc_id:
        print(f"Requested doc_id: {normalized_doc_id}")
        print(f"Canonical doc_id: {canonical_doc_id}")
    print(f"Reingested doc_id: {canonical_doc_id}")
    print(f"Parser: {args.parser}")
    print(f"Collection: {args.collection}")
    print(f"Chunks: {len(chunks)}")
    print(f"Text chunks: {text_chunks}")
    print(f"Table chunks: {table_chunks}")
    if args.manifest.strip():
        print(f"Manifest updated: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
