from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_project_root_on_path()

from qdrant_client import QdrantClient  # noqa: E402

from src.adapters.parsing.marker_parser import MarkerParser  # noqa: E402
from src.app.adapters.embeddings.openai_embedding_adapter import OpenAIEmbeddingAdapter  # noqa: E402
from src.app.ingestion.dedup_utils import build_doc_identity, validate_unique_doc_identities  # noqa: E402
from src.app.ingestion.doc_id_utils import doc_id_from_path  # noqa: E402
from src.app.adapters.vectorstores.qdrant_repository import QdrantRepository  # noqa: E402
from src.app.ingestion.manifest_utils import build_manifest_doc_entry, write_rebuild_manifest  # noqa: E402
from src.app.tables.table_chunker import UnifiedChunker  # noqa: E402
from scripts.test_e2e_flow import ensure_collection, normalize_tables  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministically rebuild a Qdrant collection from a sorted PDF set."
    )
    parser.add_argument(
        "--pdf-dir",
        required=True,
        help="Directory containing PDFs to ingest.",
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
        "--glob",
        default="*.pdf",
        help="Glob pattern under --pdf-dir.",
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
        "--manifest-out",
        default="data/ingestion_manifests/rebuild_manifest.json",
        help="Path to write rebuild manifest JSON.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing later PDFs after a per-document failure and report all failures at the end.",
    )
    parser.add_argument(
        "--failure-report-out",
        default="",
        help="Optional path to write a JSON report describing per-document rebuild failures.",
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
    stage: str,
    error: Exception,
    doc_id: str | None = None,
) -> dict[str, str]:
    error_message = str(error).strip() or f"{error.__class__.__name__}"
    return {
        "pdf_path": str(pdf_path),
        "doc_id": doc_id or "",
        "stage": stage,
        "error": error_message,
    }


def write_failure_report(
    *,
    output_path: str | Path,
    collection: str,
    pdf_dir: str | Path,
    failures: list[dict[str, str]],
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "collection": collection,
        "pdf_dir": str(pdf_dir),
        "failure_count": len(failures),
        "failures": failures,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"ERROR: PDF directory not found: {pdf_dir}")
        return 1
    if not args.embedding_api_key:
        print("ERROR: embedding API key is required. Provide --embedding-api-key or set EMBEDDING_API_KEY.")
        return 1

    pdf_paths = sorted(path for path in pdf_dir.rglob(args.glob) if path.is_file())
    if not pdf_paths:
        print(f"ERROR: no PDFs matched {args.glob!r} under {pdf_dir}")
        return 1
    try:
        validate_unique_doc_identities(
            [
                build_doc_identity(
                    doc_id=doc_id_from_path(path),
                    source_file=path.name,
                    local_file=str(path),
                )
                for path in pdf_paths
            ],
            context="Rebuild input set",
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    embedding_fn = OpenAIEmbeddingAdapter(
        api_key=args.embedding_api_key,
        model=args.embedding_model,
        provider=args.embedding_provider,
        azure_endpoint=args.embedding_azure_endpoint or None,
        azure_api_version=args.embedding_azure_api_version or None,
        dimensions=None if args.embedding_dimensions == 0 else args.embedding_dimensions,
    )
    parser = MarkerParser()
    chunker = UnifiedChunker(max_chars=args.max_chars, overlap_paragraphs=args.overlap_paragraphs)
    client = QdrantClient(url=args.qdrant_url)
    repository: QdrantRepository | None = None

    manifest_docs: list[dict[str, object]] = []
    total_chunks = 0
    failures: list[dict[str, str]] = []

    for index, pdf_path in enumerate(pdf_paths):
        print(f"[{index + 1}/{len(pdf_paths)}] Parsing {pdf_path}")
        doc_id: str | None = None
        try:
            parsed = parser.parse(pdf_path)
            normalized_tables = normalize_tables(parsed.tables, file_name=pdf_path.name)
            doc_id = doc_id_from_path(pdf_path)
            chunks = chunker.chunk_document(
                doc_id=doc_id,
                source_file=pdf_path.name,
                markdown_text=parsed.markdown_text,
                tables=normalized_tables,
                local_file=str(pdf_path),
            )
            if not chunks:
                raise RuntimeError("no chunks generated")

            if repository is None:
                vector_size = len(embedding_fn([chunks[0].content])[0])
                ensure_collection(
                    client=client,
                    collection_name=args.collection,
                    vector_size=vector_size,
                    recreate=True,
                )
                repository = QdrantRepository(
                    qdrant_client=client,
                    collection_name=args.collection,
                    embedding_fn=embedding_fn,
                )

            repository.upsert_chunks(chunks)
            total_chunks += len(chunks)
            manifest_docs.append(
                build_manifest_doc_entry(
                    doc_id=doc_id,
                    source_file=pdf_path.name,
                    local_file=str(pdf_path),
                    chunks=chunks,
                    ingestion_version=UnifiedChunker.INGESTION_VERSION,
                    chunking_version=UnifiedChunker.CHUNKING_VERSION,
                )
            )
        except Exception as exc:  # noqa: BLE001
            failure = build_failure_record(
                pdf_path=pdf_path,
                doc_id=doc_id,
                stage="rebuild_document",
                error=exc,
            )
            failures.append(failure)
            print(
                f"ERROR: failed to rebuild {pdf_path}"
                f"{f' (doc_id={doc_id})' if doc_id else ''}: {failure['error']}"
            )
            if not args.continue_on_error:
                return 1

    if not manifest_docs:
        print("ERROR: rebuild produced no successfully ingested documents.")
        if args.failure_report_out.strip() and failures:
            failure_report_path = write_failure_report(
                output_path=args.failure_report_out,
                collection=args.collection,
                pdf_dir=pdf_dir,
                failures=failures,
            )
            print(f"Failure report: {failure_report_path}")
        return 1

    manifest_path = Path(args.manifest_out)
    write_rebuild_manifest(
        manifest_path=manifest_path,
        collection=args.collection,
        pdf_dir=str(pdf_dir),
        glob_pattern=args.glob,
        docs=manifest_docs,
        ingestion_version=UnifiedChunker.INGESTION_VERSION,
        chunking_version=UnifiedChunker.CHUNKING_VERSION,
    )

    print(f"Collection rebuilt: {args.collection}")
    print(f"Documents ingested: {len(manifest_docs)}")
    print(f"Chunks stored: {total_chunks}")
    print(f"Manifest: {manifest_path}")
    if failures and args.failure_report_out.strip():
        failure_report_path = write_failure_report(
            output_path=args.failure_report_out,
            collection=args.collection,
            pdf_dir=pdf_dir,
            failures=failures,
        )
        print(f"Failure report: {failure_report_path}")
    if failures:
        print(f"Failures: {len(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
