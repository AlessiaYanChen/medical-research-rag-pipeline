from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_project_root_on_path()

from qdrant_client import QdrantClient  # noqa: E402

from src.adapters.parsing.marker_parser import MarkerParser  # noqa: E402
from src.app.adapters.embeddings.openai_embedding_adapter import OpenAIEmbeddingAdapter  # noqa: E402
from src.app.adapters.vectorstores.qdrant_repository import QdrantRepository  # noqa: E402
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

    for index, pdf_path in enumerate(pdf_paths):
        print(f"[{index + 1}/{len(pdf_paths)}] Parsing {pdf_path}")
        parsed = parser.parse(pdf_path)
        normalized_tables = normalize_tables(parsed.tables, file_name=pdf_path.name)
        doc_id = pdf_path.stem
        chunks = chunker.chunk_document(
            doc_id=doc_id,
            source_file=pdf_path.name,
            markdown_text=parsed.markdown_text,
            tables=normalized_tables,
            local_file=str(pdf_path),
        )
        if not chunks:
            print(f"ERROR: no chunks generated for {pdf_path}")
            return 1

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
        text_chunks = sum(chunk.metadata.chunk_type == "text" for chunk in chunks)
        table_chunks = sum(chunk.metadata.chunk_type == "table" for chunk in chunks)
        total_chunks += len(chunks)
        manifest_docs.append(
            {
                "doc_id": doc_id,
                "source_file": pdf_path.name,
                "local_file": str(pdf_path),
                "chunk_count": len(chunks),
                "text_chunk_count": text_chunks,
                "table_chunk_count": table_chunks,
            }
        )

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "collection": args.collection,
                "pdf_dir": str(pdf_dir),
                "glob": args.glob,
                "doc_count": len(manifest_docs),
                "chunk_count": total_chunks,
                "ingestion_version": UnifiedChunker.INGESTION_VERSION,
                "chunking_version": UnifiedChunker.CHUNKING_VERSION,
                "docs": manifest_docs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Collection rebuilt: {args.collection}")
    print(f"Documents ingested: {len(manifest_docs)}")
    print(f"Chunks stored: {total_chunks}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
