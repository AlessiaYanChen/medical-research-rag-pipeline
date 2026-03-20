from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_project_root_on_path()

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: E402

from src.adapters.parsing.marker_parser import MarkerParser  # noqa: E402
from src.app.adapters.embeddings.openai_embedding_adapter import OpenAIEmbeddingAdapter  # noqa: E402
from src.app.adapters.vectorstores.qdrant_repository import QdrantRepository  # noqa: E402
from src.app.tables.table_chunker import UnifiedChunker  # noqa: E402
from scripts.test_e2e_flow import normalize_tables  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reparse and replace one document in an existing Qdrant collection."
    )
    parser.add_argument("--doc-id", required=True, help="Document ID stored in Qdrant.")
    parser.add_argument("--pdf", required=True, help="Path to the source PDF.")
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
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        return 1
    if not args.embedding_api_key:
        print("ERROR: embedding API key is required. Provide --embedding-api-key or set EMBEDDING_API_KEY.")
        return 1

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
        print(f"ERROR: embedding preflight failed: {exc}")
        return 1

    parser = MarkerParser()
    chunker = UnifiedChunker(max_chars=args.max_chars, overlap_paragraphs=args.overlap_paragraphs)

    print(f"Parsing: {pdf_path}")
    parsed = parser.parse(pdf_path)
    normalized_tables = normalize_tables(parsed.tables, file_name=pdf_path.name)
    chunks = chunker.chunk_document(
        doc_id=args.doc_id,
        source_file=pdf_path.name,
        markdown_text=parsed.markdown_text,
        tables=normalized_tables,
    )

    client = QdrantClient(url=args.qdrant_url)
    repository = QdrantRepository(
        qdrant_client=client,
        collection_name=args.collection,
        embedding_fn=embedding_fn,
    )

    client.delete(
        collection_name=args.collection,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=args.doc_id))]
        ),
        wait=True,
    )
    repository.upsert_chunks(chunks)

    text_chunks = sum(chunk.metadata.chunk_type == "text" for chunk in chunks)
    table_chunks = sum(chunk.metadata.chunk_type == "table" for chunk in chunks)
    print(f"Reingested doc_id: {args.doc_id}")
    print(f"Collection: {args.collection}")
    print(f"Chunks: {len(chunks)}")
    print(f"Text chunks: {text_chunks}")
    print(f"Table chunks: {table_chunks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
