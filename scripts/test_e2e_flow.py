from __future__ import annotations

import argparse
from io import StringIO
import os
from pathlib import Path
import sys

import pandas as pd


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_project_root_on_path()

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import Distance, VectorParams  # noqa: E402

from src.app.adapters.embeddings.openai_embedding_adapter import OpenAIEmbeddingAdapter  # noqa: E402
from src.app.ingestion.doc_id_utils import doc_id_from_path, normalize_doc_id  # noqa: E402
from src.app.ingestion.parser_factory import DEFAULT_PARSER_NAME, PARSER_CHOICES, build_parser  # noqa: E402
from src.app.adapters.vectorstores.qdrant_repository import QdrantRepository  # noqa: E402
from src.app.services.retrieval_service import RetrievalService  # noqa: E402
from src.app.tables.table_chunker import UnifiedChunker  # noqa: E402
from src.app.tables.table_normalizer import TableNormalizer  # noqa: E402
from src.ports.parser_port import ParsedTable  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end PDF -> chunk -> Qdrant -> retrieval integration flow."
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF.")
    parser.add_argument("--query", required=True, help="Retrieval query to execute after ingestion.")
    parser.add_argument("--doc-id", help="Document ID. Defaults to the PDF stem.")
    parser.add_argument(
        "--parser",
        choices=PARSER_CHOICES,
        default=DEFAULT_PARSER_NAME,
        help="Parser used during ingestion.",
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
        "--limit",
        type=int,
        default=5,
        help="Number of chunks to retrieve.",
    )
    parser.add_argument(
        "--include-tables",
        action="store_true",
        help="Include table chunks during retrieval.",
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
        "--recreate-collection",
        action="store_true",
        help="Delete and recreate the target collection before upsert.",
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


def build_embedding_fn(args: argparse.Namespace) -> OpenAIEmbeddingAdapter:
    return OpenAIEmbeddingAdapter(
        api_key=args.embedding_api_key,
        model=args.embedding_model,
        provider=args.embedding_provider,
        azure_endpoint=args.embedding_azure_endpoint or None,
        azure_api_version=args.embedding_azure_api_version or None,
        dimensions=None if args.embedding_dimensions == 0 else args.embedding_dimensions,
    )


def normalize_tables(parsed_tables: list[ParsedTable], file_name: str) -> list[dict[str, object]]:
    normalizer = TableNormalizer()
    normalized: list[dict[str, object]] = []

    for table in parsed_tables:
        df = parsed_table_to_dataframe(table)
        cleaned_df = normalizer.sanitize_table(df=df, file_name=file_name)

        csv_text = cleaned_df.to_csv(index=False, header=False).strip()
        rows = dataframe_to_rows(cleaned_df)
        artifact = {
            "csv": csv_text,
            "rows": rows,
        }

        metadata_artifact = normalizer.get_last_metadata_artifact()
        if metadata_artifact:
            artifact["normalization_metadata"] = metadata_artifact

        normalized.append(artifact)

    return normalized


def parsed_table_to_dataframe(table: ParsedTable) -> pd.DataFrame:
    if table.headers or table.rows:
        row_lists: list[list[str]] = []
        if table.headers:
            row_lists.append([str(value) for value in table.headers])
        for row in table.rows:
            row_lists.append([str(row.get(header, "")) for header in table.headers])
        return pd.DataFrame(row_lists)

    return pd.read_csv(StringIO(table.csv), header=None, engine="python", on_bad_lines="skip")


def dataframe_to_rows(df: pd.DataFrame) -> list[dict[str, str]]:
    if df.empty or len(df) < 2:
        return []

    headers = [str(value) for value in df.iloc[0].tolist()]
    rows: list[dict[str, str]] = []
    for row_idx in range(1, len(df)):
        values = ["" if pd.isna(value) else str(value) for value in df.iloc[row_idx].tolist()]
        if len(values) < len(headers):
            values.extend([""] * (len(headers) - len(values)))
        rows.append(dict(zip(headers, values)))
    return rows


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    recreate: bool,
) -> None:
    exists = client.collection_exists(collection_name)
    if exists and recreate:
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        return 1

    try:
        doc_id = normalize_doc_id(args.doc_id) if args.doc_id else doc_id_from_path(pdf_path)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    source_file = pdf_path.name
    if not args.embedding_api_key:
        print("ERROR: embedding API key is required. Provide --embedding-api-key or set EMBEDDING_API_KEY.")
        return 1

    try:
        embedding_fn = build_embedding_fn(args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: embedding setup failed: {exc}")
        return 1

    try:
        document_parser = build_parser(args.parser)
        parsed_document = document_parser.parse(pdf_path)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: parsing failed: {exc}")
        return 1

    try:
        normalized_tables = normalize_tables(parsed_document.tables, file_name=source_file)
        chunker = UnifiedChunker(
            max_chars=args.max_chars,
            overlap_paragraphs=args.overlap_paragraphs,
        )
        chunks = chunker.chunk_document(
            doc_id=doc_id,
            source_file=source_file,
            markdown_text=parsed_document.markdown_text,
            tables=normalized_tables,
            local_file=str(pdf_path),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: chunk preparation failed: {exc}")
        return 1

    if not chunks:
        print("ERROR: no chunks were generated.")
        return 1

    try:
        client = QdrantClient(url=args.qdrant_url)
        vector_size = len(embedding_fn([chunks[0].content])[0])
        ensure_collection(
            client=client,
            collection_name=args.collection,
            vector_size=vector_size,
            recreate=args.recreate_collection,
        )

        repository = QdrantRepository(
            qdrant_client=client,
            collection_name=args.collection,
            embedding_fn=embedding_fn,
        )
        repository.upsert_chunks(chunks)

        retrieval_service = RetrievalService(
            repo=repository,
            embedding_fn=embedding_fn,
            include_tables=args.include_tables,
        )
        retrieved_chunks = retrieval_service.retrieve(
            query=args.query,
            doc_id=doc_id,
            limit=args.limit,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: persistence or retrieval failed: {exc}")
        return 1

    text_chunk_count = sum(chunk.metadata.chunk_type == "text" for chunk in chunks)
    table_chunk_count = sum(chunk.metadata.chunk_type == "table" for chunk in chunks)

    print(f"PDF: {pdf_path}")
    print(f"Document ID: {doc_id}")
    print(f"Parser: {args.parser}")
    print(f"Collection: {args.collection}")
    print(f"Chunks generated: {len(chunks)}")
    print(f"Text chunks: {text_chunk_count}")
    print(f"Table chunks: {table_chunk_count}")
    print("\nRetrieved Context\n")
    print(
        retrieval_service.serialize_for_prompt(retrieved_chunks)
        if retrieved_chunks
        else "[no retrieval results]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
