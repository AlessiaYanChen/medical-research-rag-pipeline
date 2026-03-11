from __future__ import annotations

import argparse
from io import StringIO
import math
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

from src.adapters.parsing.marker_parser import MarkerParser  # noqa: E402
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
        "--max-chars",
        type=int,
        default=900,
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
    return parser.parse_args()


def simple_embedding_fn(texts: list[str], dim: int = 32) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * dim
        if not text:
            vectors.append(vector)
            continue

        encoded = text.encode("utf-8", errors="ignore")
        for idx, byte in enumerate(encoded):
            vector[idx % dim] += float(byte)

        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        vectors.append(vector)
    return vectors


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

    doc_id = args.doc_id or pdf_path.stem
    source_file = pdf_path.name

    try:
        parser = MarkerParser()
        parsed_document = parser.parse(pdf_path)
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
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: chunk preparation failed: {exc}")
        return 1

    if not chunks:
        print("ERROR: no chunks were generated.")
        return 1

    try:
        client = QdrantClient(url=args.qdrant_url)
        vector_size = len(simple_embedding_fn([chunks[0].content])[0])
        ensure_collection(
            client=client,
            collection_name=args.collection,
            vector_size=vector_size,
            recreate=args.recreate_collection,
        )

        repository = QdrantRepository(
            qdrant_client=client,
            collection_name=args.collection,
            embedding_fn=simple_embedding_fn,
        )
        repository.upsert_chunks(chunks)

        retrieval_service = RetrievalService(
            repo=repository,
            embedding_fn=simple_embedding_fn,
        )
        retrieved_context = retrieval_service.retrieve(
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
    print(f"Collection: {args.collection}")
    print(f"Chunks generated: {len(chunks)}")
    print(f"Text chunks: {text_chunk_count}")
    print(f"Table chunks: {table_chunk_count}")
    print("\nRetrieved Context\n")
    print(retrieved_context if retrieved_context else "[no retrieval results]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
