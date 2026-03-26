from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from dotenv import load_dotenv


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_project_root_on_path()
load_dotenv()

from qdrant_client import QdrantClient  # noqa: E402

from src.app.adapters.embeddings.openai_embedding_adapter import OpenAIEmbeddingAdapter  # noqa: E402
from src.app.adapters.vectorstores.qdrant_repository import QdrantRepository  # noqa: E402
from src.app.evaluation.retrieval_eval import load_evaluation_queries  # noqa: E402
from src.app.services.retrieval_service import RetrievalService  # noqa: E402


def _console_safe_text(text: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except UnicodeEncodeError:
        return text.encode(encoding, errors="replace").decode(encoding)
    return text


def _emit(text: str) -> None:
    print(_console_safe_text(text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect retrieval candidates for a single query across retrieval stages."
    )
    parser.add_argument(
        "--query",
        help="Raw query text to inspect.",
    )
    parser.add_argument(
        "--query-id",
        help="Query id to load from --dataset.",
    )
    parser.add_argument(
        "--dataset",
        default="data/eval/ood_adversarial_queries.json",
        help="Dataset path used when --query-id is provided.",
    )
    parser.add_argument(
        "--doc-id",
        default="",
        help="Optional doc filter override when using --query.",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "medical_research_chunks_docling_v1"),
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant base URL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Final retrieval limit.",
    )
    parser.add_argument(
        "--stage-limit",
        type=int,
        default=12,
        help="How many chunks to print for each intermediate stage.",
    )
    parser.add_argument(
        "--include-tables",
        action="store_true",
        help="Allow table chunks during retrieval.",
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


def resolve_query(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.query:
        return args.query.strip(), (args.doc_id.strip() or None)

    if not args.query_id:
        raise ValueError("Provide --query or --query-id.")

    queries = load_evaluation_queries(args.dataset)
    for item in queries:
        if item.id == args.query_id:
            return item.query, item.doc_id
    raise ValueError(f"Query id not found in dataset: {args.query_id}")


def summarize_chunk(index: int, chunk, query: str, service: RetrievalService) -> str:
    content_role = str(chunk.metadata.extra.get("content_role", chunk.metadata.chunk_type))
    title_overlap = service._doc_title_overlap(query=query, chunk=chunk)
    preview = service._clean_markdown(str(chunk.metadata.extra.get("parent_content", chunk.content))).replace("\n", " ")
    preview = preview[:140]
    return (
        f"{index:02d}. doc={chunk.metadata.doc_id} | header={service._header_for_display(chunk)}"
        f" | role={content_role} | title_overlap={title_overlap} | preview={preview}"
    )


def print_stage(name: str, chunks: list, query: str, service: RetrievalService, stage_limit: int) -> None:
    _emit(f"\n{name} ({len(chunks)} chunks)")
    for index, chunk in enumerate(chunks[:stage_limit], start=1):
        _emit(summarize_chunk(index=index, chunk=chunk, query=query, service=service))


def main() -> int:
    args = parse_args()
    if not args.embedding_api_key:
        _emit("ERROR: embedding API key is required. Provide --embedding-api-key or set EMBEDDING_API_KEY.")
        return 1

    query, doc_id = resolve_query(args)
    embedding_fn = build_embedding_fn(args)
    client = QdrantClient(url=args.qdrant_url)
    repository = QdrantRepository(
        qdrant_client=client,
        collection_name=args.collection,
        embedding_fn=embedding_fn,
    )
    service = RetrievalService(
        repo=repository,
        embedding_fn=embedding_fn,
        include_tables=args.include_tables,
    )

    query_vector = embedding_fn([query])[0]
    initial_limit = service._initial_search_limit(query=query, doc_filter=doc_id, limit=args.limit)
    filters = service._build_search_filters(query=query, doc_id=doc_id)
    initial_chunks = repository.search(query_vector, doc_id=doc_id, limit=initial_limit, filters=filters)
    filtered_chunks = service._filter_chunks(query=query, chunks=initial_chunks)
    if doc_id is not None:
        filtered_chunks = service._suppress_metadata_fallback(query=query, chunks=filtered_chunks)
    candidate_limit = max(args.limit * 6, 30) if service._query_prefers_tables(query) else max(args.limit * 4, 20)
    candidate_chunks = service._select_candidate_chunks(
        query=query,
        initial_chunks=filtered_chunks,
        candidate_limit=candidate_limit,
    )
    ranked_chunks = service._rank_chunks(query=query, chunks=candidate_chunks)
    final_chunks = service.retrieve(query=query, doc_id=doc_id, limit=args.limit)

    _emit(f"Query: {query}")
    _emit(f"Doc filter: {doc_id or '<none>'}")
    _emit(f"Collection: {args.collection}")
    _emit(f"Initial limit: {initial_limit}")
    _emit(f"Candidate limit: {candidate_limit}")
    _emit(f"Filters: {filters}")

    print_stage("Initial Search", initial_chunks, query, service, args.stage_limit)
    print_stage("Post Filter", filtered_chunks, query, service, args.stage_limit)
    print_stage("Ranked Candidates", ranked_chunks, query, service, args.stage_limit)

    _emit(f"\nFinal Returned Chunks ({len(final_chunks)} chunks)")
    for index, chunk in enumerate(final_chunks, start=1):
        preview = chunk.content.replace("\n", " ")[:160]
        _emit(
            f"{index:02d}. doc={chunk.doc_id} | header={chunk.source} | role={chunk.content_role}"
            f" | chunk_type={chunk.chunk_type} | preview={preview}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
