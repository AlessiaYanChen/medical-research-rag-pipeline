from __future__ import annotations

import argparse
import csv
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

from src.app.adapters.embeddings.openai_embedding_adapter import OpenAIEmbeddingAdapter  # noqa: E402
from src.app.adapters.rerankers.transformers_reranker import TransformersReRanker  # noqa: E402
from src.app.adapters.vectorstores.qdrant_repository import QdrantRepository  # noqa: E402
from src.app.evaluation.retrieval_eval import (  # noqa: E402
    build_summary,
    evaluate_retrieval_results,
    load_evaluation_queries,
)
from src.app.services.retrieval_service import RetrievalService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small retrieval evaluation set against an indexed Qdrant collection."
    )
    parser.add_argument(
        "--dataset",
        default="data/eval/sample_queries.json",
        help="Path to evaluation dataset JSON.",
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
        help="Number of retrieved chunks per query.",
    )
    parser.add_argument(
        "--include-tables",
        action="store_true",
        help="Allow table chunks during retrieval.",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Enable local re-ranking during evaluation.",
    )
    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model to use when --use-reranker is set.",
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
    parser.add_argument(
        "--json-out",
        default="",
        help="Path to write detailed JSON results. Defaults to a dataset-specific file under data/eval/results/.",
    )
    parser.add_argument(
        "--csv-out",
        default="",
        help="Path to write flattened CSV results. Defaults to a dataset-specific file under data/eval/results/.",
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


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query",
        "doc_filter",
        "include_tables",
        "labels",
        "expected_doc_hit",
        "expected_header_hit",
        "top1_expected_doc_hit",
        "top1_expected_header_hit",
        "doc_precision",
        "header_precision",
        "result_count",
        "table_hits",
        "citation_noise_hits",
        "duplicate_hits",
        "non_structural_header_hits",
        "distinct_doc_count",
        "distinct_header_count",
        "expected_docs",
        "expected_headers",
        "result_docs",
        "result_headers",
        "non_structural_headers",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "doc_filter": row["doc_filter"],
                    "include_tables": row["include_tables"],
                    "labels": " | ".join(row["labels"]),
                    "expected_doc_hit": row["expected_doc_hit"],
                    "expected_header_hit": row["expected_header_hit"],
                    "top1_expected_doc_hit": row["top1_expected_doc_hit"],
                    "top1_expected_header_hit": row["top1_expected_header_hit"],
                    "doc_precision": row["doc_precision"],
                    "header_precision": row["header_precision"],
                    "result_count": row["result_count"],
                    "table_hits": row["table_hits"],
                    "citation_noise_hits": row["citation_noise_hits"],
                    "duplicate_hits": row["duplicate_hits"],
                    "non_structural_header_hits": row["non_structural_header_hits"],
                    "distinct_doc_count": row["distinct_doc_count"],
                    "distinct_header_count": row["distinct_header_count"],
                    "expected_docs": " | ".join(row["expected_docs"]),
                    "expected_headers": " | ".join(row["expected_headers"]),
                    "result_docs": " | ".join(row["result_docs"]),
                    "result_headers": " | ".join(row["result_headers"]),
                    "non_structural_headers": " | ".join(row["non_structural_headers"]),
                    "notes": row["notes"],
                }
            )


def resolve_output_paths(
    *,
    dataset_path: Path,
    json_out: str,
    csv_out: str,
) -> tuple[Path, Path]:
    resolved_json = Path(json_out) if str(json_out).strip() else default_json_output_path(dataset_path)
    resolved_csv = Path(csv_out) if str(csv_out).strip() else default_csv_output_path(dataset_path)
    return resolved_json, resolved_csv


def default_json_output_path(dataset_path: Path) -> Path:
    return Path("data/eval/results") / f"{_dataset_output_stem(dataset_path)}.json"


def default_csv_output_path(dataset_path: Path) -> Path:
    return Path("data/eval/results") / f"{_dataset_output_stem(dataset_path)}.csv"


def _dataset_output_stem(dataset_path: Path) -> str:
    dataset_name = dataset_path.name.lower()
    if dataset_name == "sample_queries.json":
        return "retrieval_eval_sample"
    if dataset_name == "expanded_queries.json":
        return "retrieval_eval_expanded"
    if dataset_name == "ood_adversarial_queries.json":
        return "ood_retrieval_eval"
    if dataset_name == "known_gap_queries.json":
        return "known_gap_retrieval_eval"
    return f"retrieval_eval_{dataset_path.stem}"


def main() -> int:
    args = parse_args()
    if not args.embedding_api_key:
        print("ERROR: embedding API key is required. Provide --embedding-api-key or set EMBEDDING_API_KEY.")
        return 1

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: evaluation dataset not found: {dataset_path}")
        return 1

    queries = load_evaluation_queries(dataset_path)
    if not queries:
        print("ERROR: evaluation dataset is empty.")
        return 1
    json_output_path, csv_output_path = resolve_output_paths(
        dataset_path=dataset_path,
        json_out=args.json_out,
        csv_out=args.csv_out,
    )

    client = QdrantClient(url=args.qdrant_url)
    if not client.collection_exists(args.collection):
        print(f"ERROR: collection not found: {args.collection}")
        return 1

    embedding_fn = build_embedding_fn(args)
    repository = QdrantRepository(
        qdrant_client=client,
        collection_name=args.collection,
        embedding_fn=embedding_fn,
    )
    re_ranker = TransformersReRanker(model_name=args.reranker_model) if args.use_reranker else None
    query_evaluations: list[dict[str, Any]] = []
    detailed_results: list[dict[str, Any]] = []
    for query in queries:
        retrieval_service = RetrievalService(
            repo=repository,
            embedding_fn=embedding_fn,
            re_ranker=re_ranker,
            include_tables=args.include_tables if query.include_tables is None else query.include_tables,
        )
        retrieved_chunks = retrieval_service.retrieve(
            query=query.query,
            doc_id=query.doc_id,
            limit=args.limit,
        )
        evaluation = evaluate_retrieval_results(query, retrieved_chunks)
        query_evaluations.append(evaluation)
        detailed_results.append(
            {
                "query_id": query.id,
                "query": query.query,
                "doc_filter": query.doc_id,
                "expected_docs": list(query.expected_docs),
                "expected_headers": list(query.expected_headers),
                "labels": list(query.labels),
                "include_tables": query.include_tables,
                "notes": query.notes,
                "evaluation": evaluation,
                "results": [
                    {
                        "rank": index,
                        "doc_id": chunk.doc_id,
                        "source": chunk.source,
                        "chunk_type": chunk.chunk_type,
                        "content_role": chunk.content_role,
                        "page_number": chunk.page_number,
                        "content": chunk.content,
                    }
                    for index, chunk in enumerate(retrieved_chunks, start=1)
                ],
            }
        )

    summary = build_summary(query_evaluations)
    output_payload = {
        "dataset": str(dataset_path),
        "collection": args.collection,
        "qdrant_url": args.qdrant_url,
        "limit": args.limit,
        "include_tables": args.include_tables,
        "use_reranker": args.use_reranker,
        "summary": summary,
        "queries": detailed_results,
    }

    write_json(output_payload, json_output_path)
    write_csv(query_evaluations, csv_output_path)

    print(f"Dataset: {dataset_path}")
    print(f"Collection: {args.collection}")
    print(f"Queries evaluated: {len(queries)}")
    print(f"Expected doc hit rate: {summary['expected_doc_hit_rate']}")
    print(f"Expected header hit rate: {summary['expected_header_hit_rate']}")
    print(f"Top-1 expected doc hit rate: {summary['top1_expected_doc_hit_rate']}")
    print(f"Top-1 expected header hit rate: {summary['top1_expected_header_hit_rate']}")
    print(f"Average doc precision: {summary['average_doc_precision']}")
    print(f"Average header precision: {summary['average_header_precision']}")
    print(f"Queries with citation noise: {summary['queries_with_citation_noise']}")
    print(f"Queries with table hits: {summary['queries_with_table_hits']}")
    print(f"Queries with non-structural headers: {summary['queries_with_non_structural_headers']}")
    print(f"Cross-document average doc precision: {summary['cross_document_average_doc_precision']}")
    print(f"JSON output: {json_output_path}")
    print(f"CSV output: {csv_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
