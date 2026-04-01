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

from src.app.adapters.embeddings.openai_embedding_adapter import OpenAIEmbeddingAdapter  # noqa: E402
from src.app.adapters.llm.openai_llm_adapter import OpenAILLMAdapter  # noqa: E402
from src.app.adapters.vectorstores.qdrant_repository import QdrantRepository  # noqa: E402
from src.app.evaluation.answer_quality_eval import (  # noqa: E402
    build_answer_quality_summary,
    evaluate_answer_quality,
    load_answer_quality_queries,
)
from src.app.services.reasoning_service import ReasoningService  # noqa: E402
from src.app.services.retrieval_service import RetrievalService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run answer-quality evaluation against a live collection and LLM."
    )
    parser.add_argument(
        "--dataset",
        default="data/eval/answer_quality_queries.json",
        help="Path to answer quality query dataset JSON.",
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
        default=8,
        help="Number of retrieved chunks per query.",
    )
    parser.add_argument(
        "--include-tables",
        action="store_true",
        help="Allow table chunks during retrieval.",
    )
    # Embedding args
    parser.add_argument(
        "--embedding-provider",
        choices=["openai", "azure_openai"],
        default=os.getenv("EMBEDDING_PROVIDER", "openai"),
    )
    parser.add_argument(
        "--embedding-api-key",
        default=os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", "")),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
    )
    parser.add_argument(
        "--embedding-azure-endpoint",
        default=os.getenv("EMBEDDING_AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT", "")),
    )
    parser.add_argument(
        "--embedding-azure-api-version",
        default=os.getenv("EMBEDDING_AZURE_OPENAI_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")),
    )
    # LLM args
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "azure_openai"],
        default=os.getenv("LLM_PROVIDER", "openai"),
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
    parser.add_argument(
        "--llm-azure-endpoint",
        default=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    )
    parser.add_argument(
        "--llm-azure-api-version",
        default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Path to write JSON results. Defaults to data/eval/results/answer_quality_eval.json.",
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


def build_llm_client(args: argparse.Namespace) -> OpenAILLMAdapter:
    return OpenAILLMAdapter(
        api_key=args.llm_api_key,
        model=args.llm_model,
        provider=args.llm_provider,
        azure_endpoint=args.llm_azure_endpoint or None,
        azure_api_version=args.llm_azure_api_version or None,
    )


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()

    if not args.embedding_api_key:
        print("ERROR: embedding API key required. Provide --embedding-api-key or set EMBEDDING_API_KEY.")
        return 1
    if not args.llm_api_key:
        print("ERROR: LLM API key required. Provide --llm-api-key or set OPENAI_API_KEY.")
        return 1

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}")
        return 1

    queries = load_answer_quality_queries(dataset_path)
    if not queries:
        print("ERROR: dataset is empty.")
        return 1

    json_output_path = Path(args.json_out) if args.json_out.strip() else Path("data/eval/results/answer_quality_eval.json")

    client = QdrantClient(url=args.qdrant_url)
    if not client.collection_exists(args.collection):
        print(f"ERROR: collection not found: {args.collection}")
        return 1

    embedding_fn = build_embedding_fn(args)
    llm_client = build_llm_client(args)
    repository = QdrantRepository(
        qdrant_client=client,
        collection_name=args.collection,
        embedding_fn=embedding_fn,
    )
    retrieval_service = RetrievalService(
        repo=repository,
        embedding_fn=embedding_fn,
        re_ranker=None,
        include_tables=args.include_tables,
    )
    reasoning_service = ReasoningService(
        retrieval_service=retrieval_service,
        llm_client=llm_client,
    )

    evaluations: list[dict[str, Any]] = []
    detailed_results: list[dict[str, Any]] = []

    for query in queries:
        print(f"  [{query.id}] {query.query[:80]}")
        answer = reasoning_service.research(query=query.query, limit=args.limit)
        evaluation = evaluate_answer_quality(query, answer)
        evaluations.append(evaluation)
        detailed_results.append({
            "query_id": query.id,
            "query": query.query,
            "notes": query.notes,
            "evaluation": evaluation,
            "answer": {
                "insight": answer.insight,
                "evidence_basis": answer.evidence_basis,
                "confidence": answer.confidence.value,
                "citation_count": len(answer.citations),
                "cited_docs": [c.doc_id for c in answer.citations],
            },
        })

    summary = build_answer_quality_summary(evaluations)
    output_payload = {
        "dataset": str(dataset_path),
        "collection": args.collection,
        "qdrant_url": args.qdrant_url,
        "limit": args.limit,
        "llm_model": args.llm_model,
        "summary": summary,
        "queries": detailed_results,
    }

    write_json(output_payload, json_output_path)

    print(f"\nDataset: {dataset_path}")
    print(f"Collection: {args.collection}")
    print(f"LLM model: {args.llm_model}")
    print(f"Queries evaluated: {len(queries)}")
    print(f"Abstain accuracy: {summary['abstain_accuracy']}")
    print(f"Has insight rate: {summary['has_insight_rate']}")
    print(f"Has evidence basis rate: {summary['has_evidence_basis_rate']}")
    if summary["confidence_meets_minimum_rate"] is not None:
        print(f"Confidence meets minimum rate: {summary['confidence_meets_minimum_rate']}")
    if summary["average_doc_id_coverage"] is not None:
        print(f"Average doc ID coverage in evidence basis: {summary['average_doc_id_coverage']}")
    print(f"Average citation count: {summary['average_citation_count']}")
    print(f"JSON output: {json_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
