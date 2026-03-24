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

from scripts.evaluate_retrieval import (  # noqa: E402
    build_embedding_fn,
    load_evaluation_queries,
    resolve_output_paths,
    write_csv,
    write_json,
)
from scripts.test_e2e_flow import ensure_collection, normalize_tables  # noqa: E402
from src.adapters.parsing.docling_parser import DoclingParser  # noqa: E402
from src.adapters.parsing.marker_parser import MarkerParser  # noqa: E402
from src.app.adapters.vectorstores.qdrant_repository import QdrantRepository  # noqa: E402
from src.app.evaluation.retrieval_eval import build_summary, evaluate_retrieval_results  # noqa: E402
from src.app.ingestion.doc_id_utils import doc_id_from_path  # noqa: E402
from src.app.services.retrieval_service import RetrievalService  # noqa: E402
from src.app.tables.table_chunker import UnifiedChunker  # noqa: E402
from src.ports.parser_port import ParsedDocument, ParserPort  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("data/parser_bakeoff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an isolated parser bakeoff without touching the active collection workflow."
    )
    parser.add_argument(
        "--pdf-dir",
        required=True,
        help="Directory containing the fixed PDF subset for parser comparison.",
    )
    parser.add_argument(
        "--parser",
        choices=["marker", "docling", "both"],
        default="both",
        help="Which parser(s) to run.",
    )
    parser.add_argument(
        "--marker-collection",
        default="medical_research_chunks_marker_bakeoff",
        help="Qdrant collection name for Marker bakeoff ingestion.",
    )
    parser.add_argument(
        "--docling-collection",
        default="medical_research_chunks_docling_bakeoff",
        help="Qdrant collection name for Docling bakeoff ingestion.",
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant base URL.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root folder for parser bakeoff artifacts and results.",
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
        "--parse-only",
        action="store_true",
        help="Only parse and write artifacts; skip chunking and Qdrant ingestion.",
    )
    parser.add_argument(
        "--recreate-collections",
        action="store_true",
        help="Delete and recreate each parser-specific bakeoff collection before ingestion.",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run retrieval evaluation on parser-specific bakeoff collections after ingestion.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of retrieved chunks per evaluation query.",
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


def selected_parsers(args: argparse.Namespace) -> list[str]:
    if args.parser == "both":
        return ["marker", "docling"]
    return [args.parser]


def parser_collection_map(args: argparse.Namespace) -> dict[str, str]:
    return {
        "marker": args.marker_collection,
        "docling": args.docling_collection,
    }


def parser_instance(parser_name: str) -> ParserPort:
    if parser_name == "marker":
        return MarkerParser()
    if parser_name == "docling":
        return DoclingParser()
    raise ValueError(f"Unsupported parser: {parser_name}")


def parser_artifact_dir(*, output_root: Path, parser_name: str, doc_id: str) -> Path:
    return output_root / "artifacts" / parser_name / doc_id


def parser_results_dir(*, output_root: Path, parser_name: str) -> Path:
    return output_root / "results" / parser_name


def write_parsed_artifacts(*, artifact_dir: Path, parsed: ParsedDocument) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "main_text.md").write_text(parsed.markdown_text, encoding="utf-8")
    for index, table in enumerate(parsed.tables, start=1):
        payload = {
            "headers": table.headers,
            "rows": table.rows,
            "csv": table.csv,
        }
        (artifact_dir / f"table_{index:02d}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (artifact_dir / f"table_{index:02d}.csv").write_text(table.csv, encoding="utf-8")


def evaluate_collection(
    *,
    collection_name: str,
    qdrant_url: str,
    embedding_fn: Any,
    limit: int,
    output_dir: Path,
) -> dict[str, Any]:
    datasets = [
        Path("data/eval/sample_queries.json"),
        Path("data/eval/expanded_queries.json"),
        Path("data/eval/ood_adversarial_queries.json"),
    ]
    client = QdrantClient(url=qdrant_url)
    repository = QdrantRepository(
        qdrant_client=client,
        collection_name=collection_name,
        embedding_fn=embedding_fn,
    )

    eval_summary: dict[str, Any] = {}
    for dataset_path in datasets:
        queries = load_evaluation_queries(dataset_path)
        query_evaluations: list[dict[str, Any]] = []
        detailed_results: list[dict[str, Any]] = []
        for query in queries:
            retrieval_service = RetrievalService(
                repo=repository,
                embedding_fn=embedding_fn,
                include_tables=query.include_tables if query.include_tables is not None else False,
            )
            retrieved_chunks = retrieval_service.retrieve(
                query=query.query,
                doc_id=query.doc_id,
                limit=limit,
            )
            evaluation = evaluate_retrieval_results(query, retrieved_chunks)
            query_evaluations.append(evaluation)
            detailed_results.append(
                {
                    "query_id": query.id,
                    "query": query.query,
                    "doc_filter": query.doc_id,
                    "labels": list(query.labels),
                    "include_tables": query.include_tables,
                    "evaluation": evaluation,
                }
            )

        summary = build_summary(query_evaluations)
        json_out, csv_out = resolve_output_paths(
            dataset_path=dataset_path,
            json_out=str(output_dir / f"{dataset_path.stem}.json"),
            csv_out=str(output_dir / f"{dataset_path.stem}.csv"),
        )
        write_json(
            {
                "dataset": str(dataset_path),
                "collection": collection_name,
                "summary": summary,
                "queries": detailed_results,
            },
            json_out,
        )
        write_csv(query_evaluations, csv_out)
        eval_summary[dataset_path.stem] = summary

    return eval_summary


def build_parser_summary(
    *,
    parser_name: str,
    collection_name: str,
    pdf_dir: Path,
    parse_only: bool,
    docs_succeeded: list[str],
    parse_failures: list[dict[str, str]],
    chunk_total: int,
    text_chunk_total: int,
    table_chunk_total: int,
    output_root: Path,
    eval_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "parser": parser_name,
        "collection": collection_name,
        "pdf_dir": str(pdf_dir),
        "parse_only": parse_only,
        "doc_count": len(docs_succeeded),
        "docs_succeeded": docs_succeeded,
        "parse_failure_count": len(parse_failures),
        "parse_failures": parse_failures,
        "chunk_count": chunk_total,
        "text_chunk_count": text_chunk_total,
        "table_chunk_count": table_chunk_total,
        "artifact_root": str(output_root / "artifacts" / parser_name),
        "results_root": str(parser_results_dir(output_root=output_root, parser_name=parser_name)),
        "eval_summary": eval_summary or {},
    }


def main() -> int:
    args = parse_args()
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"ERROR: PDF directory not found: {pdf_dir}")
        return 1

    pdf_paths = sorted(path for path in pdf_dir.rglob(args.glob) if path.is_file())
    if not pdf_paths:
        print(f"ERROR: no PDFs matched {args.glob!r} under {pdf_dir}")
        return 1

    if (not args.parse_only or args.run_eval) and not args.embedding_api_key:
        print("ERROR: embedding API key is required unless --parse-only is used without --run-eval.")
        return 1
    if args.run_eval and args.parse_only:
        print("ERROR: --run-eval cannot be combined with --parse-only.")
        return 1

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    embedding_fn = build_embedding_fn(args) if not args.parse_only or args.run_eval else None
    client = QdrantClient(url=args.qdrant_url) if not args.parse_only else None
    chunker = UnifiedChunker(max_chars=args.max_chars, overlap_paragraphs=args.overlap_paragraphs)
    collections = parser_collection_map(args)
    parser_summaries: list[dict[str, Any]] = []

    for parser_name in selected_parsers(args):
        parser = parser_instance(parser_name)
        collection_name = collections[parser_name]
        repository: QdrantRepository | None = None
        collection_initialized = False
        docs_succeeded: list[str] = []
        parse_failures: list[dict[str, str]] = []
        chunk_total = 0
        text_chunk_total = 0
        table_chunk_total = 0

        for pdf_path in pdf_paths:
            doc_id = doc_id_from_path(pdf_path)
            artifact_dir = parser_artifact_dir(output_root=output_root, parser_name=parser_name, doc_id=doc_id)
            try:
                parsed = parser.parse(pdf_path)
                write_parsed_artifacts(artifact_dir=artifact_dir, parsed=parsed)
                docs_succeeded.append(doc_id)
            except Exception as exc:  # noqa: BLE001
                parse_failures.append(
                    {
                        "doc_id": doc_id,
                        "pdf_path": str(pdf_path),
                        "error": str(exc).strip() or exc.__class__.__name__,
                    }
                )
                continue

            if args.parse_only:
                continue

            normalized_tables = normalize_tables(parsed.tables, file_name=pdf_path.name)
            chunks = chunker.chunk_document(
                doc_id=doc_id,
                source_file=pdf_path.name,
                markdown_text=parsed.markdown_text,
                tables=normalized_tables,
                local_file=str(pdf_path),
            )
            if not chunks:
                parse_failures.append(
                    {
                        "doc_id": doc_id,
                        "pdf_path": str(pdf_path),
                        "error": "no chunks generated",
                    }
                )
                continue

            if client is None or embedding_fn is None:
                print("ERROR: parser bakeoff ingestion requires embedding and Qdrant setup.")
                return 1

            if not collection_initialized:
                vector_size = len(embedding_fn([chunks[0].content])[0])
                ensure_collection(
                    client=client,
                    collection_name=collection_name,
                    vector_size=vector_size,
                    recreate=args.recreate_collections,
                )
                repository = QdrantRepository(
                    qdrant_client=client,
                    collection_name=collection_name,
                    embedding_fn=embedding_fn,
                )
                collection_initialized = True

            assert repository is not None
            repository.upsert_chunks(chunks)
            chunk_total += len(chunks)
            text_chunk_total += sum(chunk.metadata.chunk_type == "text" for chunk in chunks)
            table_chunk_total += sum(chunk.metadata.chunk_type == "table" for chunk in chunks)

        eval_summary: dict[str, Any] | None = None
        if args.run_eval and docs_succeeded:
            assert embedding_fn is not None
            eval_summary = evaluate_collection(
                collection_name=collection_name,
                qdrant_url=args.qdrant_url,
                embedding_fn=embedding_fn,
                limit=args.limit,
                output_dir=parser_results_dir(output_root=output_root, parser_name=parser_name),
            )

        summary = build_parser_summary(
            parser_name=parser_name,
            collection_name=collection_name,
            pdf_dir=pdf_dir,
            parse_only=args.parse_only,
            docs_succeeded=docs_succeeded,
            parse_failures=parse_failures,
            chunk_total=chunk_total,
            text_chunk_total=text_chunk_total,
            table_chunk_total=table_chunk_total,
            output_root=output_root,
            eval_summary=eval_summary,
        )
        summary_path = parser_results_dir(output_root=output_root, parser_name=parser_name) / "summary.json"
        write_json(summary, summary_path)
        parser_summaries.append(summary)

        print(f"Parser: {parser_name}")
        print(f"Collection: {collection_name}")
        print(f"Parsed docs: {len(docs_succeeded)}")
        print(f"Parse failures: {len(parse_failures)}")
        if not args.parse_only:
            print(f"Chunks stored: {chunk_total}")
            print(f"Text chunks: {text_chunk_total}")
            print(f"Table chunks: {table_chunk_total}")
        print(f"Summary: {summary_path}")

    comparison_path = output_root / "results" / "comparisons" / "parser_comparison.json"
    write_json(
        {
            "pdf_dir": str(pdf_dir),
            "parsers": parser_summaries,
        },
        comparison_path,
    )
    print(f"Comparison: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
