from __future__ import annotations

import csv
import os
from pathlib import Path
import re
import sys

from dotenv import load_dotenv


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_project_root_on_path()
load_dotenv()

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Streamlit is not installed. Run '.\\.venv\\Scripts\\python.exe -m pip install streamlit' first."
    ) from exc

from qdrant_client import QdrantClient  # noqa: E402

from src.app.adapters.embeddings.openai_embedding_adapter import OpenAIEmbeddingAdapter  # noqa: E402
from src.app.ingestion.dedup_utils import ensure_doc_identity_is_available, fetch_collection_doc_identities  # noqa: E402
from src.app.ingestion.doc_id_utils import doc_id_from_path  # noqa: E402
from src.app.ingestion.file_identity_utils import compute_file_identity  # noqa: E402
from src.app.ingestion.parser_factory import DEFAULT_PARSER_NAME, PARSER_CHOICES, build_parser  # noqa: E402
from src.app.ingestion.runtime_utils import ensure_collection, normalize_tables  # noqa: E402
from src.app.ingestion.registry_utils import (  # noqa: E402
    get_collection_docs as registry_collection_docs,
    load_registry as load_registry_file,
    save_registry as save_registry_file,
    sync_collection_from_manifest,
    upsert_collection_doc,
)
from src.app.adapters.llm.openai_llm_adapter import OpenAILLMAdapter  # noqa: E402
from src.app.adapters.rerankers.transformers_reranker import TransformersReRanker  # noqa: E402
from src.app.adapters.vectorstores.qdrant_repository import QdrantRepository  # noqa: E402
from src.app.services.reasoning_service import ConfidenceLevel, ResearchAnswer, ReasoningService  # noqa: E402
from src.app.services.retrieval_service import RetrievedChunk, RetrievalService  # noqa: E402
from src.app.tables.table_chunker import UnifiedChunker  # noqa: E402


UPLOAD_DIR = Path("data/raw_pdfs/uploaded")
REGISTRY_PATH = Path("data/kb_registry.json")
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "medical_research_chunks_docling_v2_batch1")
DEFAULT_EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "azure_openai")
COLLECTION_ROLES: dict[str, str] = {
    "medical_research_chunks_docling_v2_batch1": "Stage-1 artifact (20 PDFs) - passing",
    "medical_research_chunks_docling_v1": "Baseline small corpus",
    "medical_research_chunks_v1": "Marker rollback - read-only",
}


def get_collection_role(name: str) -> str:
    return COLLECTION_ROLES.get(name, "Custom collection")


def build_embedding_fn(
    provider: str,
    api_key: str,
    model: str,
    dimensions: int | None,
    azure_endpoint: str,
    azure_api_version: str,
) -> OpenAIEmbeddingAdapter:
    return OpenAIEmbeddingAdapter(
        api_key=api_key,
        model=model,
        provider=provider,
        azure_endpoint=azure_endpoint or None,
        azure_api_version=azure_api_version or None,
        dimensions=dimensions,
    )


def validate_collection_exists(client: QdrantClient, collection_name: str) -> None:
    if not client.collection_exists(collection_name):
        raise RuntimeError(
            f"Collection '{collection_name}' does not exist in Qdrant yet. "
            "Ingest at least one PDF into the active collection first."
        )


def ingest_pdf(
    pdf_path: Path,
    doc_id: str,
    collection_name: str,
    parser_name: str,
    qdrant_url: str,
    max_chars: int,
    overlap_paragraphs: int,
    embedding_fn: OpenAIEmbeddingAdapter,
) -> dict[str, object]:
    file_identity = compute_file_identity(pdf_path)
    existing_registry_docs = list(get_collection_docs(collection_name).values())
    try:
        ensure_doc_identity_is_available(
            doc_id=doc_id,
            source_file=pdf_path.name,
            local_file=str(pdf_path),
            source_sha256=str(file_identity["source_sha256"]),
            existing_entries=existing_registry_docs,
            context=f"Registry collection '{collection_name}'",
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    client = QdrantClient(url=qdrant_url)
    try:
        ensure_doc_identity_is_available(
            doc_id=doc_id,
            source_file=pdf_path.name,
            local_file=str(pdf_path),
            source_sha256=str(file_identity["source_sha256"]),
            existing_entries=fetch_collection_doc_identities(client, collection_name=collection_name),
            context=f"Qdrant collection '{collection_name}'",
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    document_parser = build_parser(parser_name)
    parsed_document = document_parser.parse(pdf_path)
    normalized_tables = normalize_tables(parsed_document.tables, file_name=pdf_path.name)

    chunker = UnifiedChunker(max_chars=max_chars, overlap_paragraphs=overlap_paragraphs)
    chunks = chunker.chunk_document(
        doc_id=doc_id,
        source_file=pdf_path.name,
        markdown_text=parsed_document.markdown_text,
        tables=normalized_tables,
        local_file=str(pdf_path),
        source_sha256=str(file_identity["source_sha256"]),
        file_size_bytes=int(file_identity["file_size_bytes"]),
    )

    if not chunks:
        raise RuntimeError("No chunks were generated from the document.")

    vector_size = len(embedding_fn([chunks[0].content])[0])
    ensure_collection(client, collection_name, vector_size)

    repository = QdrantRepository(
        qdrant_client=client,
        collection_name=collection_name,
        embedding_fn=embedding_fn,
    )
    repository.upsert_chunks(chunks)

    text_chunk_count = sum(chunk.metadata.chunk_type == "text" for chunk in chunks)
    table_chunk_count = sum(chunk.metadata.chunk_type == "table" for chunk in chunks)
    return {
        "doc_id": doc_id,
        "pdf_path": str(pdf_path),
        "chunks": len(chunks),
        "text_chunks": text_chunk_count,
        "table_chunks": table_chunk_count,
        "ingestion_version": UnifiedChunker.INGESTION_VERSION,
        "chunker_version": UnifiedChunker.CHUNKER_VERSION,
        "chunking_version": UnifiedChunker.CHUNKING_VERSION,
        "parser": parser_name,
        "source_file": pdf_path.name,
        "source_sha256": str(file_identity["source_sha256"]),
        "file_size_bytes": int(file_identity["file_size_bytes"]),
    }


def ask_question(
    query: str,
    collection_name: str,
    qdrant_url: str,
    limit: int,
    use_reranker: bool,
    reranker_model: str,
    embedding_fn: OpenAIEmbeddingAdapter,
    include_tables: bool,
) -> list[RetrievedChunk]:
    client = QdrantClient(url=qdrant_url)
    validate_collection_exists(client, collection_name)
    repository = QdrantRepository(
        qdrant_client=client,
        collection_name=collection_name,
        embedding_fn=embedding_fn,
    )
    re_ranker = None
    if use_reranker:
        re_ranker = TransformersReRanker(model_name=reranker_model)
    retrieval_service = RetrievalService(
        repo=repository,
        embedding_fn=embedding_fn,
        re_ranker=re_ranker,
        include_tables=include_tables,
    )
    return retrieval_service.retrieve(query=query, limit=limit)


def ask_research_question(
    query: str,
    collection_name: str,
    qdrant_url: str,
    limit: int,
    llm_provider: str,
    llm_api_key: str,
    llm_model: str,
    azure_endpoint: str,
    azure_api_version: str,
    use_reranker: bool,
    reranker_model: str,
    embedding_fn: OpenAIEmbeddingAdapter,
    include_tables: bool,
) -> ResearchAnswer:
    client = QdrantClient(url=qdrant_url)
    validate_collection_exists(client, collection_name)
    repository = QdrantRepository(
        qdrant_client=client,
        collection_name=collection_name,
        embedding_fn=embedding_fn,
    )
    re_ranker = None
    if use_reranker:
        re_ranker = TransformersReRanker(model_name=reranker_model)
    retrieval_service = RetrievalService(
        repo=repository,
        embedding_fn=embedding_fn,
        re_ranker=re_ranker,
        include_tables=include_tables,
    )
    llm_client = OpenAILLMAdapter(
        api_key=llm_api_key,
        model=llm_model,
        provider=llm_provider,
        azure_endpoint=azure_endpoint or None,
        azure_api_version=azure_api_version or None,
    )
    reasoning_service = ReasoningService(
        retrieval_service=retrieval_service,
        llm_client=llm_client,
    )
    return reasoning_service.research(query=query, limit=limit)


def save_uploaded_file(uploaded_file: object) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / uploaded_file.name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def _format_runtime_error(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message
    return f"{exc.__class__.__name__} occurred during runtime."


def load_registry() -> dict[str, dict[str, object]]:
    return load_registry_file(REGISTRY_PATH)


def save_registry(registry: dict[str, dict[str, object]]) -> None:
    save_registry_file(REGISTRY_PATH, registry)


def get_collection_docs(collection_name: str) -> dict[str, dict[str, object]]:
    registry = load_registry()
    sync_collection_from_manifest(registry, collection_name=collection_name)
    save_registry(registry)
    return registry_collection_docs(registry, collection_name)


def update_collection_docs(collection_name: str, doc_id: str, summary: dict[str, object]) -> None:
    registry = load_registry()
    upsert_collection_doc(
        registry,
        collection_name=collection_name,
        doc_id=doc_id,
        summary=summary,
    )
    save_registry(registry)


def init_state() -> None:
    st.session_state.setdefault("active_collection", "")
    st.session_state.setdefault("ingested_docs", None)
    st.session_state.setdefault("last_answer", [])
    st.session_state.setdefault("last_research_answer", None)


def render_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(196, 226, 255, 0.9), transparent 35%),
                linear-gradient(160deg, #f6f3ec 0%, #e7ecef 100%);
        }
        .hero-card {
            padding: 1.2rem 1.4rem;
            border: 1px solid rgba(30, 42, 56, 0.12);
            border-radius: 18px;
            background: rgba(255, 252, 247, 0.86);
            box-shadow: 0 10px 30px rgba(35, 48, 65, 0.08);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 2rem;
            color: #1c2833;
            margin-bottom: 0.2rem;
        }
        .hero-copy {
            color: #425466;
            font-size: 1rem;
        }
        .kb-card {
            padding: 1rem 1.2rem;
            border: 1px solid rgba(30, 42, 56, 0.12);
            border-radius: 16px;
            background: rgba(252, 249, 244, 0.9);
            box-shadow: 0 8px 24px rgba(35, 48, 65, 0.06);
            margin: 0.4rem 0 1.2rem 0;
        }
        .kb-title {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.15rem;
            color: #1c2833;
            margin-bottom: 0.5rem;
        }
        .kb-row {
            padding: 0.45rem 0;
            border-top: 1px solid rgba(30, 42, 56, 0.08);
            color: #304050;
        }
        .kb-row:first-of-type {
            border-top: none;
        }
        .result-card {
            padding: 1rem 1.2rem;
            border: 1px solid rgba(30, 42, 56, 0.12);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 8px 24px rgba(35, 48, 65, 0.05);
            margin: 0.75rem 0;
        }
        .result-title {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.05rem;
            color: #1c2833;
            margin-bottom: 0.3rem;
        }
        .result-meta {
            color: #5a6877;
            font-size: 0.92rem;
            margin-bottom: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_knowledge_base_summary() -> None:
    if not st.session_state.ingested_docs:
        st.markdown(
            """
            <div class="kb-card">
                <div class="kb-title">Current Knowledge Base</div>
                <div class="kb-row">No documents registered for the active collection.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    rows = []
    for doc_id, summary in st.session_state.ingested_docs.items():
        pdf_name = Path(summary["pdf_path"]).name
        rows.append(
            f"<div class='kb-row'><strong>{pdf_name}</strong> "
            f"({doc_id})<br>{summary['chunks']} chunks | "
            f"{summary['text_chunks']} text | {summary['table_chunks']} table</div>"
        )

    st.markdown(
        (
            "<div class='kb-card'>"
            "<div class='kb-title'>Current Knowledge Base</div>"
            + "".join(rows)
            + "</div>"
        ),
        unsafe_allow_html=True,
    )


def _parse_csv_table(text: str) -> pd.DataFrame | None:
    try:
        reader = csv.reader(StringIO(text))
        rows = list(reader)
        if len(rows) < 2:
            return None
        max_cols = max(len(row) for row in rows)
        if max_cols < 2:
            return None
        normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]
        header = _make_unique_columns(normalized_rows[0], max_cols)
        data = normalized_rows[1:]
        return pd.DataFrame(data, columns=header)
    except Exception:
        return None


def _make_unique_columns(header: list[str], width: int) -> list[str]:
    cleaned_header = [(value or "").strip() for value in header]
    unique_header: list[str] = []
    seen: dict[str, int] = {}

    for idx in range(width):
        base_name = cleaned_header[idx] if idx < len(cleaned_header) else ""
        if not base_name:
            base_name = f"column_{idx + 1}"

        count = seen.get(base_name, 0)
        seen[base_name] = count + 1
        if count == 0:
            unique_header.append(base_name)
        else:
            unique_header.append(f"{base_name}_{count + 1}")

    return unique_header


def render_retrieved_context(chunks: list[RetrievedChunk], docs: dict[str, dict[str, object]]) -> None:
    if not chunks:
        st.info("No retrieval output yet.")
        return

    for index, chunk in enumerate(chunks, start=1):
        doc_summary = docs.get(chunk.doc_id, {})
        pdf_path = chunk.local_file.strip() or str(doc_summary.get("pdf_path", "")).strip()

        with st.container(border=True):
            st.markdown(
                (
                    f"<div class='result-title'>Chunk {index}: {chunk.source}</div>"
                    f"<div class='result-meta'>Document: {chunk.doc_id or 'Unknown'}</div>"
                ),
                unsafe_allow_html=True,
            )

            if pdf_path:
                st.caption(f"Local file: {pdf_path}")
                pdf_file = Path(pdf_path)
                if pdf_file.exists():
                    st.download_button(
                        label=f"Download {pdf_file.name}",
                        data=pdf_file.read_bytes(),
                        file_name=pdf_file.name,
                        mime="application/pdf",
                        key=f"download-{chunk.doc_id}-{index}",
                    )

            content = chunk.content
            lines = content.splitlines()
            preface_lines: list[str] = []
            if lines and lines[0].startswith("Source File:"):
                preface_lines.append(lines[0])
                content = "\n".join(lines[1:]).strip()

            if preface_lines:
                for line in preface_lines:
                    st.caption(line)

            if chunk.chunk_type == "table" or chunk.content_role == "table":
                table_df = _parse_csv_table(content)
                if table_df is not None:
                    st.dataframe(table_df, use_container_width=True, hide_index=True)
                else:
                    st.markdown(content if content else "_No content_")
            else:
                st.markdown(content if content else "_No content_")


def main() -> None:
    st.set_page_config(page_title="Medical Research RAG", page_icon="M", layout="wide")
    init_state()
    render_styles()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Medical Research RAG Workbench</div>
            <div class="hero-copy">
                Upload PDF studies, ingest them into Qdrant, and query document-scoped evidence using the current pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Runtime")
        qdrant_url = st.text_input("Qdrant URL", value=DEFAULT_QDRANT_URL)
        collection_name = st.text_input("Collection", value=DEFAULT_COLLECTION_NAME)
        if collection_name != st.session_state.active_collection:
            st.session_state.active_collection = collection_name
            st.session_state.last_retrieval_result = None
            st.session_state.last_research_answer = None
            st.session_state.last_research_latency_ms = None
            st.session_state.ingested_docs = None
        st.caption(get_collection_role(collection_name))
        parser_name = st.selectbox(
            "Parser",
            options=list(PARSER_CHOICES),
            index=list(PARSER_CHOICES).index(DEFAULT_PARSER_NAME),
            help="Only affects newly ingested documents.",
        )
        max_chars = st.number_input("Max chars", min_value=200, max_value=4000, value=1800, step=100)
        overlap_paragraphs = st.number_input("Overlap paragraphs", min_value=0, max_value=5, value=1, step=1)
        retrieval_limit = st.number_input("Retrieval limit", min_value=1, max_value=20, value=5, step=1)
        st.header("Embeddings")
        embedding_provider_label = st.selectbox(
            "Embedding Provider",
            options=["OpenAI", "Azure OpenAI"],
            index=1 if DEFAULT_EMBEDDING_PROVIDER == "azure_openai" else 0,
        )
        embedding_provider = "azure_openai" if embedding_provider_label == "Azure OpenAI" else "openai"
        embedding_api_key = st.text_input(
            "Embedding API Key",
            value=os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            type="password",
        )
        embedding_model = st.text_input(
            "Embedding Model / Deployment",
            value=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        )
        embedding_dimensions = st.number_input(
            "Embedding Dimensions",
            min_value=0,
            max_value=3072,
            value=int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
            step=256,
            help="Set to 0 to use the provider default output dimension.",
        )
        embedding_azure_endpoint = ""
        embedding_azure_api_version = ""
        if embedding_provider == "azure_openai":
            embedding_azure_endpoint = st.text_input(
                "Embedding Azure Endpoint",
                value=os.getenv("EMBEDDING_AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT", "")),
                placeholder="https://your-resource.openai.azure.com",
            )
            embedding_azure_api_version = st.text_input(
                "Embedding Azure API Version",
                value=os.getenv("EMBEDDING_AZURE_OPENAI_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")),
            )
        st.header("Retrieval")
        include_tables = st.checkbox("Include table chunks", value=False)
        use_reranker = st.checkbox("Enable re-ranking", value=False)
        reranker_model = st.text_input(
            "Re-ranker model",
            value="cross-encoder/ms-marco-MiniLM-L-6-v2",
            disabled=not use_reranker,
        )
        if use_reranker:
            st.caption("Re-ranking downloads and runs a cross-encoder model locally on first use.")
        st.header("Research LLM")
        llm_provider_label = st.selectbox(
            "Provider",
            options=["OpenAI", "Azure OpenAI"],
            index=0,
        )
        llm_provider = "azure_openai" if llm_provider_label == "Azure OpenAI" else "openai"
        llm_api_key = st.text_input(
            "API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
        )
        llm_model = st.text_input("Model / Deployment", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        azure_endpoint = ""
        azure_api_version = ""
        if llm_provider == "azure_openai":
            azure_endpoint = st.text_input(
                "Azure Endpoint",
                value=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                placeholder="https://your-resource.openai.azure.com",
            )
            azure_api_version = st.text_input(
                "Azure API Version",
                value=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            )

        st.session_state.ingested_docs = get_collection_docs(collection_name)

        st.header("Indexed Documents")
        if st.session_state.ingested_docs:
            for doc_id, summary in st.session_state.ingested_docs.items():
                st.caption(
                    f"{doc_id}: {summary['chunks']} chunks "
                    f"({summary['text_chunks']} text / {summary['table_chunks']} table) "
                    f"[parser={summary.get('parser', '') or 'unknown'}]"
                )
        else:
            st.caption("No documents ingested in this session.")

    embedding_fn: OpenAIEmbeddingAdapter | None = None
    embedding_setup_error = ""
    embedding_dimensions_value = None if int(embedding_dimensions) == 0 else int(embedding_dimensions)
    if embedding_api_key.strip():
        try:
            embedding_fn = build_embedding_fn(
                provider=embedding_provider,
                api_key=embedding_api_key.strip(),
                model=embedding_model.strip() or "text-embedding-3-large",
                dimensions=embedding_dimensions_value,
                azure_endpoint=embedding_azure_endpoint.strip(),
                azure_api_version=embedding_azure_api_version.strip(),
            )
        except Exception as exc:  # noqa: BLE001
            embedding_setup_error = str(exc)

    if embedding_setup_error:
        st.warning(f"Embedding configuration error: {embedding_setup_error}")

    upload_col, _ = st.columns([0.9, 1.5], gap="large")

    with upload_col:
        st.subheader("Upload and Ingest")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
        is_rollback_collection = collection_name == "medical_research_chunks_v1"
        if is_rollback_collection:
            st.warning("Rollback collection selected. Ingestion is disabled for this collection.")

        if st.button("Ingest Uploaded PDFs", use_container_width=True):
            if is_rollback_collection:
                st.warning("Switch to a different collection before ingesting.")
            elif not uploaded_files:
                st.warning("Choose at least one PDF file first.")
            elif embedding_fn is None:
                st.warning("Provide embedding credentials in the sidebar before ingesting.")
            else:
                progress = st.progress(0.0)
                success_count = 0
                failed_uploads: list[str] = []
                for index, uploaded_file in enumerate(uploaded_files, start=1):
                    try:
                        pdf_path = save_uploaded_file(uploaded_file)
                        doc_id = doc_id_from_path(pdf_path)
                        with st.spinner(f"Ingesting {uploaded_file.name}"):
                            summary = ingest_pdf(
                                pdf_path=pdf_path,
                                doc_id=doc_id,
                                collection_name=collection_name,
                                parser_name=parser_name,
                                qdrant_url=qdrant_url,
                                max_chars=int(max_chars),
                                overlap_paragraphs=int(overlap_paragraphs),
                                embedding_fn=embedding_fn,
                            )
                        st.session_state.ingested_docs[doc_id] = summary
                        update_collection_docs(collection_name, doc_id, summary)
                        success_count += 1
                    except Exception as exc:  # noqa: BLE001
                        failed_uploads.append(uploaded_file.name)
                        st.error(f"Failed to ingest {uploaded_file.name}: {_format_runtime_error(exc)}")
                    progress.progress(index / len(uploaded_files))

                if success_count and not failed_uploads:
                    st.success(f"Ingestion complete. {success_count} file(s) ingested.")
                elif success_count and failed_uploads:
                    st.warning(
                        f"Ingestion finished with partial failures. "
                        f"Successful: {success_count}. Failed: {len(failed_uploads)}."
                    )
                else:
                    st.error("Ingestion failed for all uploaded files.")

    st.divider()
    render_knowledge_base_summary()
    st.subheader("Ask Questions")
    st.caption("Questions are answered from the current knowledge base in the active collection.")
    query = st.text_area(
        "Question",
        value="What does the paper say about lipid biomarkers?",
        height=140,
    )

    if st.button("Retrieve Evidence", use_container_width=True):
        if not st.session_state.ingested_docs:
            st.warning("Ingest at least one document before running retrieval.")
        elif not query.strip():
            st.warning("Enter a question first.")
        elif embedding_fn is None:
            st.warning("Provide embedding credentials in the sidebar before running retrieval.")
        else:
            with st.spinner("Retrieving knowledge-base context"):
                try:
                    st.session_state.last_answer = ask_question(
                        query=query.strip(),
                        collection_name=collection_name,
                        qdrant_url=qdrant_url,
                        limit=int(retrieval_limit),
                        use_reranker=use_reranker,
                        reranker_model=reranker_model.strip() or "cross-encoder/ms-marco-MiniLM-L-6-v2",
                        embedding_fn=embedding_fn,
                        include_tables=include_tables,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

    if st.button("Ask Research Question", use_container_width=True):
        if not st.session_state.ingested_docs:
            st.warning("Ingest at least one document before running research.")
        elif not query.strip():
            st.warning("Enter a question first.")
        elif embedding_fn is None:
            st.warning("Provide embedding credentials in the sidebar before running research.")
        elif not llm_api_key.strip():
            st.warning("Provide an API key in the sidebar.")
        elif llm_provider == "azure_openai" and not azure_endpoint.strip():
            st.warning("Provide an Azure OpenAI endpoint in the sidebar.")
        elif llm_provider == "azure_openai" and not azure_api_version.strip():
            st.warning("Provide an Azure OpenAI API version in the sidebar.")
        else:
            with st.spinner("Synthesizing research insight"):
                try:
                    st.session_state.last_research_answer = ask_research_question(
                        query=query.strip(),
                        collection_name=collection_name,
                        qdrant_url=qdrant_url,
                        limit=int(retrieval_limit),
                        llm_provider=llm_provider,
                        llm_api_key=llm_api_key.strip(),
                        llm_model=llm_model.strip() or "gpt-4o-mini",
                        azure_endpoint=azure_endpoint.strip(),
                        azure_api_version=azure_api_version.strip(),
                        use_reranker=use_reranker,
                        reranker_model=reranker_model.strip() or "cross-encoder/ms-marco-MiniLM-L-6-v2",
                        embedding_fn=embedding_fn,
                        include_tables=include_tables,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

    st.markdown("### Retrieved Context")
    render_retrieved_context(
        st.session_state.last_answer,
        st.session_state.ingested_docs or {},
    )

    st.markdown("### Research Insight")
    answer: ResearchAnswer | None = st.session_state.last_research_answer
    if answer is None:
        st.markdown("_No research answer yet._")
    else:
        distinct_docs = len({c.doc_id for c in answer.citations})
        confidence_label = (
            f"Evidence confidence: {answer.confidence.value} "
            f"({len(answer.citations)} chunks, {distinct_docs} document(s))"
        )
        if answer.confidence == ConfidenceLevel.HIGH:
            st.success(confidence_label)
        elif answer.confidence == ConfidenceLevel.MEDIUM:
            st.info(confidence_label)
        elif answer.confidence == ConfidenceLevel.LOW:
            st.warning(confidence_label)
        else:
            st.error(confidence_label)

        st.markdown(answer.insight or "_No insight generated._")
        if answer.evidence_basis:
            with st.expander("Evidence Basis"):
                st.markdown(answer.evidence_basis)
        if answer.citations:
            with st.expander(f"Citations ({len(answer.citations)} chunks)"):
                for chunk in answer.citations:
                    page = f", p. {chunk.page_number}" if chunk.page_number else ""
                    st.markdown(
                        f"- **{chunk.doc_id}** — {chunk.source} ({chunk.chunk_type}{page})"
                    )


if __name__ == "__main__":
    main()
