from __future__ import annotations

import csv
import json
import math
import os
from io import StringIO
from pathlib import Path
import re
import sys

import pandas as pd


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_project_root_on_path()

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Streamlit is not installed. Run '.\\.venv\\Scripts\\python.exe -m pip install streamlit' first."
    ) from exc

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import Distance, VectorParams  # noqa: E402

from src.adapters.parsing.marker_parser import MarkerParser  # noqa: E402
from src.app.adapters.llm.openai_llm_adapter import OpenAILLMAdapter  # noqa: E402
from src.app.adapters.rerankers.transformers_reranker import TransformersReRanker  # noqa: E402
from src.app.adapters.vectorstores.qdrant_repository import QdrantRepository  # noqa: E402
from src.app.services.reasoning_service import ReasoningService  # noqa: E402
from src.app.services.retrieval_service import RetrievalService  # noqa: E402
from src.app.tables.table_chunker import UnifiedChunker  # noqa: E402
from src.app.tables.table_normalizer import TableNormalizer  # noqa: E402
from src.ports.parser_port import ParsedTable  # noqa: E402


UPLOAD_DIR = Path("data/raw_pdfs/uploaded")
REGISTRY_PATH = Path("data/kb_registry.json")


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


def normalize_tables(parsed_tables: list[ParsedTable], file_name: str) -> list[dict[str, object]]:
    normalizer = TableNormalizer()
    normalized: list[dict[str, object]] = []

    for table in parsed_tables:
        df = parsed_table_to_dataframe(table)
        cleaned_df = normalizer.sanitize_table(df=df, file_name=file_name)
        artifact = {
            "csv": cleaned_df.to_csv(index=False, header=False).strip(),
            "rows": dataframe_to_rows(cleaned_df),
        }

        metadata_artifact = normalizer.get_last_metadata_artifact()
        if metadata_artifact:
            artifact["normalization_metadata"] = metadata_artifact

        normalized.append(artifact)

    return normalized


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def ingest_pdf(
    pdf_path: Path,
    doc_id: str,
    collection_name: str,
    qdrant_url: str,
    max_chars: int,
    overlap_paragraphs: int,
) -> dict[str, object]:
    parser = MarkerParser()
    parsed_document = parser.parse(pdf_path)
    normalized_tables = normalize_tables(parsed_document.tables, file_name=pdf_path.name)

    chunker = UnifiedChunker(max_chars=max_chars, overlap_paragraphs=overlap_paragraphs)
    chunks = chunker.chunk_document(
        doc_id=doc_id,
        source_file=pdf_path.name,
        markdown_text=parsed_document.markdown_text,
        tables=normalized_tables,
    )

    if not chunks:
        raise RuntimeError("No chunks were generated from the document.")

    client = QdrantClient(url=qdrant_url)
    vector_size = len(simple_embedding_fn([chunks[0].content])[0])
    ensure_collection(client, collection_name, vector_size)

    repository = QdrantRepository(
        qdrant_client=client,
        collection_name=collection_name,
        embedding_fn=simple_embedding_fn,
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
    }


def ask_question(
    query: str,
    collection_name: str,
    qdrant_url: str,
    limit: int,
    use_reranker: bool,
    reranker_model: str,
) -> str:
    client = QdrantClient(url=qdrant_url)
    repository = QdrantRepository(
        qdrant_client=client,
        collection_name=collection_name,
        embedding_fn=simple_embedding_fn,
    )
    re_ranker = None
    if use_reranker:
        re_ranker = TransformersReRanker(model_name=reranker_model)
    retrieval_service = RetrievalService(
        repo=repository,
        embedding_fn=simple_embedding_fn,
        re_ranker=re_ranker,
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
) -> str:
    client = QdrantClient(url=qdrant_url)
    repository = QdrantRepository(
        qdrant_client=client,
        collection_name=collection_name,
        embedding_fn=simple_embedding_fn,
    )
    re_ranker = None
    if use_reranker:
        re_ranker = TransformersReRanker(model_name=reranker_model)
    retrieval_service = RetrievalService(
        repo=repository,
        embedding_fn=simple_embedding_fn,
        re_ranker=re_ranker,
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


def load_registry() -> dict[str, dict[str, object]]:
    if not REGISTRY_PATH.exists():
        return {}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def save_registry(registry: dict[str, dict[str, object]]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def get_collection_docs(collection_name: str) -> dict[str, dict[str, object]]:
    registry = load_registry()
    collections = registry.get("collections", {})
    collection_docs = collections.get(collection_name, {})
    if not isinstance(collection_docs, dict):
        return {}
    return collection_docs


def update_collection_docs(collection_name: str, doc_id: str, summary: dict[str, object]) -> None:
    registry = load_registry()
    collections = registry.setdefault("collections", {})
    collection_docs = collections.setdefault(collection_name, {})
    collection_docs[doc_id] = summary
    save_registry(registry)


def init_state() -> None:
    st.session_state.setdefault("ingested_docs", None)
    st.session_state.setdefault("last_answer", "")
    st.session_state.setdefault("last_research_answer", "")


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


def parse_retrieved_context(answer: str) -> list[dict[str, str]]:
    if not answer.strip():
        return []

    blocks = re.split(r"\n{2,}(?=Source: )", answer.strip())
    parsed: list[dict[str, str]] = []
    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        first_line = lines[0]
        match = re.match(r"^Source:\s*(.*?)\s*\|\s*Document:\s*(.+?)\s*$", first_line)
        if match:
            source = match.group(1).strip()
            doc_id = match.group(2).strip()
        else:
            source = first_line.removeprefix("Source:").strip()
            doc_id = ""

        content = "\n".join(lines[1:]).strip()
        parsed.append({"source": source, "doc_id": doc_id, "content": content})
    return parsed


def _looks_like_csv_block(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    csv_candidate_lines = [line for line in lines if line.count(",") >= 1]
    return len(csv_candidate_lines) >= 3


def _parse_csv_table(text: str) -> pd.DataFrame | None:
    if not _looks_like_csv_block(text):
        return None

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


def render_retrieved_context(answer: str, docs: dict[str, dict[str, object]]) -> None:
    chunks = parse_retrieved_context(answer)
    if not chunks:
        st.info("No retrieval output yet.")
        return

    for index, chunk in enumerate(chunks, start=1):
        doc_summary = docs.get(chunk["doc_id"], {})
        pdf_path = str(doc_summary.get("pdf_path", "")).strip()

        with st.container(border=True):
            st.markdown(
                (
                    f"<div class='result-title'>Chunk {index}: {chunk['source']}</div>"
                    f"<div class='result-meta'>Document: {chunk['doc_id'] or 'Unknown'}</div>"
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
                        key=f"download-{chunk['doc_id']}-{index}",
                    )

            content = chunk["content"]
            lines = content.splitlines()
            preface_lines: list[str] = []
            if lines and lines[0].startswith("Source File:"):
                preface_lines.append(lines[0])
                content = "\n".join(lines[1:]).strip()

            if preface_lines:
                for line in preface_lines:
                    st.caption(line)

            table_df = _parse_csv_table(content)
            if table_df is not None:
                st.dataframe(table_df, use_container_width=True, hide_index=True)
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
        qdrant_url = st.text_input("Qdrant URL", value="http://localhost:6333")
        collection_name = st.text_input("Collection", value="medical_research_chunks")
        max_chars = st.number_input("Max chars", min_value=200, max_value=4000, value=900, step=100)
        overlap_paragraphs = st.number_input("Overlap paragraphs", min_value=0, max_value=5, value=1, step=1)
        retrieval_limit = st.number_input("Retrieval limit", min_value=1, max_value=20, value=5, step=1)
        st.header("Retrieval")
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
                    f"({summary['text_chunks']} text / {summary['table_chunks']} table)"
                )
        else:
            st.caption("No documents ingested in this session.")

    upload_col, _ = st.columns([0.9, 1.5], gap="large")

    with upload_col:
        st.subheader("Upload and Ingest")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("Ingest Uploaded PDFs", use_container_width=True):
            if not uploaded_files:
                st.warning("Choose at least one PDF file first.")
            else:
                progress = st.progress(0.0)
                for index, uploaded_file in enumerate(uploaded_files, start=1):
                    pdf_path = save_uploaded_file(uploaded_file)
                    doc_id = pdf_path.stem
                    with st.spinner(f"Ingesting {uploaded_file.name}"):
                        summary = ingest_pdf(
                            pdf_path=pdf_path,
                            doc_id=doc_id,
                            collection_name=collection_name,
                            qdrant_url=qdrant_url,
                            max_chars=int(max_chars),
                            overlap_paragraphs=int(overlap_paragraphs),
                        )
                    st.session_state.ingested_docs[doc_id] = summary
                    update_collection_docs(collection_name, doc_id, summary)
                    progress.progress(index / len(uploaded_files))
                st.success("Ingestion complete.")

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
        else:
            with st.spinner("Retrieving knowledge-base context"):
                st.session_state.last_answer = ask_question(
                    query=query.strip(),
                    collection_name=collection_name,
                    qdrant_url=qdrant_url,
                    limit=int(retrieval_limit),
                    use_reranker=use_reranker,
                    reranker_model=reranker_model.strip() or "cross-encoder/ms-marco-MiniLM-L-6-v2",
                )

    if st.button("Ask Research Question", use_container_width=True):
        if not st.session_state.ingested_docs:
            st.warning("Ingest at least one document before running research.")
        elif not query.strip():
            st.warning("Enter a question first.")
        elif not llm_api_key.strip():
            st.warning("Provide an API key in the sidebar.")
        elif llm_provider == "azure_openai" and not azure_endpoint.strip():
            st.warning("Provide an Azure OpenAI endpoint in the sidebar.")
        elif llm_provider == "azure_openai" and not azure_api_version.strip():
            st.warning("Provide an Azure OpenAI API version in the sidebar.")
        else:
            with st.spinner("Synthesizing research insight"):
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
                )

    st.markdown("### Retrieved Context")
    render_retrieved_context(
        st.session_state.last_answer,
        st.session_state.ingested_docs or {},
    )

    st.markdown("### Research Insight")
    st.markdown(st.session_state.last_research_answer or "_No research answer yet._")


if __name__ == "__main__":
    main()
