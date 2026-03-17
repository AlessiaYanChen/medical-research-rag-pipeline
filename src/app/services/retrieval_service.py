from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import re

from src.app.ports.re_ranker_port import ReRankerPort
from src.app.ports.repositories.vector_repository import VectorRepositoryPort
from src.domain.models.chunk import Chunk


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    doc_id: str
    content: str
    chunk_type: str
    content_role: str
    page_number: int | None = None
    local_file: str = ""


class RetrievalService:
    """Application service for semantic retrieval over the active knowledge base."""

    def __init__(
        self,
        repo: VectorRepositoryPort,
        embedding_fn: Callable[[list[str]], list[list[float]]],
        re_ranker: ReRankerPort | None = None,
        include_tables: bool = False,
    ) -> None:
        self._repo = repo
        self._embedding_fn = embedding_fn
        self._re_ranker = re_ranker
        self._include_tables = include_tables

    def retrieve(self, query: str, doc_id: str | None = None, limit: int = 5) -> list[RetrievedChunk]:
        query_vector = self._embedding_fn([query])[0]
        initial_limit = max(limit * 4, 20)
        initial_chunks = self._repo.search(query_vector, doc_id=doc_id, limit=initial_limit)
        filtered_initial_chunks = self._filter_chunks(query=query, chunks=initial_chunks)
        candidate_limit = max(limit * 4, 20)
        chunks = self._select_candidate_chunks(query=query, initial_chunks=filtered_initial_chunks, candidate_limit=candidate_limit)
        ranked_chunks = self._rank_chunks(chunks)

        retrieved_chunks: list[RetrievedChunk] = []
        seen_parent_ids: set[str] = set()
        selected_contents: list[str] = []
        header_counts: dict[str, int] = {}
        doc_counts: dict[str, int] = {}
        for chunk in ranked_chunks:
            parent_id = str(chunk.metadata.extra.get("parent_id", chunk.id))
            if parent_id in seen_parent_ids:
                continue

            raw_content = self._select_return_content(chunk)
            cleaned_content = self._clean_markdown(raw_content)
            if len(cleaned_content) < 30:
                continue
            if self._is_near_duplicate(cleaned_content, selected_contents):
                continue
            if not self._passes_diversity_limits(
                chunk=chunk,
                header_counts=header_counts,
                doc_counts=doc_counts,
                doc_filter=doc_id,
                limit=limit,
            ):
                continue

            seen_parent_ids.add(parent_id)
            selected_contents.append(cleaned_content)
            header_key = self._normalize_header_key(chunk.metadata.parent_header)
            header_counts[header_key] = header_counts.get(header_key, 0) + 1
            doc_key = self._clean_markdown(chunk.metadata.doc_id).lower()
            doc_counts[doc_key] = doc_counts.get(doc_key, 0) + 1
            retrieved_chunks.append(
                RetrievedChunk(
                    source=self._clean_markdown(chunk.metadata.parent_header),
                    doc_id=self._clean_markdown(chunk.metadata.doc_id),
                    content=cleaned_content,
                    chunk_type=chunk.metadata.chunk_type,
                    content_role=str(chunk.metadata.extra.get("content_role", chunk.metadata.chunk_type)),
                    page_number=chunk.metadata.page_number,
                    local_file=str(chunk.metadata.extra.get("local_file", "")),
                )
            )
            if len(retrieved_chunks) >= limit:
                break
        return retrieved_chunks[:limit]

    @staticmethod
    def serialize_for_prompt(chunks: list[RetrievedChunk]) -> str:
        return "\n\n".join(
            f"Source: {chunk.source} | Document: {chunk.doc_id}\n{chunk.content}"
            for chunk in chunks
        )

    def _rank_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        return sorted(chunks, key=self._chunk_priority, reverse=True)

    def _passes_diversity_limits(
        self,
        chunk: Chunk,
        header_counts: dict[str, int],
        doc_counts: dict[str, int],
        doc_filter: str | None,
        limit: int,
    ) -> bool:
        header_key = self._normalize_header_key(chunk.metadata.parent_header)
        if header_counts.get(header_key, 0) >= self._max_chunks_for_header(header_key):
            return False

        if doc_filter is not None:
            return True

        doc_key = self._clean_markdown(chunk.metadata.doc_id).lower()
        return doc_counts.get(doc_key, 0) < self._max_chunks_for_doc(limit)

    def _select_candidate_chunks(
        self,
        query: str,
        initial_chunks: list[Chunk],
        candidate_limit: int,
    ) -> list[Chunk]:
        if self._re_ranker is None:
            return initial_chunks[:candidate_limit]

        try:
            return self._re_ranker.rank(query=query, chunks=initial_chunks, top_n=candidate_limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Re-ranker failed; falling back to vector order: %s", exc)
            return initial_chunks[:candidate_limit]

    def _filter_chunks(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        query_prefers_tables = self._query_prefers_tables(query)
        filtered: list[Chunk] = []
        for chunk in chunks:
            content_role = str(chunk.metadata.extra.get("content_role", chunk.metadata.chunk_type))
            section_role = str(chunk.metadata.extra.get("section_role", "body"))
            parent_header = self._clean_markdown(chunk.metadata.parent_header)
            parent_content = str(chunk.metadata.extra.get("parent_content", chunk.content))
            normalized_parent = self._clean_markdown(parent_content)

            if self._is_excluded_section(section_role=section_role, content_role=content_role, parent_header=parent_header):
                continue
            if self._looks_like_low_value_content(normalized_parent):
                continue
            if content_role == "table" and not (self._include_tables or query_prefers_tables):
                continue
            filtered.append(chunk)
        return filtered

    def _select_return_content(self, chunk: Chunk) -> str:
        content_role = str(chunk.metadata.extra.get("content_role", chunk.metadata.chunk_type))
        if content_role != "child":
            return str(chunk.metadata.extra.get("parent_content", chunk.content))

        parent_sentences_raw = chunk.metadata.extra.get("parent_sentences")
        if not isinstance(parent_sentences_raw, list) or not parent_sentences_raw:
            return str(chunk.metadata.extra.get("parent_content", chunk.content))

        parent_sentences = [str(sentence).strip() for sentence in parent_sentences_raw if str(sentence).strip()]
        if not parent_sentences:
            return str(chunk.metadata.extra.get("parent_content", chunk.content))

        start = int(chunk.metadata.extra.get("child_sentence_start", 0))
        end = int(chunk.metadata.extra.get("child_sentence_end", max(1, start + 1)))
        window_start = max(0, start - 1)
        window_end = min(len(parent_sentences), end + 1)
        return " ".join(parent_sentences[window_start:window_end]).strip()

    @staticmethod
    def _query_prefers_tables(query: str) -> bool:
        normalized = query.lower()
        table_signals = ("table", "compare", "comparison", "value", "values", "sensitivity", "specificity", "odds ratio", "ct-value", "ct value", "assay panel")
        return any(signal in normalized for signal in table_signals)

    @staticmethod
    def _is_excluded_section(section_role: str, content_role: str, parent_header: str) -> bool:
        normalized_header = parent_header.lower()
        if content_role == "reference" or section_role == "references":
            return True
        if section_role == "front_matter":
            return True
        if normalized_header in {"unknown", ""}:
            return True
        if any(token in normalized_header for token in ("reference", "bibliograph", "acknowledg", "funding", "conflict", "competing interest", "author contribution", "data availability", "copyright")):
            return True
        return False

    @staticmethod
    def _looks_like_low_value_content(text: str) -> bool:
        normalized = " ".join(text.lower().split())
        if len(normalized) < 30:
            return True
        patterns = (
            r"\bet al\.",
            r"\bdoi[:/]",
            r"\bcopyright\b",
            r"\bcompeting interests?\b",
            r"\bfunding\b",
            r"\bdata availability\b",
            r"\bcorrespond(?:ence|ing)\b",
        )
        score = sum(1 for pattern in patterns if re.search(pattern, normalized))
        return score >= 2

    def _chunk_priority(self, chunk: Chunk) -> tuple[int, int]:
        header = self._clean_markdown(chunk.metadata.parent_header).lower()
        content_role = str(chunk.metadata.extra.get("content_role", chunk.metadata.chunk_type))
        header_bonus = 0
        if "conclusion" in header:
            header_bonus += 4
        if "result" in header:
            header_bonus += 3
        if "discussion" in header:
            header_bonus += 3
        if "abstract" in header:
            header_bonus -= 1
        if "method" in header:
            header_bonus -= 1
        role_bonus = 0
        if content_role == "table":
            role_bonus -= 2
        if content_role == "child":
            role_bonus += 1
        return (header_bonus, role_bonus)

    @staticmethod
    def _normalize_header_key(header: str) -> str:
        normalized = RetrievalService._clean_markdown(header).lower()
        return normalized or "unknown"

    @staticmethod
    def _max_chunks_for_header(header_key: str) -> int:
        if any(token in header_key for token in ("discussion", "result", "conclusion")):
            return 2
        if any(token in header_key for token in ("abstract", "introduction", "method")):
            return 1
        return 1

    @staticmethod
    def _max_chunks_for_doc(limit: int) -> int:
        return max(1, min(2, limit))

    @staticmethod
    def _is_near_duplicate(candidate: str, selected_contents: list[str]) -> bool:
        candidate_tokens = RetrievalService._tokenize(candidate)
        if not candidate_tokens:
            return False
        for selected in selected_contents:
            selected_tokens = RetrievalService._tokenize(selected)
            if not selected_tokens:
                continue
            overlap = len(candidate_tokens & selected_tokens) / max(1, min(len(candidate_tokens), len(selected_tokens)))
            if overlap > 0.9:
                return True
        return False

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}

    @staticmethod
    def _clean_markdown(text: str) -> str:
        cleaned = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"<[^>]*>\s*</[^>]*>", "", cleaned)
        cleaned = re.sub(r"<span[^>]*></span>", "", cleaned)
        cleaned = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "",
            cleaned,
        )
        cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", cleaned)
        cleaned = re.sub(r"\[([^\]]+)\]\(#page-[^)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"[*_`]+", "", cleaned)
        cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
        return cleaned.strip()
