from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import re

from src.app.ports.re_ranker_port import ReRankerPort
from src.app.ports.repositories.vector_repository import (
    MetadataFilter,
    VectorRepositoryPort,
    VectorSearchFilters,
)
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
        initial_limit = self._initial_search_limit(query=query, doc_filter=doc_id, limit=limit)
        initial_chunks = self._repo.search(
            query_vector,
            doc_id=doc_id,
            limit=initial_limit,
            filters=self._build_search_filters(query=query, doc_id=doc_id),
        )
        filtered_initial_chunks = self._filter_chunks(query=query, chunks=initial_chunks)
        if doc_id is not None:
            filtered_initial_chunks = self._suppress_metadata_fallback(query=query, chunks=filtered_initial_chunks)
        candidate_limit = max(limit * 6, 30) if self._query_prefers_tables(query) else max(limit * 4, 20)
        chunks = self._select_candidate_chunks(query=query, initial_chunks=filtered_initial_chunks, candidate_limit=candidate_limit)
        chunks = self._maybe_lock_contrastive_single_doc_query(query=query, doc_id=doc_id, chunks=chunks)
        ranked_chunks = self._rank_chunks(query=query, chunks=chunks)

        retrieved_chunks: list[RetrievedChunk] = []
        seen_parent_ids: set[str] = set()
        selected_contents: list[str] = []
        header_counts: dict[str, int] = {}
        doc_counts: dict[str, int] = {}
        docs_with_selected_body_sections: set[str] = set()
        max_selected_title_overlap = 0
        for chunk in ranked_chunks:
            parent_id = str(chunk.metadata.extra.get("parent_id", chunk.id))
            if parent_id in seen_parent_ids:
                continue

            if self._should_skip_new_document_for_query(
                query=query,
                chunk=chunk,
                doc_filter=doc_id,
                doc_counts=doc_counts,
                max_selected_title_overlap=max_selected_title_overlap,
            ):
                continue
            if self._should_skip_metadata_for_selected_doc(
                query=query,
                chunk=chunk,
                doc_filter=doc_id,
                docs_with_selected_body_sections=docs_with_selected_body_sections,
            ):
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
                selected_count=len(retrieved_chunks),
            ):
                continue

            seen_parent_ids.add(parent_id)
            selected_contents.append(cleaned_content)
            display_header = self._header_for_display(chunk)
            header_key = self._normalize_header_key(display_header)
            header_counts[header_key] = header_counts.get(header_key, 0) + 1
            doc_key = self._clean_markdown(chunk.metadata.doc_id).lower()
            doc_counts[doc_key] = doc_counts.get(doc_key, 0) + 1
            if not self._is_metadata_like_header(self._header_for_ranking(chunk).lower()):
                docs_with_selected_body_sections.add(doc_key)
            max_selected_title_overlap = max(
                max_selected_title_overlap,
                self._doc_title_overlap(query=query, chunk=chunk),
            )
            retrieved_chunks.append(
                RetrievedChunk(
                    source=display_header,
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

    def _rank_chunks(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        docs_with_body_sections = {
            self._clean_markdown(chunk.metadata.doc_id).lower()
            for chunk in chunks
            if not self._is_metadata_like_header(self._header_for_ranking(chunk).lower())
        }
        return sorted(
            chunks,
            key=lambda chunk: self._chunk_priority(
                query=query,
                chunk=chunk,
                docs_with_body_sections=docs_with_body_sections,
            ),
            reverse=True,
        )

    def _passes_diversity_limits(
        self,
        chunk: Chunk,
        header_counts: dict[str, int],
        doc_counts: dict[str, int],
        doc_filter: str | None,
        limit: int,
        selected_count: int,
    ) -> bool:
        header_key = self._normalize_header_key(self._header_for_display(chunk))
        if header_counts.get(header_key, 0) >= self._max_chunks_for_header(header_key):
            return False

        if doc_filter is not None:
            return True

        doc_key = self._clean_markdown(chunk.metadata.doc_id).lower()
        return doc_counts.get(doc_key, 0) < self._max_chunks_for_doc(limit=limit, selected_count=selected_count)

    def _should_skip_new_document_for_query(
        self,
        query: str,
        chunk: Chunk,
        doc_filter: str | None,
        doc_counts: dict[str, int],
        max_selected_title_overlap: int,
    ) -> bool:
        if doc_filter is not None or not doc_counts:
            return False
        if not self._query_prefers_single_document_target(query):
            return self._should_skip_zero_title_overlap_doc(
                query=query,
                chunk=chunk,
                max_selected_title_overlap=max_selected_title_overlap,
            )

        doc_key = self._clean_markdown(chunk.metadata.doc_id).lower()
        return doc_key not in doc_counts

    def _should_skip_zero_title_overlap_doc(
        self,
        query: str,
        chunk: Chunk,
        max_selected_title_overlap: int,
    ) -> bool:
        if max_selected_title_overlap <= 0:
            return False
        if not self._query_prefers_cross_document_titles(query):
            return False

        return self._doc_title_overlap(query=query, chunk=chunk) <= 0

    def _should_skip_metadata_for_selected_doc(
        self,
        query: str,
        chunk: Chunk,
        doc_filter: str | None,
        docs_with_selected_body_sections: set[str],
    ) -> bool:
        if doc_filter is not None or self._query_prefers_metadata(query):
            return False
        if not self._is_metadata_like_header(self._header_for_ranking(chunk).lower()):
            return False

        doc_key = self._clean_markdown(chunk.metadata.doc_id).lower()
        return doc_key in docs_with_selected_body_sections

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

    def _maybe_lock_contrastive_single_doc_query(
        self,
        query: str,
        doc_id: str | None,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        if doc_id is not None or not self._query_uses_contrastive_stewardship_doc_lock(query):
            return chunks
        if not chunks:
            return chunks

        locked_doc = self._best_document_for_contrastive_stewardship_query(query=query, chunks=chunks)
        if locked_doc is None:
            return chunks

        locked_doc_key = locked_doc.lower()
        locked_chunks = [
            chunk
            for chunk in chunks
            if self._clean_markdown(chunk.metadata.doc_id).lower() == locked_doc_key
        ]
        return locked_chunks or chunks

    def _filter_chunks(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        query_prefers_tables = self._query_prefers_tables(query)
        query_requires_tables = self._query_requires_tables(query)
        query_requires_metric_tables = self._query_requires_metric_tables(query)
        filtered: list[Chunk] = []
        for chunk in chunks:
            content_role = str(chunk.metadata.extra.get("content_role", chunk.metadata.chunk_type))
            parent_content = str(chunk.metadata.extra.get("parent_content", chunk.content))
            normalized_parent = self._clean_markdown(parent_content)

            if self._looks_like_low_value_content(normalized_parent):
                continue
            if (
                not query_prefers_tables
                and content_role != "table"
                and self._looks_like_embedded_markdown_table(parent_content)
            ):
                continue
            if (
                query_requires_tables
                and content_role != "table"
                and self._chunk_supports_explicit_table_query(chunk)
                and not self._linked_table_reference_supports_query(query=query, chunk=chunk)
            ):
                continue
            if query_requires_metric_tables and content_role == "table" and not self._table_matches_metric_query(chunk):
                continue
            filtered.append(chunk)
        return filtered

    def _build_search_filters(self, query: str, doc_id: str | None) -> VectorSearchFilters:
        must_not = [
            MetadataFilter(key="content_role", values=("reference", "front_matter")),
            MetadataFilter(key="section_role", values=("references", "front_matter", "unknown")),
            MetadataFilter(key="is_low_value", value=True),
        ]
        should: list[MetadataFilter] = []
        must: list[MetadataFilter] = []

        if not (self._include_tables or self._query_prefers_tables(query)):
            must_not.append(MetadataFilter(key="content_role", value="table"))

        if self._query_requires_metric_tables(query):
            must.append(MetadataFilter(key="content_role", value="table"))
        elif self._query_requires_tables(query):
            should.append(MetadataFilter(key="content_role", value="table"))
            referenced_table_indices = self._query_table_indices(query)
            if referenced_table_indices:
                should.append(
                    MetadataFilter(
                        key="referenced_table_indices",
                        values=tuple(referenced_table_indices),
                    )
                )
            else:
                should.append(MetadataFilter(key="has_table_reference", value=True))

        return VectorSearchFilters(
            doc_id=doc_id,
            must=tuple(must),
            must_not=tuple(must_not),
            should=tuple(should),
            minimum_should_match=1,
        )

    def _suppress_metadata_fallback(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if self._query_prefers_metadata(query):
            return chunks

        has_body_section = any(
            not self._is_metadata_like_header(self._header_for_ranking(chunk).lower())
            for chunk in chunks
        )
        if not has_body_section:
            return chunks

        body_chunks = [
            chunk for chunk in chunks if not self._is_metadata_like_header(self._header_for_ranking(chunk).lower())
        ]
        return body_chunks or chunks

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
    def _query_requires_tables(query: str) -> bool:
        normalized = query.lower()
        return "table" in normalized or "tabular" in normalized

    @staticmethod
    def _query_requires_metric_tables(query: str) -> bool:
        normalized = query.lower()
        if not RetrievalService._query_requires_tables(query):
            return False
        metric_signals = ("sensitivity", "specificity", "accuracy", "diagnostic accuracy")
        return any(signal in normalized for signal in metric_signals)

    @staticmethod
    def _query_table_indices(query: str) -> list[int]:
        return sorted(
            {
                int(match.group(1))
                for match in re.finditer(r"\btable\s+(\d+)\b", query, flags=re.IGNORECASE)
            }
        )

    def _linked_table_reference_supports_query(self, query: str, chunk: Chunk) -> bool:
        referenced_table_indices = chunk.metadata.extra.get("referenced_table_indices")
        if not isinstance(referenced_table_indices, list) or not referenced_table_indices:
            return False

        query_table_indices = self._query_table_indices(query)
        if query_table_indices:
            return any(index in referenced_table_indices for index in query_table_indices)

        query_tokens = self._table_reference_query_tokens(query)
        if not query_tokens:
            return True

        text = self._clean_markdown(str(chunk.metadata.extra.get("parent_content", chunk.content))).lower()
        text_tokens = self._tokenize(text)
        overlap = sum(1 for token in query_tokens if token in text_tokens)
        return overlap >= min(3, len(query_tokens))

    @staticmethod
    def _table_reference_query_tokens(query: str) -> set[str]:
        generic_tokens = {
            "which",
            "indexed",
            "paper",
            "papers",
            "study",
            "studies",
            "document",
            "documents",
            "contains",
            "contain",
            "table",
            "tabular",
            "results",
            "result",
            "data",
            "reported",
            "report",
            "findings",
            "finds",
            "obtained",
        }
        return {
            token
            for token in RetrievalService._query_tokens(query)
            if token not in generic_tokens
        }

    @staticmethod
    def _table_matches_metric_query(chunk: Chunk) -> bool:
        contains_metric_values = chunk.metadata.extra.get("contains_metric_values")
        if isinstance(contains_metric_values, bool):
            return contains_metric_values

        table_semantics = chunk.metadata.extra.get("table_semantics")
        if isinstance(table_semantics, list) and any(str(item).strip().lower() == "metric" for item in table_semantics):
            return True

        text_parts = [
            chunk.content,
            str(chunk.metadata.extra.get("parent_content", "")),
            str(chunk.metadata.parent_header),
            str(chunk.metadata.extra.get("normalized_parent_header", "")),
        ]
        normalized = RetrievalService._clean_markdown(" ".join(part for part in text_parts if part)).lower()
        metric_patterns = (
            r"\bsensitivity\b",
            r"\bspecificity\b",
            r"\baccuracy\b",
            r"\bdiagnostic accuracy\b",
            r"\bagreement\b",
            r"\bdetection rate\b",
            r"\bfalse positive\b",
            r"\bfalse negative\b",
            r"\bppv\b",
            r"\bnpv\b",
            r"\bauc\b",
            r"\broc\b",
            r"\blod\b",
            r"\blimit of detection\b",
        )
        if not any(re.search(pattern, normalized) for pattern in metric_patterns):
            return False

        quantitative_signals = (
            r"\b\d+(?:\.\d+)?\s*%",
            r"\b\d+\.\d+\b",
            r"\bpositive predictive\b",
            r"\bnegative predictive\b",
            r"\bppv\b",
            r"\bnpv\b",
            r"\bauc\b",
            r"\broc\b",
            r"\blod\b",
            r"\blimit of detection\b",
        )
        return any(re.search(pattern, normalized) for pattern in quantitative_signals)

    @staticmethod
    def _looks_like_embedded_markdown_table(text: str) -> bool:
        normalized = text.strip()
        if normalized.count("|") < 6:
            return False
        if re.search(r"\|\s*-{3,}", normalized):
            return True
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        tableish_lines = sum(1 for line in lines if line.count("|") >= 2)
        return tableish_lines >= 3

    @staticmethod
    def _chunk_supports_explicit_table_query(chunk: Chunk) -> bool:
        if str(chunk.metadata.extra.get("content_role", chunk.metadata.chunk_type)) == "table":
            return True
        referenced_table_indices = chunk.metadata.extra.get("referenced_table_indices")
        if not isinstance(referenced_table_indices, list):
            return False
        return any(str(item).strip().isdigit() for item in referenced_table_indices)

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

    def _chunk_priority(
        self,
        query: str,
        chunk: Chunk,
        docs_with_body_sections: set[str],
    ) -> tuple[int, int, int]:
        query_profile = self._query_section_profile(query)
        query_tokens = self._query_tokens(query)
        header = self._header_for_ranking(chunk).lower()
        content_role = str(chunk.metadata.extra.get("content_role", chunk.metadata.chunk_type))
        doc_key = self._clean_markdown(chunk.metadata.doc_id).lower()
        header_bonus = 0
        if "document metadata/abstract" in header:
            header_bonus -= 3
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
        if "introduction" in header:
            header_bonus -= 1

        for section, bonus in query_profile.items():
            if section in header:
                header_bonus += bonus

        if self._is_metadata_like_header(header) and doc_key in docs_with_body_sections:
            header_bonus -= 5

        role_bonus = 0
        if content_role == "table":
            role_bonus += 8 if self._query_prefers_tables(query) else -2
        if self._query_requires_tables(query) and self._chunk_supports_explicit_table_query(chunk):
            role_bonus += 6
        if content_role == "child":
            role_bonus += 1
        if content_role == "table" and not self._query_prefers_tables(query):
            role_bonus -= 1
        if self._query_prefers_tables(query) and content_role != "table":
            role_bonus -= 1
        lexical_bonus = self._lexical_bonus(query=query, query_tokens=query_tokens, chunk=chunk)
        contrast_bonus = self._contrastive_query_bonus(query=query, chunk=chunk)
        body_bonus = 0 if "document metadata/abstract" in header else 1
        return (header_bonus, lexical_bonus + contrast_bonus, role_bonus + body_bonus)

    def _best_document_for_contrastive_stewardship_query(
        self,
        query: str,
        chunks: list[Chunk],
    ) -> str | None:
        docs_with_body_sections = {
            self._clean_markdown(chunk.metadata.doc_id).lower()
            for chunk in chunks
            if not self._is_metadata_like_header(self._header_for_ranking(chunk).lower())
        }
        doc_scores: dict[str, tuple[int, int, int, tuple[int, int, int]]] = {}
        doc_ids: dict[str, str] = {}

        for chunk in chunks:
            doc_key = self._clean_markdown(chunk.metadata.doc_id).lower()
            doc_ids.setdefault(doc_key, self._clean_markdown(chunk.metadata.doc_id))
            contrast_bonus = self._contrastive_query_bonus(query=query, chunk=chunk)
            title_overlap = self._doc_title_overlap(query=query, chunk=chunk)
            chunk_priority = self._chunk_priority(
                query=query,
                chunk=chunk,
                docs_with_body_sections=docs_with_body_sections,
            )
            current = doc_scores.get(doc_key)
            next_score = (
                max(current[0], contrast_bonus) if current else contrast_bonus,
                (current[1] if current else 0) + max(0, contrast_bonus),
                max(current[2], title_overlap) if current else title_overlap,
                max(current[3], chunk_priority) if current else chunk_priority,
            )
            doc_scores[doc_key] = next_score

        if not doc_scores:
            return None

        best_doc_key = max(doc_scores, key=doc_scores.__getitem__)
        return doc_ids[best_doc_key]

    @staticmethod
    def _query_section_profile(query: str) -> dict[str, int]:
        normalized = query.lower()
        bonuses: dict[str, int] = {}

        def add_bonus(sections: tuple[str, ...], value: int) -> None:
            for section in sections:
                bonuses[section] = bonuses.get(section, 0) + value

        if (
            RetrievalService._query_targets_stewardship_process(query)
            and RetrievalService._query_contrasts_against_trial_or_platform(query)
        ):
            add_bonus(("introduction", "discussion", "summary", "conclusion"), 4)
            add_bonus(("result", "results", "methods", "document metadata/abstract", "abstract"), -4)

        if any(token in normalized for token in ("performance", "sensitivity", "specificity", "findings", "result", "outcome", "outcomes", "data", "diagnostic performance")):
            add_bonus(("result", "results"), 3)
            add_bonus(("conclusion",), -2)
        if any(token in normalized for token in ("biomarker", "biomarkers", "differentiat", "marker")):
            add_bonus(("result", "results", "discussion"), 2)
            add_bonus(("conclusion",), -3)
        if any(token in normalized for token in ("optimization", "optimiz", "method", "methods", "experimental", "assay", "protocol")):
            add_bonus(("method", "methods", "materials and methods"), 4)
            add_bonus(("result", "results"), 1)
            add_bonus(("conclusion",), -2)
        if any(token in normalized for token in ("compare", "compares", "comparing", "versus", " vs ", "with and without")):
            add_bonus(("result", "results"), 4)
            add_bonus(("discussion",), -1)
        if any(token in normalized for token in ("limitation", "limitations", "caveat", "caveats", "implication", "implications", "conclusion", "conclusions", "usefulness", "clinical usefulness")):
            add_bonus(("discussion", "conclusion"), 3)
        if any(token in normalized for token in ("review", "overview", "what does the review say", "approaches", "arguments")):
            add_bonus(("introduction", "discussion", "summary", "abstract"), 2)
            add_bonus(("conclusion",), -6)
        if any(token in normalized for token in ("stewardship", "optimiz", "hospital setting", "blood culture use")):
            add_bonus(("discussion", "introduction", "summary"), 4)
            add_bonus(("document metadata/abstract", "abstract"), -7)
            add_bonus(("conclusion",), -2)
        if RetrievalService._query_prefers_tables(query):
            add_bonus(("result", "results"), 2)
        return bonuses

    @staticmethod
    def _initial_search_limit(query: str, doc_filter: str | None, limit: int) -> int:
        if doc_filter is not None:
            return max(limit * 8, 40)
        if RetrievalService._query_prefers_tables(query):
            return max(limit * 12, 60)
        if RetrievalService._query_prefers_cross_document_titles(query):
            return max(limit * 10, 50)
        return max(limit * 8, 40)

    @staticmethod
    def _query_prefers_cross_document_titles(query: str) -> bool:
        normalized = query.lower()
        signals = ("which documents", "which papers", "across the indexed studies", "across studies", "across the corpus")
        return any(signal in normalized for signal in signals)

    @staticmethod
    def _query_prefers_single_document_target(query: str) -> bool:
        normalized = query.lower()
        plural_signals = ("which papers", "which documents", "which studies")
        if any(signal in normalized for signal in plural_signals):
            return False

        singular_signals = (
            "which paper",
            "which indexed paper",
            "which document",
            "which indexed document",
            "which study",
            "which indexed study",
            "which trial",
            "which indexed trial",
            "which randomized trial",
            "which indexed randomized trial",
        )
        return any(signal in normalized for signal in singular_signals)

    @staticmethod
    def _query_uses_contrastive_stewardship_doc_lock(query: str) -> bool:
        return (
            RetrievalService._query_prefers_single_document_target(query)
            and RetrievalService._query_targets_stewardship_process(query)
            and RetrievalService._query_contrasts_against_trial_or_platform(query)
        )

    @staticmethod
    def _query_prefers_metadata(query: str) -> bool:
        normalized = query.lower()
        signals = ("abstract", "summary", "overview", "opening summary")
        return any(signal in normalized for signal in signals)

    @staticmethod
    def _query_tokens(query: str) -> set[str]:
        stop_words = {
            "what", "which", "does", "the", "about", "were", "with", "from", "that", "this",
            "into", "across", "study", "studies", "paper", "papers", "documents", "document",
            "indexed", "reported", "report", "findings", "results", "result",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", query.lower())
            if len(token) > 3 and token not in stop_words
        }

    def _lexical_bonus(self, query: str, query_tokens: set[str], chunk: Chunk) -> int:
        if not query_tokens:
            return 0

        effective_query_tokens = self._effective_query_tokens_for_chunk_scoring(query=query, query_tokens=query_tokens)
        if not effective_query_tokens:
            return 0

        header_text = self._header_for_ranking(chunk).lower()
        doc_text = self._doc_text_for_ranking(chunk)
        content_text = self._clean_markdown(str(chunk.metadata.extra.get("parent_content", chunk.content))).lower()
        title_tokens = self._title_weighted_tokens(effective_query_tokens)

        doc_overlap = sum(1 for token in effective_query_tokens if token in doc_text)
        header_overlap = sum(1 for token in effective_query_tokens if token in header_text)
        content_overlap = sum(1 for token in effective_query_tokens if token in content_text)
        title_overlap = self._doc_title_overlap(query=query, chunk=chunk)

        lexical_bonus = min(4, content_overlap) + min(3, header_overlap * 2)
        if self._query_prefers_cross_document_titles(query):
            lexical_bonus += min(6, title_overlap * 4)
        else:
            lexical_bonus += min(2, doc_overlap)
        return lexical_bonus

    def _contrastive_query_bonus(self, query: str, chunk: Chunk) -> int:
        normalized = query.lower()
        if not self._query_targets_stewardship_process(query):
            return 0

        doc_text = self._doc_text_for_ranking(chunk)
        content_text = self._clean_markdown(str(chunk.metadata.extra.get("parent_content", chunk.content))).lower()
        combined_text = f"{doc_text} {content_text}"
        title_overlap = self._doc_title_overlap(query=query, chunk=chunk)

        process_markers = (
            "stewardship",
            "utilization",
            "optimiz",
            "ordering",
            "ordered",
            "collection",
            "collected",
            "draw blood",
            "blood culture use",
            "hospital setting",
            "preanalytical",
        )
        trial_markers = (
            "trial",
            "randomized",
            "patient outcome",
            "patient outcomes",
            "platform",
            "rapid multiplex",
            "stewardship support",
            "templated comments",
        )

        bonus = 0
        if any(marker in combined_text for marker in process_markers):
            bonus += 4
        if self._query_contrasts_against_trial_or_platform(query) and any(marker in combined_text for marker in trial_markers):
            bonus -= 10
        if self._query_contrasts_against_trial_or_platform(query):
            if title_overlap > 0:
                bonus += min(8, title_overlap * 4)
            elif self._query_prefers_single_document_target(query):
                bonus -= 4
        return bonus

    def _doc_title_overlap(self, query: str, chunk: Chunk) -> int:
        query_tokens = self._effective_query_tokens_for_chunk_scoring(
            query=query,
            query_tokens=self._query_tokens(query),
        )
        if not query_tokens:
            return 0

        title_tokens = self._title_weighted_tokens(query_tokens)
        if not title_tokens:
            return 0

        doc_text = self._doc_text_for_ranking(chunk)
        return sum(1 for token in title_tokens if token in doc_text)

    @staticmethod
    def _effective_query_tokens_for_chunk_scoring(query: str, query_tokens: set[str]) -> set[str]:
        if not (
            RetrievalService._query_targets_stewardship_process(query)
            and RetrievalService._query_contrasts_against_trial_or_platform(query)
        ):
            return query_tokens

        contrast_tokens = {
            "trial",
            "trials",
            "patient",
            "patients",
            "outcome",
            "outcomes",
            "platform",
            "platforms",
            "interventional",
            "intervention",
        }
        return {token for token in query_tokens if token not in contrast_tokens}

    @staticmethod
    def _query_targets_stewardship_process(query: str) -> bool:
        normalized = query.lower()
        process_signals = (
            "stewardship",
            "utilization",
            "hospital setting",
            "blood culture use",
            "ordering",
            "ordered",
            "collection",
            "collected",
            "draw blood",
            "improving when and how blood cultures",
        )
        return any(signal in normalized for signal in process_signals)

    @staticmethod
    def _query_contrasts_against_trial_or_platform(query: str) -> bool:
        normalized = query.lower()
        contrast_signals = (
            "instead of",
            "rather than",
            "not about",
            "not just",
        )
        target_signals = (
            "trial",
            "patient-outcomes",
            "patient outcomes",
            "platform",
        )
        return any(signal in normalized for signal in contrast_signals) and any(
            signal in normalized for signal in target_signals
        )

    @staticmethod
    def _title_weighted_tokens(query_tokens: set[str]) -> set[str]:
        generic_tokens = {
            "blood",
            "culture",
            "cultures",
            "diagnostic",
            "diagnosis",
            "sensitivity",
            "specificity",
            "performance",
            "tabular",
            "table",
            "rapid",
            "testing",
            "outcomes",
            "outcome",
            "findings",
            "papers",
            "documents",
        }
        return {token for token in query_tokens if token not in generic_tokens}

    @staticmethod
    def _is_metadata_like_header(header: str) -> bool:
        return any(token in header for token in ("document metadata/abstract", "abstract", "summary"))

    @staticmethod
    def _doc_text_for_ranking(chunk: Chunk) -> str:
        doc_parts = [
            str(chunk.metadata.doc_id),
            str(chunk.metadata.extra.get("local_file", "")),
            str(chunk.metadata.extra.get("source_file", "")),
        ]
        normalized = " ".join(part.replace("-", " ").replace("_", " ") for part in doc_parts if part)
        return RetrievalService._clean_markdown(normalized).lower()

    @staticmethod
    def _normalize_header_key(header: str) -> str:
        normalized = RetrievalService._clean_markdown(header).lower()
        return normalized or "unknown"

    def _header_for_display(self, chunk: Chunk) -> str:
        normalized_header = str(chunk.metadata.extra.get("normalized_parent_header", chunk.metadata.parent_header))
        return self._clean_markdown(normalized_header)

    def _header_for_ranking(self, chunk: Chunk) -> str:
        normalized_header = str(chunk.metadata.extra.get("normalized_parent_header", chunk.metadata.parent_header))
        return self._clean_markdown(normalized_header)

    @staticmethod
    def _max_chunks_for_header(header_key: str) -> int:
        if any(token in header_key for token in ("discussion", "result", "conclusion")):
            return 2
        if any(token in header_key for token in ("abstract", "introduction", "method")):
            return 1
        return 1

    @staticmethod
    def _max_chunks_for_doc(limit: int, selected_count: int) -> int:
        if limit <= 3:
            return 1
        if selected_count < 3:
            return 1
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
