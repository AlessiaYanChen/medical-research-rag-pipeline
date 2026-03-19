from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from src.domain.models.chunk import Chunk, ChunkMetadata


class UnifiedChunker:
    """Chunk mixed markdown + table artifacts into a unified sequence of Chunk models."""

    DEFAULT_OPENING_HEADER = "Document Metadata/Abstract"
    DEFAULT_OPENING_HEADER_ROLE = "opening_metadata"

    def __init__(
        self,
        max_chars: int = 900,
        overlap_paragraphs: int = 1,
        child_sentence_window: int = 2,
        child_sentence_overlap: int = 1,
    ) -> None:
        self.max_chars = max_chars
        self.overlap_paragraphs = max(0, overlap_paragraphs)
        self.child_sentence_window = max(1, child_sentence_window)
        self.child_sentence_overlap = max(0, min(child_sentence_overlap, self.child_sentence_window - 1))

    def chunk_document(
        self,
        doc_id: str,
        source_file: str,
        markdown_text: str,
        tables: list[dict[str, Any]] | None = None,
        document_path: str | Path | None = None,
    ) -> list[Chunk]:
        if tables is not None:
            table_artifacts = tables
        elif document_path is not None:
            table_artifacts = self.load_table_artifacts(document_path)
        else:
            table_artifacts = []

        loaded_table_artifacts_count = len(table_artifacts)
        blocks = self._split_blocks(markdown_text)

        chunks: list[Chunk] = []
        parent_counter = 0
        active_header = self.DEFAULT_OPENING_HEADER
        active_original_header = self.DEFAULT_OPENING_HEADER
        active_header_role = self.DEFAULT_OPENING_HEADER_ROLE
        active_section_role = self._classify_section_role(active_header)
        active_structural_level = 1
        active_structural_header = self.DEFAULT_OPENING_HEADER
        table_index = 0
        pending_paragraphs: list[str] = []
        pending_header = active_header
        pending_original_header = active_original_header
        pending_header_role = active_header_role
        first_header_seen = False

        for block in blocks:
            stripped = block.strip()
            if not stripped:
                continue

            header = self._extract_header(stripped)
            if header is not None:
                built_chunks, parent_counter = self._build_text_chunks(
                    doc_id=doc_id,
                    parent_header=pending_header,
                    original_parent_header=pending_original_header,
                    header_role=pending_header_role,
                    paragraphs=pending_paragraphs,
                    parent_id_start=parent_counter + 1,
                )
                chunks.extend(built_chunks)
                pending_paragraphs = []
                resolved_header = self._resolve_header_context(
                    header=header["text"],
                    header_level=header["level"],
                    active_structural_header=active_structural_header,
                    active_structural_level=active_structural_level,
                    is_first_header=not first_header_seen,
                )
                active_header = resolved_header["normalized_header"]
                active_original_header = resolved_header["original_header"]
                active_header_role = resolved_header["header_role"]
                active_structural_header = resolved_header["active_structural_header"]
                active_structural_level = resolved_header["active_structural_level"]
                active_section_role = self._classify_section_role(active_header)
                pending_header = active_header
                pending_original_header = active_original_header
                pending_header_role = active_header_role
                first_header_seen = True
                continue

            if self._is_table_block(stripped):
                built_chunks, parent_counter = self._build_text_chunks(
                    doc_id=doc_id,
                    parent_header=pending_header,
                    original_parent_header=pending_original_header,
                    header_role=pending_header_role,
                    paragraphs=pending_paragraphs,
                    parent_id_start=parent_counter + 1,
                )
                chunks.extend(built_chunks)
                pending_paragraphs = []
                table_index += 1
                artifact = (
                    table_artifacts[table_index - 1]
                    if table_index - 1 < len(table_artifacts)
                    else {"table_markdown": stripped}
                )
                chunks.append(
                    self._build_table_chunk(
                        chunk_id=self._make_table_chunk_id(doc_id, table_index),
                        doc_id=doc_id,
                        source_file=source_file,
                        table_index=table_index,
                        parent_header=active_header,
                        original_parent_header=active_original_header,
                        header_role=active_header_role,
                        table_artifact=artifact,
                        section_role=active_section_role,
                    )
                )
                continue

            pending_paragraphs.append(stripped)

        # Ensure discovered artifacts are kept as atomic units even if markdown table
        # blocks are missing or fewer than extracted artifacts.
        while table_index < len(table_artifacts):
            table_index += 1
            chunks.append(
                self._build_table_chunk(
                    chunk_id=self._make_table_chunk_id(doc_id, table_index),
                    doc_id=doc_id,
                    source_file=source_file,
                    table_index=table_index,
                    parent_header=active_header,
                    original_parent_header=active_original_header,
                    header_role=active_header_role,
                    table_artifact=table_artifacts[table_index - 1],
                    section_role=active_section_role,
                )
            )

        built_chunks, parent_counter = self._build_text_chunks(
            doc_id=doc_id,
            parent_header=pending_header,
            original_parent_header=pending_original_header,
            header_role=pending_header_role,
            paragraphs=pending_paragraphs,
            parent_id_start=parent_counter + 1,
        )
        chunks.extend(built_chunks)

        if loaded_table_artifacts_count > 0:
            assert any(chunk.metadata.chunk_type == "table" for chunk in chunks), (
                "If table artifacts are loaded, table chunks must be present in final output."
            )

        return chunks

    @staticmethod
    def load_table_artifacts(document_path: str | Path) -> list[dict[str, Any]]:
        """
        Discover table artifacts as siblings of the currently processed document.

        Example:
        - data/marker_markdown/MyDoc.main_text.md
        - data/marker_markdown/MyDoc.table_01.json
        - data/marker_markdown/MyDoc.table_01.csv
        """
        doc_path = Path(document_path)
        artifact_dir = doc_path.parent

        base_stem = doc_path.stem
        if base_stem.endswith(".main_text"):
            base_stem = base_stem[: -len(".main_text")]

        artifacts_by_index: dict[int, dict[str, Any]] = {}

        for json_path in sorted(artifact_dir.glob(f"{base_stem}.table_*.json")):
            match = re.search(r"\.table_(\d+)\.json$", json_path.name)
            if not match:
                continue
            idx = int(match.group(1))
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            artifact = artifacts_by_index.setdefault(idx, {})
            if isinstance(payload, dict):
                if "rows" in payload:
                    artifact["rows"] = payload["rows"]
                if "page_number" in payload:
                    artifact["page_number"] = payload["page_number"]
            else:
                artifact["rows"] = payload

        for csv_path in sorted(artifact_dir.glob(f"{base_stem}.table_*.csv")):
            match = re.search(r"\.table_(\d+)\.csv$", csv_path.name)
            if not match:
                continue
            idx = int(match.group(1))
            artifact = artifacts_by_index.setdefault(idx, {})
            artifact["csv"] = csv_path.read_text(encoding="utf-8")

        return [artifacts_by_index[idx] for idx in sorted(artifacts_by_index)]

    @staticmethod
    def _split_blocks(markdown_text: str) -> list[str]:
        return re.split(r"\n\s*\n", markdown_text.strip())

    @staticmethod
    def _extract_header(block: str) -> dict[str, Any] | None:
        line = block.splitlines()[0].strip()
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if match:
            return {
                "level": len(match.group(1)),
                "text": match.group(2).strip(),
            }
        return None

    @classmethod
    def _normalize_header(cls, header: str, is_first_header: bool) -> str:
        cleaned = header.strip()
        if not cleaned:
            return cls.DEFAULT_OPENING_HEADER
        if is_first_header and not cls._looks_like_structural_header(cleaned):
            return cls.DEFAULT_OPENING_HEADER
        return cleaned

    @classmethod
    def _resolve_header_context(
        cls,
        header: str,
        header_level: int,
        active_structural_header: str,
        active_structural_level: int,
        is_first_header: bool,
    ) -> dict[str, Any]:
        cleaned = header.strip()
        if not cleaned:
            return {
                "normalized_header": cls.DEFAULT_OPENING_HEADER,
                "original_header": cls.DEFAULT_OPENING_HEADER,
                "header_role": cls.DEFAULT_OPENING_HEADER_ROLE,
                "active_structural_header": active_structural_header,
                "active_structural_level": active_structural_level,
                "is_structural": False,
            }

        if cls._looks_like_structural_header(cleaned):
            return {
                "normalized_header": cleaned,
                "original_header": cleaned,
                "header_role": "structural",
                "active_structural_header": cleaned,
                "active_structural_level": header_level,
                "is_structural": True,
            }

        if cls._looks_like_thematic_section_header(
            cleaned,
            is_first_header=is_first_header,
        ):
            normalized_header = cls._normalize_thematic_structural_header(
                header=cleaned,
                header_level=header_level,
                active_structural_header=active_structural_header,
                active_structural_level=active_structural_level,
            )
            return {
                "normalized_header": normalized_header,
                "original_header": cleaned,
                "header_role": "thematic_structural",
                "active_structural_header": normalized_header,
                "active_structural_level": header_level,
                "is_structural": True,
            }

        if is_first_header:
            return {
                "normalized_header": cls.DEFAULT_OPENING_HEADER,
                "original_header": cleaned,
                "header_role": cls._classify_non_structural_header(cleaned),
                "active_structural_header": active_structural_header,
                "active_structural_level": active_structural_level,
                "is_structural": False,
            }

        return {
            "normalized_header": active_structural_header or cls.DEFAULT_OPENING_HEADER,
            "original_header": cleaned,
            "header_role": cls._classify_non_structural_header(cleaned),
            "active_structural_header": active_structural_header or cls.DEFAULT_OPENING_HEADER,
            "active_structural_level": active_structural_level,
            "is_structural": False,
        }

    @staticmethod
    def _is_table_block(block: str) -> bool:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            return False
        if not all(line.count("|") >= 2 for line in lines):
            return False
        divider = lines[1]
        return bool(re.match(r"^\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$", divider))

    def _build_text_chunks(
        self,
        doc_id: str,
        parent_header: str,
        original_parent_header: str,
        header_role: str,
        paragraphs: list[str],
        parent_id_start: int,
    ) -> tuple[list[Chunk], int]:
        if not paragraphs:
            return [], parent_id_start - 1

        chunks: list[Chunk] = []
        start = 0
        total = len(paragraphs)
        step_floor = max(1, self.overlap_paragraphs)
        next_parent_id = parent_id_start

        while start < total:
            current: list[str] = []
            chars = 0
            idx = start
            while idx < total:
                paragraph = paragraphs[idx]
                candidate_size = chars + len(paragraph) + (2 if current else 0)
                if current and candidate_size > self.max_chars:
                    break
                current.append(paragraph)
                chars = candidate_size
                idx += 1

            parent_content = "\n\n".join(current)
            parent_id = self._make_parent_chunk_id(doc_id, next_parent_id)
            chunks.extend(
                self._build_child_text_chunks(
                    doc_id=doc_id,
                    parent_id=parent_id,
                    parent_content=parent_content,
                    parent_header=parent_header,
                    original_parent_header=original_parent_header,
                    header_role=header_role,
                    section_role=self._classify_section_role(parent_header),
                    content_role=self._classify_parent_content_role(parent_header, parent_content),
                )
            )
            next_parent_id += 1

            if idx >= total:
                break
            start = max(start + step_floor, idx - self.overlap_paragraphs)

        return chunks, next_parent_id - 1

    def _build_child_text_chunks(
        self,
        doc_id: str,
        parent_id: str,
        parent_content: str,
        parent_header: str,
        original_parent_header: str,
        header_role: str,
        section_role: str,
        content_role: str,
    ) -> list[Chunk]:
        sentences = self._split_sentences(parent_content)
        page_number = self._extract_page_number(parent_content)
        if not sentences:
            sentences = [parent_content]

        child_windows = self._build_child_sentence_windows(sentences)
        return [
            Chunk(
                id=self._make_child_chunk_id(parent_id, child_index + 1),
                content=child_window["content"],
                metadata=ChunkMetadata(
                    doc_id=doc_id,
                    chunk_type="text",
                    parent_header=parent_header,
                    page_number=page_number,
                    extra={
                        "content_role": content_role,
                        "section_role": section_role,
                        "original_parent_header": original_parent_header,
                        "normalized_parent_header": parent_header,
                        "header_role": header_role,
                        "is_low_value": header_role == "citation_like",
                        "parent_id": parent_id,
                        "parent_content": parent_content,
                        "parent_sentences": sentences,
                        "child_index": child_index + 1,
                        "child_sentence_start": child_window["start"],
                        "child_sentence_end": child_window["end"],
                    },
                ),
            )
            for child_index, child_window in enumerate(child_windows)
        ]

    def _build_table_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        source_file: str,
        table_index: int,
        parent_header: str,
        original_parent_header: str,
        header_role: str,
        table_artifact: dict[str, Any],
        section_role: str,
    ) -> Chunk:
        table_payload = self._table_payload(table_artifact)
        context_header = (
            f"Source File: {source_file} | "
            f"Table Index: {table_index} | "
            f"Section: {parent_header}"
        )
        content = f"{context_header}\n{table_payload}"

        page_number = table_artifact.get("page_number")
        if page_number is None:
            page_number = self._extract_page_number(table_payload)

        table_role = self._classify_table_content_role(parent_header=parent_header, table_payload=table_payload, section_role=section_role)
        return Chunk(
            id=chunk_id,
            content=content,
            metadata=ChunkMetadata(
                doc_id=doc_id,
                chunk_type="table",
                parent_header=parent_header,
                page_number=page_number,
                extra={
                    "content_role": table_role,
                    "section_role": section_role,
                    "original_parent_header": original_parent_header,
                    "normalized_parent_header": parent_header,
                    "header_role": header_role,
                    "is_low_value": header_role == "citation_like",
                },
            ),
        )

    @staticmethod
    def _table_payload(table_artifact: dict[str, Any]) -> str:
        if "csv" in table_artifact and table_artifact["csv"]:
            return str(table_artifact["csv"])
        if "rows" in table_artifact and table_artifact["rows"] is not None:
            return json.dumps(table_artifact["rows"], ensure_ascii=True)
        if "json_rows" in table_artifact and table_artifact["json_rows"] is not None:
            return json.dumps(table_artifact["json_rows"], ensure_ascii=True)
        if "table_markdown" in table_artifact and table_artifact["table_markdown"]:
            return str(table_artifact["table_markdown"])
        return json.dumps(table_artifact, ensure_ascii=True)

    @staticmethod
    def _extract_page_number(content: str) -> int | None:
        match = re.search(r"\bpage\s*(\d+)\b", content, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
            if sentence.strip()
        ]

    def _build_child_sentence_windows(self, sentences: list[str]) -> list[dict[str, Any]]:
        if len(sentences) <= self.child_sentence_window:
            return [
                {
                    "start": 0,
                    "end": len(sentences),
                    "content": " ".join(sentences).strip(),
                }
            ]

        windows: list[dict[str, Any]] = []
        step = max(1, self.child_sentence_window - self.child_sentence_overlap)
        for start in range(0, len(sentences), step):
            window = sentences[start : start + self.child_sentence_window]
            if not window:
                continue
            windows.append(
                {
                    "start": start,
                    "end": start + len(window),
                    "content": " ".join(window).strip(),
                }
            )
            if start + self.child_sentence_window >= len(sentences):
                break
        return windows

    @staticmethod
    def _make_parent_chunk_id(doc_id: str, sequence: int) -> str:
        return f"{doc_id}:P{sequence:05d}"

    @staticmethod
    def _make_child_chunk_id(parent_id: str, child_index: int) -> str:
        return f"{parent_id}:C{child_index:02d}"

    @staticmethod
    def _make_table_chunk_id(doc_id: str, table_index: int) -> str:
        return f"{doc_id}:T{table_index:05d}"

    @staticmethod
    def _looks_like_structural_header(header: str) -> bool:
        normalized = header.strip().lower()
        if normalized in {"unknown", ""}:
            return False
        known_tokens = (
            "abstract",
            "introduction",
            "background",
            "methods",
            "materials",
            "results",
            "discussion",
            "conclusion",
            "summary",
            "references",
            "bibliography",
            "acknowledg",
            "funding",
            "conflict",
            "author contribution",
            "data availability",
        )
        return any(token in normalized for token in known_tokens)

    @classmethod
    def _looks_like_thematic_section_header(cls, header: str, is_first_header: bool) -> bool:
        normalized = header.strip()
        normalized_lower = normalized.lower()
        if not normalized or is_first_header:
            return False
        if cls._classify_non_structural_header(normalized) == "citation_like":
            return False
        if any(
            token in normalized_lower
            for token in (
                "references",
                "bibliograph",
                "acknowledg",
                "funding",
                "conflict",
                "copyright",
                "data availability",
            )
        ):
            return False

        words = normalized.split()
        if normalized.isupper() and len(words) >= 4:
            return True
        if len(words) >= 5 and re.fullmatch(r"[A-Za-z0-9&/\-(),:; ]+", normalized):
            return True
        return False

    @classmethod
    def _normalize_thematic_structural_header(
        cls,
        header: str,
        header_level: int,
        active_structural_header: str,
        active_structural_level: int,
    ) -> str:
        normalized = header.strip().lower()
        if "summary" in normalized:
            return "Summary"
        if any(token in normalized for token in ("reference", "bibliograph")):
            return "References"

        if active_structural_header == cls.DEFAULT_OPENING_HEADER:
            return "Introduction"

        if header_level > active_structural_level:
            return active_structural_header

        if active_structural_header == "Introduction":
            return "Discussion"
        if active_structural_header == "Discussion":
            return "Conclusion"
        return active_structural_header

    @staticmethod
    def _classify_non_structural_header(header: str) -> str:
        normalized = header.strip()
        normalized_lower = normalized.lower()
        if not normalized:
            return "opening_metadata"
        if re.search(r"\b(editorial commentary|commentary by|clinical infectious diseases|doi[:/]|vol(?:ume)?\b|issue\b)\b", normalized_lower):
            return "citation_like"
        if len(normalized.split()) <= 4 and re.fullmatch(r"[A-Za-z0-9&/\- ]+", normalized):
            return "subsection"
        if normalized.isupper() and len(normalized.split()) <= 8:
            return "subsection"
        return "title_like"

    @staticmethod
    def _classify_section_role(header: str) -> str:
        normalized = header.strip().lower()
        if normalized in {"unknown", ""}:
            return "unknown"
        if any(token in normalized for token in ("reference", "bibliograph")):
            return "references"
        if any(token in normalized for token in ("funding", "acknowledg", "competing interest", "conflict of interest", "copyright", "author contribution", "data availability")):
            return "front_matter"
        if any(token in normalized for token in ("abstract", "introduction", "background", "methods", "materials", "results", "discussion", "conclusion")):
            return "body"
        return "body"

    def _classify_parent_content_role(self, parent_header: str, parent_content: str) -> str:
        section_role = self._classify_section_role(parent_header)
        if self._looks_like_reference_block(parent_content):
            return "reference"
        if section_role == "references":
            return "reference"
        if section_role == "front_matter":
            return "front_matter"
        if section_role == "unknown" and self._looks_like_front_matter_block(parent_content):
            return "front_matter"
        return "child"

    def _classify_table_content_role(self, parent_header: str, table_payload: str, section_role: str) -> str:
        if section_role == "references" or self._looks_like_reference_block(table_payload):
            return "reference"
        if section_role == "front_matter":
            return "front_matter"
        return "table"

    @staticmethod
    def _looks_like_reference_block(text: str) -> bool:
        normalized = " ".join(text.lower().split())
        patterns = (
            r"\bet al\.",
            r"\bdoi[:/]",
            r"\bjournal\b",
            r"\bvol(?:ume)?\b",
            r"\bpmid\b",
            r"\bissn\b",
            r"\bhttp[s]?://",
        )
        score = sum(1 for pattern in patterns if re.search(pattern, normalized))
        return score >= 2

    @staticmethod
    def _looks_like_front_matter_block(text: str) -> bool:
        normalized = " ".join(text.lower().split())
        patterns = (
            r"\bcorrespond(?:ence|ing)\b",
            r"\baffiliation\b",
            r"\bcopyright\b",
            r"\bcompeting interests?\b",
            r"\bfunding\b",
            r"\bdata availability\b",
            r"@[a-z0-9.-]+",
        )
        score = sum(1 for pattern in patterns if re.search(pattern, normalized))
        return score >= 2
