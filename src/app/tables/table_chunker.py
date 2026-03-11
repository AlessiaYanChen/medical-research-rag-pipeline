from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from src.domain.models.chunk import Chunk, ChunkMetadata


class UnifiedChunker:
    """Chunk mixed markdown + table artifacts into a unified sequence of Chunk models."""

    def __init__(self, max_chars: int = 900, overlap_paragraphs: int = 1) -> None:
        self.max_chars = max_chars
        self.overlap_paragraphs = max(0, overlap_paragraphs)

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
        chunk_counter = 0
        active_header = "Unknown"
        table_index = 0
        pending_paragraphs: list[str] = []
        pending_header = active_header

        for block in blocks:
            stripped = block.strip()
            if not stripped:
                continue

            header = self._extract_header(stripped)
            if header is not None:
                built_chunks, chunk_counter = self._build_text_chunks(
                    doc_id=doc_id,
                    parent_header=pending_header,
                    paragraphs=pending_paragraphs,
                    chunk_id_start=chunk_counter + 1,
                )
                chunks.extend(built_chunks)
                pending_paragraphs = []
                active_header = header
                pending_header = active_header
                continue

            if self._is_table_block(stripped):
                built_chunks, chunk_counter = self._build_text_chunks(
                    doc_id=doc_id,
                    parent_header=pending_header,
                    paragraphs=pending_paragraphs,
                    chunk_id_start=chunk_counter + 1,
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
                        chunk_id=self._make_chunk_id(doc_id, chunk_counter + 1),
                        doc_id=doc_id,
                        source_file=source_file,
                        table_index=table_index,
                        parent_header=active_header,
                        table_artifact=artifact,
                    )
                )
                chunk_counter += 1
                continue

            pending_paragraphs.append(stripped)

        # Ensure discovered artifacts are kept as atomic units even if markdown table
        # blocks are missing or fewer than extracted artifacts.
        while table_index < len(table_artifacts):
            table_index += 1
            chunks.append(
                self._build_table_chunk(
                    chunk_id=self._make_chunk_id(doc_id, chunk_counter + 1),
                    doc_id=doc_id,
                    source_file=source_file,
                    table_index=table_index,
                    parent_header=active_header,
                    table_artifact=table_artifacts[table_index - 1],
                )
            )
            chunk_counter += 1

        built_chunks, chunk_counter = self._build_text_chunks(
            doc_id=doc_id,
            parent_header=pending_header,
            paragraphs=pending_paragraphs,
            chunk_id_start=chunk_counter + 1,
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
    def _extract_header(block: str) -> str | None:
        line = block.splitlines()[0].strip()
        if re.match(r"^#{1,6}\s+", line):
            return re.sub(r"^#{1,6}\s+", "", line).strip()
        return None

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
        paragraphs: list[str],
        chunk_id_start: int,
    ) -> tuple[list[Chunk], int]:
        if not paragraphs:
            return [], chunk_id_start - 1

        chunks: list[Chunk] = []
        start = 0
        total = len(paragraphs)
        step_floor = max(1, self.overlap_paragraphs)
        next_chunk_id = chunk_id_start

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

            content = "\n\n".join(current)
            chunks.append(
                Chunk(
                    id=self._make_chunk_id(doc_id, next_chunk_id),
                    content=content,
                    metadata=ChunkMetadata(
                        doc_id=doc_id,
                        chunk_type="text",
                        parent_header=parent_header,
                        page_number=self._extract_page_number(content),
                    ),
                )
            )
            next_chunk_id += 1

            if idx >= total:
                break
            start = max(start + step_floor, idx - self.overlap_paragraphs)

        return chunks, next_chunk_id - 1

    def _build_table_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        source_file: str,
        table_index: int,
        parent_header: str,
        table_artifact: dict[str, Any],
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

        return Chunk(
            id=chunk_id,
            content=content,
            metadata=ChunkMetadata(
                doc_id=doc_id,
                chunk_type="table",
                parent_header=parent_header,
                page_number=page_number,
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
    def _make_chunk_id(doc_id: str, sequence: int) -> str:
        return f"{doc_id}:{sequence:05d}"
