from __future__ import annotations

from collections.abc import Callable
import csv
from io import StringIO
from pathlib import Path
import re
from typing import Any

from src.ports.parser_port import ParsedDocument, ParsedTable, ParserPort


class DoclingParser(ParserPort):
    """PDF parser adapter that converts Docling output into the repo's parser contract."""

    def __init__(self, document_converter: Callable[[Path], Any] | None = None) -> None:
        self._document_converter = document_converter or self._convert_with_docling

    def parse(self, pdf_path: Path | str) -> ParsedDocument:
        source_path = Path(pdf_path)
        if not source_path.exists():
            raise FileNotFoundError(f"PDF file not found: {source_path}")

        rendered = self._document_converter(source_path)
        markdown_text = self._clean_markdown_text(self._extract_markdown_text(rendered))
        tables = self._extract_tables(rendered)
        if not markdown_text.strip() and not tables:
            raise RuntimeError("Docling returned no markdown text or tables.")

        return ParsedDocument(
            source_path=source_path,
            markdown_text=markdown_text.strip(),
            tables=tables,
        )

    @staticmethod
    def _convert_with_docling(pdf_path: Path) -> Any:
        try:
            from docling.document_converter import DocumentConverter  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Docling is not installed. Provide a document_converter for tests "
                "or install Docling for parser bakeoff experiments."
            ) from exc

        try:
            converter = DocumentConverter()
            return converter.convert(str(pdf_path))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Docling conversion failed: {exc}") from exc

    @classmethod
    def _extract_markdown_text(cls, rendered: Any) -> str:
        candidates = [rendered]
        document = cls._extract_document_object(rendered)
        if document is not None and document is not rendered:
            candidates.append(document)

        for candidate in candidates:
            if isinstance(candidate, dict):
                for key in ("markdown_text", "markdown", "text"):
                    value = candidate.get(key)
                    if isinstance(value, str) and value.strip():
                        return value
            else:
                for attr in ("markdown_text", "markdown", "text"):
                    value = getattr(candidate, attr, None)
                    if isinstance(value, str) and value.strip():
                        return value
                export_method = getattr(candidate, "export_to_markdown", None)
                if callable(export_method):
                    value = export_method()
                    if isinstance(value, str) and value.strip():
                        return value

        return ""

    @classmethod
    def _extract_tables(cls, rendered: Any) -> list[ParsedTable]:
        candidates = [rendered]
        document = cls._extract_document_object(rendered)
        if document is not None and document is not rendered:
            candidates.append(document)

        raw_tables: Any = None
        for candidate in candidates:
            if isinstance(candidate, dict) and isinstance(candidate.get("tables"), list):
                raw_tables = candidate["tables"]
                break
            value = getattr(candidate, "tables", None)
            if isinstance(value, list):
                raw_tables = value
                break

        if not isinstance(raw_tables, list):
            return []

        return [cls._normalize_table(table) for table in raw_tables]

    @classmethod
    def _clean_markdown_text(cls, markdown_text: str) -> str:
        cleaned = str(markdown_text or "")
        cleaned = re.sub(r"<!--\s*image\s*-->", " ", cleaned, flags=re.IGNORECASE)
        cleaned = cls._collapse_pdf_spacing_artifacts(cleaned)
        cleaned = cls._dedupe_opening_boilerplate(cleaned)
        cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _collapse_pdf_spacing_artifacts(text: str) -> str:
        collapsed = text
        collapsed = re.sub(r"(?<=\w)[ \t]{2,}(?=\w)", " ", collapsed)
        collapsed = re.sub(r"([A-Za-z])\s+fi\s+rst\b", r"\1first", collapsed, flags=re.IGNORECASE)
        collapsed = re.sub(r"([A-Za-z])\s+fi\s+nal\b", r"\1final", collapsed, flags=re.IGNORECASE)
        collapsed = re.sub(r"([A-Za-z])\s+ed\s+(?=[A-Za-z]{2,}\b)", r"\1ed ", collapsed)
        return collapsed

    @staticmethod
    def _dedupe_opening_boilerplate(text: str) -> str:
        lines = text.splitlines()
        cleaned_lines: list[str] = []
        seen_author_notice = False
        seen_methods_abstract = False

        for line in lines:
            normalized = " ".join(line.split()).strip()
            lower = normalized.lower()
            if not normalized:
                cleaned_lines.append("")
                continue

            if "published by oxford university press" in lower and "all rights reserved" in lower:
                if seen_author_notice:
                    continue
                seen_author_notice = True

            if lower.startswith("methods.") and "randomized to soc testing" in lower:
                if seen_methods_abstract:
                    continue
                seen_methods_abstract = True

            cleaned_lines.append(normalized)

        return "\n".join(cleaned_lines)

    @staticmethod
    def _extract_document_object(rendered: Any) -> Any | None:
        if isinstance(rendered, dict):
            return rendered.get("document")
        return getattr(rendered, "document", None)

    @classmethod
    def _normalize_table(cls, raw_table: Any) -> ParsedTable:
        if isinstance(raw_table, ParsedTable):
            return raw_table

        dataframe_export = getattr(raw_table, "export_to_dataframe", None)
        if callable(dataframe_export):
            return cls._normalize_dataframe_table(raw_table)

        payload = cls._table_payload(raw_table)
        headers = [str(value).strip() for value in payload.get("headers", []) if str(value).strip()]
        rows_payload = payload.get("rows", [])
        csv_text = str(payload.get("csv", "")).strip()

        rows = cls._normalize_rows(rows_payload, headers=headers)
        if not headers and rows:
            headers = list(rows[0].keys())
        if not csv_text:
            csv_text = cls._build_csv(headers=headers, rows=rows)

        return ParsedTable(headers=headers, rows=rows, csv=csv_text)

    @classmethod
    def _normalize_dataframe_table(cls, raw_table: Any) -> ParsedTable:
        try:
            dataframe = raw_table.export_to_dataframe()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to export Docling table to dataframe: {exc}") from exc

        headers = [str(column).strip() for column in list(dataframe.columns)]
        rows: list[dict[str, str]] = []
        for record in dataframe.fillna("").to_dict(orient="records"):
            rows.append(
                {
                    header: str(record.get(header, "")).strip()
                    for header in headers
                }
            )

        csv_text = cls._build_csv(headers=headers, rows=rows)
        return ParsedTable(headers=headers, rows=rows, csv=csv_text)

    @staticmethod
    def _table_payload(raw_table: Any) -> dict[str, Any]:
        if isinstance(raw_table, dict):
            return raw_table

        payload: dict[str, Any] = {}
        for attr in ("headers", "rows", "csv"):
            value = getattr(raw_table, attr, None)
            if value is not None:
                payload[attr] = value
        if not payload:
            raise RuntimeError("Unsupported Docling table format; expected headers/rows/csv fields.")
        return payload

    @classmethod
    def _normalize_rows(
        cls,
        rows_payload: Any,
        *,
        headers: list[str],
    ) -> list[dict[str, str]]:
        if not isinstance(rows_payload, list):
            return []
        if not rows_payload:
            return []

        first_row = rows_payload[0]
        if isinstance(first_row, dict):
            normalized_headers = headers or [str(key).strip() for key in first_row.keys()]
            rows: list[dict[str, str]] = []
            for raw_row in rows_payload:
                if not isinstance(raw_row, dict):
                    continue
                rows.append(
                    {
                        header: str(raw_row.get(header, "")).strip()
                        for header in normalized_headers
                    }
                )
            return rows

        if isinstance(first_row, (list, tuple)):
            matrix = [[str(cell).strip() for cell in row] for row in rows_payload if isinstance(row, (list, tuple))]
            if not matrix:
                return []
            normalized_headers = headers or matrix[0]
            data_rows = matrix if headers else matrix[1:]
            normalized_rows: list[dict[str, str]] = []
            for row in data_rows:
                values = list(row)
                if len(values) < len(normalized_headers):
                    values.extend([""] * (len(normalized_headers) - len(values)))
                elif len(values) > len(normalized_headers):
                    values = values[: len(normalized_headers)]
                normalized_rows.append(dict(zip(normalized_headers, values)))
            return normalized_rows

        return []

    @staticmethod
    def _build_csv(*, headers: list[str], rows: list[dict[str, str]]) -> str:
        if not headers:
            return ""
        buffer = StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(header, "") for header in headers])
        return buffer.getvalue().strip()
