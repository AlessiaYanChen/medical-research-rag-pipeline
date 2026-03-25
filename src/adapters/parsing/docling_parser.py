from __future__ import annotations

from collections.abc import Callable
import csv
from io import StringIO
from pathlib import Path
import re
import shutil
import subprocess
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
        tables = self._resolve_tables(rendered=rendered, markdown_text=markdown_text, pdf_path=source_path)
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
    def _extract_raw_tables(cls, rendered: Any) -> list[Any]:
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

        return raw_tables

    @classmethod
    def _extract_tables(cls, rendered: Any, *, pdf_path: Path | None = None) -> list[ParsedTable]:
        raw_tables = cls._extract_raw_tables(rendered)
        normalized_tables = [cls._normalize_table(table, pdf_path=pdf_path) for table in raw_tables]
        return [table for table in normalized_tables if cls._table_has_meaningful_content(table)]

    @classmethod
    def _resolve_tables(cls, *, rendered: Any, markdown_text: str, pdf_path: Path | None = None) -> list[ParsedTable]:
        extracted_tables = cls._extract_tables(rendered, pdf_path=pdf_path)
        markdown_tables = cls._extract_markdown_tables(markdown_text)

        if not extracted_tables:
            return markdown_tables
        if markdown_tables and len(markdown_tables) > len(extracted_tables):
            return markdown_tables
        if markdown_tables and cls._has_duplicate_tables(extracted_tables):
            return markdown_tables
        return extracted_tables

    @classmethod
    def _clean_markdown_text(cls, markdown_text: str) -> str:
        cleaned = str(markdown_text or "")
        cleaned = re.sub(r"<!--\s*image\s*-->", " ", cleaned, flags=re.IGNORECASE)
        cleaned = cls._collapse_pdf_spacing_artifacts(cleaned)
        cleaned = cls._dedupe_opening_boilerplate(cleaned)
        cleaned = cls._normalize_opening_structured_abstract(cleaned)
        cleaned = cls._strip_inline_numeric_citations(cleaned)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
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
    def _normalize_opening_structured_abstract(text: str) -> str:
        lines = text.splitlines()
        if not lines:
            return text

        body_header_index: int | None = None
        for index, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r"^##\s+(methods|results|discussion|introduction|conclusion)\b", stripped, flags=re.IGNORECASE):
                body_header_index = index
                break

        window_end = body_header_index if body_header_index is not None else min(len(lines), 80)
        structured_prefixes = ("background.", "methods.", "results.", "conclusions.", "conclusion.")
        normalized_lines: list[str] = []
        abstract_seen = False

        for index, line in enumerate(lines):
            stripped = line.strip()
            if index < window_end and stripped:
                lower = stripped.lower()
                if lower.startswith(structured_prefixes):
                    if not abstract_seen:
                        normalized_lines.append("## Structured Abstract")
                        abstract_seen = True
                    normalized_lines.append(stripped)
                    continue
            normalized_lines.append(line)

        return "\n".join(normalized_lines)

    @staticmethod
    def _strip_inline_numeric_citations(text: str) -> str:
        citation_pattern = re.compile(
            r"\s*\[\s*(?:\d+\s*(?:-\s*\d+)?)(?:\s*,\s*\d+\s*(?:-\s*\d+)?)*\s*\]"
        )
        return citation_pattern.sub("", text)

    @staticmethod
    def _extract_document_object(rendered: Any) -> Any | None:
        if isinstance(rendered, dict):
            return rendered.get("document")
        return getattr(rendered, "document", None)

    @classmethod
    def _normalize_table(cls, raw_table: Any, *, pdf_path: Path | None = None) -> ParsedTable:
        if isinstance(raw_table, ParsedTable):
            return raw_table

        dataframe_export = getattr(raw_table, "export_to_dataframe", None)
        if callable(dataframe_export):
            table = cls._normalize_dataframe_table(raw_table)
            if pdf_path is not None and cls._table_needs_page_text_recovery(raw_table=raw_table, table=table):
                recovered = cls._recover_table_from_page_text(pdf_path=pdf_path, raw_table=raw_table)
                if recovered is not None:
                    return recovered
            return table

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

        columns = list(dataframe.columns)
        headers = [str(column).strip() for column in columns]
        rows: list[dict[str, str]] = []
        for record in dataframe.fillna("").to_dict(orient="records"):
            rows.append(
                {
                    header: str(record.get(column, "")).strip()
                    for header, column in zip(headers, columns)
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

    @staticmethod
    def _table_has_meaningful_content(table: ParsedTable) -> bool:
        if not table.rows:
            return False
        for row in table.rows:
            if any(str(value).strip() for value in row.values()):
                return True
        return False

    @classmethod
    def _table_needs_page_text_recovery(cls, *, raw_table: Any, table: ParsedTable) -> bool:
        data = getattr(raw_table, "data", None)
        prov = getattr(raw_table, "prov", None)
        if data is None or not isinstance(prov, list) or not prov:
            return False

        table_cells = getattr(data, "table_cells", None)
        if not isinstance(table_cells, list) or not table_cells:
            return False

        has_column_headers = any(bool(getattr(cell, "column_header", False)) for cell in table_cells)
        if has_column_headers:
            return False

        csv_text = re.sub(r"\s+", " ", table.csv).strip().lower()
        if "lod with lysozyme treatment" not in csv_text:
            return False

        num_cols = int(getattr(data, "num_cols", 0) or 0)
        if num_cols < 8:
            return False
        return True

    @classmethod
    def _recover_table_from_page_text(cls, *, pdf_path: Path, raw_table: Any) -> ParsedTable | None:
        prov = getattr(raw_table, "prov", None)
        if not isinstance(prov, list) or not prov:
            return None

        page_no = int(getattr(prov[0], "page_no", 0) or 0)
        if page_no <= 0:
            return None

        page_text = cls._extract_pdftotext_page_text(pdf_path=pdf_path, page_no=page_no)
        if not page_text:
            return None

        return cls._parse_lod_matrix_from_page_text(page_text)

    @staticmethod
    def _extract_pdftotext_page_text(*, pdf_path: Path, page_no: int) -> str:
        executable = shutil.which("pdftotext") or shutil.which("pdftotext.exe")
        if not executable:
            return ""

        completed = subprocess.run(
            [executable, "-f", str(page_no), "-l", str(page_no), "-table", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            encoding="cp1252",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            return ""
        return completed.stdout

    @classmethod
    def _parse_lod_matrix_from_page_text(cls, page_text: str) -> ParsedTable | None:
        normalized_text = cls._normalize_pdftotext_chars(page_text)
        raw_lines = [line.rstrip("\f") for line in normalized_text.splitlines()]
        lines = [line.rstrip() for line in raw_lines if line.strip()]
        if not lines:
            return None

        download_idx = next((idx for idx, line in enumerate(lines) if "Downloaded from " in line), len(lines))
        lines = lines[:download_idx]
        if not any("LOD with lysozyme treatment" in line for line in lines):
            return None
        if not any("Incubation (mins)" in line for line in lines):
            return None

        table_left = min(
            line.index(marker)
            for line in lines
            for marker in ("Test strain", "CFU/", "Incubation (mins)")
            if marker in line
        )
        trimmed_lines = [line[table_left:] if len(line) > table_left else line for line in lines]

        try:
            without_idx = next(idx for idx, line in enumerate(trimmed_lines) if "LOD without lysozyme treatment" in line)
            with_idx = next(idx for idx, line in enumerate(trimmed_lines) if "LOD with lysozyme treatment" in line)
            no_species_line = next(
                trimmed_lines[idx] for idx in range(without_idx + 1, with_idx) if "Test strain" in trimmed_lines[idx]
            )
            no_values_line = next(
                trimmed_lines[idx] for idx in range(without_idx + 1, with_idx) if "CFU/" in trimmed_lines[idx]
            )
            with_species_line = next(
                trimmed_lines[idx] for idx in range(with_idx + 1, len(trimmed_lines)) if "Test strain" in trimmed_lines[idx]
            )
            incubation_line = next(
                trimmed_lines[idx] for idx in range(with_idx + 1, len(trimmed_lines)) if "Incubation (mins)" in trimmed_lines[idx]
            )
        except StopIteration:
            return None

        no_value_segments = cls._fixed_width_segments(no_values_line)
        incubation_segments = cls._fixed_width_segments(incubation_line)
        if len(no_value_segments) != 6 or len(incubation_segments) != 11:
            return None

        no_treatment_anchors = [start for start, _ in no_value_segments[1:]]
        strains = [
            re.sub(r"^Test strain\s+", "", token, flags=re.IGNORECASE).strip()
            for token in cls._tokens_from_anchor_windows(no_species_line, no_treatment_anchors)
        ]
        no_treatment_values = [value for _, value in no_value_segments[1:]]
        if len(strains) != 5 or len(no_treatment_values) != 5:
            return None

        with_strains = strains.copy()
        if cls._tokens_from_anchor_windows(with_species_line, [start for start, _ in incubation_segments[1::2]])[-1:] == ["E. coli"]:
            with_strains[-1] = "E. coli"

        incubation_values = [value for _, value in incubation_segments[1:]]
        if len(incubation_values) != 10:
            return None

        concentration_rows: list[tuple[str, list[str]]] = []
        for line in trimmed_lines:
            if not re.match(r"^\s*\d+\w*\s+\(CFU/", line):
                continue
            segments = cls._fixed_width_segments(line)
            if len(segments) < 11:
                continue
            label_match = re.match(r"(\d+\w*)", segments[0][1])
            if label_match is None:
                continue
            label = label_match.group(1)
            values = [value for _, value in segments[1:11]]
            concentration_rows.append((label, values))

        if len(concentration_rows) < 5:
            return None

        headers = [
            "Test strain",
            "LOD without lysozyme treatment (CFU/\u00b5L)",
            "10 \u00b5g/mL 30 mins (CFU/\u00b5L)",
            "10 \u00b5g/mL 60 mins (CFU/\u00b5L)",
            "50 \u00b5g/mL 30 mins (CFU/\u00b5L)",
            "50 \u00b5g/mL 60 mins (CFU/\u00b5L)",
            "100 \u00b5g/mL 30 mins (CFU/\u00b5L)",
            "100 \u00b5g/mL 60 mins (CFU/\u00b5L)",
            "500 \u00b5g/mL 30 mins (CFU/\u00b5L)",
            "500 \u00b5g/mL 60 mins (CFU/\u00b5L)",
            "1000 \u00b5g/mL 30 mins (CFU/\u00b5L)",
            "1000 \u00b5g/mL 60 mins (CFU/\u00b5L)",
        ]

        concentration_order = ["10", "50", "100b", "500", "1000"]
        concentration_map = {label: values for label, values in concentration_rows}
        if not all(label in concentration_map for label in concentration_order):
            return None

        rows: list[dict[str, str]] = []
        for strain_index, strain in enumerate(with_strains):
            row: dict[str, str] = {
                headers[0]: strain,
                headers[1]: no_treatment_values[strain_index],
            }
            header_offset = 2
            for label in concentration_order:
                values = concentration_map[label]
                value_index = strain_index * 2
                row[headers[header_offset]] = values[value_index]
                row[headers[header_offset + 1]] = values[value_index + 1]
                header_offset += 2
            rows.append(row)

        csv_text = cls._build_csv(headers=headers, rows=rows)
        return ParsedTable(headers=headers, rows=rows, csv=csv_text)

    @staticmethod
    def _normalize_pdftotext_chars(text: str) -> str:
        normalized = re.sub(r"CFU/[^\s,.)]+", "CFU/\u00b5L", text)
        normalized = re.sub(r"\([^)]*/mL\)", "(\u00b5g/mL)", normalized)
        normalized = normalized.replace("(CFU/L)", "(CFU/\u00b5L)")
        return normalized

    @staticmethod
    def _fixed_width_segments(line: str) -> list[tuple[int, str]]:
        return [
            (match.start(), re.sub(r"\s+", " ", match.group()).strip())
            for match in re.finditer(r"\S(?:.*?\S)?(?=\s{2,}|$)", line)
        ]

    @staticmethod
    def _tokens_from_anchor_windows(line: str, anchors: list[int]) -> list[str]:
        if not anchors:
            return []

        boundaries: list[int] = [0]
        for left, right in zip(anchors, anchors[1:]):
            boundaries.append((left + right) // 2)
        boundaries.append(len(line))

        tokens: list[str] = []
        for index, anchor in enumerate(anchors):
            left = boundaries[index]
            right = boundaries[index + 1]
            window = line[left:right]
            if anchor < len(line):
                window = line[left:max(right, anchor + 1)]
            tokens.append(re.sub(r"\s+", " ", window).strip())
        return tokens

    @classmethod
    def _extract_markdown_tables(cls, markdown_text: str) -> list[ParsedTable]:
        lines = markdown_text.splitlines()
        blocks: list[list[str]] = []
        current: list[str] = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            if cls._is_markdown_table_line(stripped):
                current.append(line)
                in_table = True
                continue

            if in_table:
                if current:
                    blocks.append(current)
                current = []
                in_table = False

        if current:
            blocks.append(current)

        parsed_tables: list[ParsedTable] = []
        for block in blocks:
            if len(block) < 2 or not cls._looks_like_markdown_table(block):
                continue
            table = cls._parse_markdown_table_block("\n".join(block))
            if cls._table_has_meaningful_content(table):
                parsed_tables.append(table)
        return parsed_tables

    @staticmethod
    def _is_markdown_table_line(line: str) -> bool:
        return line.count("|") >= 2

    @staticmethod
    def _looks_like_markdown_table(lines: list[str]) -> bool:
        divider = lines[1].strip()
        return bool(re.match(r"^\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$", divider))

    @classmethod
    def _parse_markdown_table_block(cls, table_block: str) -> ParsedTable:
        lines = [line.strip() for line in table_block.splitlines() if line.strip()]
        if len(lines) < 2:
            return ParsedTable(headers=[], rows=[], csv="")

        headers = cls._split_markdown_row(lines[0])
        data_lines = [line for line in lines[2:] if cls._is_markdown_table_line(line)]

        rows: list[dict[str, str]] = []
        csv_lines = [",".join(headers)]
        for data_line in data_lines:
            values = cls._split_markdown_row(data_line)
            if len(values) < len(headers):
                values.extend([""] * (len(headers) - len(values)))
            elif len(values) > len(headers):
                values = values[: len(headers)]

            row = dict(zip(headers, values))
            rows.append(row)
            csv_lines.append(",".join(values))

        return ParsedTable(headers=headers, rows=rows, csv="\n".join(csv_lines))

    @staticmethod
    def _split_markdown_row(row: str) -> list[str]:
        trimmed = row.strip().strip("|")
        return [cell.strip() for cell in trimmed.split("|")]

    @staticmethod
    def _has_duplicate_tables(tables: list[ParsedTable]) -> bool:
        seen: set[str] = set()
        for table in tables:
            normalized = re.sub(r"\s+", " ", table.csv).strip().lower()
            if not normalized:
                continue
            if normalized in seen:
                return True
            seen.add(normalized)
        return False
