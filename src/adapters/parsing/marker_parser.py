from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re

from src.ports.parser_port import ParsedDocument, ParsedTable, ParserPort


class MarkerParser(ParserPort):
    """PDF parser adapter that converts to markdown and isolates markdown tables."""

    def __init__(self, markdown_converter: Callable[[Path], str] | None = None) -> None:
        self._markdown_converter = markdown_converter or self._convert_with_marker

    def parse(self, pdf_path: Path | str) -> ParsedDocument:
        source_path = Path(pdf_path)
        if not source_path.exists():
            raise FileNotFoundError(f"PDF file not found: {source_path}")

        markdown = self._markdown_converter(source_path)
        table_blocks = self._extract_markdown_table_blocks(markdown)
        tables = [self._parse_table_block(block) for block in table_blocks]

        return ParsedDocument(
            source_path=source_path,
            markdown_text=markdown.strip(),
            tables=tables,
        )

    @staticmethod
    def _extract_markdown_table_blocks(markdown: str) -> list[str]:
        lines = markdown.splitlines()
        blocks: list[list[str]] = []
        current: list[str] = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            if MarkerParser._is_table_line(stripped):
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

        parsed_blocks: list[str] = []
        for block in blocks:
            if len(block) >= 2 and MarkerParser._looks_like_markdown_table(block):
                parsed_blocks.append("\n".join(block))

        return parsed_blocks

    @staticmethod
    def _parse_table_block(table_block: str) -> ParsedTable:
        lines = [line.strip() for line in table_block.splitlines() if line.strip()]
        if len(lines) < 2:
            return ParsedTable(headers=[], rows=[], csv="")

        header_cells = MarkerParser._split_markdown_row(lines[0])
        data_lines = [line for line in lines[2:] if MarkerParser._is_table_line(line)]

        rows: list[dict[str, str]] = []
        csv_lines = [",".join(header_cells)]
        for data_line in data_lines:
            values = MarkerParser._split_markdown_row(data_line)
            if len(values) < len(header_cells):
                values.extend([""] * (len(header_cells) - len(values)))
            elif len(values) > len(header_cells):
                values = values[: len(header_cells)]

            row = dict(zip(header_cells, values))
            rows.append(row)
            csv_lines.append(",".join(values))

        return ParsedTable(headers=header_cells, rows=rows, csv="\n".join(csv_lines))

    @staticmethod
    def _looks_like_markdown_table(lines: list[str]) -> bool:
        if len(lines) < 2:
            return False
        divider = lines[1].strip()
        return bool(re.match(r"^\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$", divider))

    @staticmethod
    def _split_markdown_row(row: str) -> list[str]:
        trimmed = row.strip().strip("|")
        return [cell.strip() for cell in trimmed.split("|")]

    @staticmethod
    def _is_table_line(line: str) -> bool:
        return line.count("|") >= 2

    @staticmethod
    def _convert_with_marker(pdf_path: Path) -> str:
        """
        Convert PDF to markdown using Marker.

        The concrete Marker API may vary by version, so this method is kept small
        and isolated to simplify adaptation during integration.
        """
        try:
            from marker.converters.pdf import PdfConverter  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Marker is not installed. Provide a markdown_converter for tests "
                "or install Marker for production parsing."
            ) from exc

        converter = None
        # Marker API changed: newer versions require artifact_dict.
        try:
            from marker.models import create_model_dict  # type: ignore

            converter = PdfConverter(artifact_dict=create_model_dict())
        except TypeError:
            # Backward compatibility with older Marker signatures.
            converter = PdfConverter()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to initialize Marker PdfConverter: {exc}") from exc

        rendered = converter(str(pdf_path))

        # Best-effort extraction for common Marker return types.
        if isinstance(rendered, str):
            return rendered
        if isinstance(rendered, dict) and "markdown" in rendered:
            return str(rendered["markdown"])
        if hasattr(rendered, "markdown"):
            return str(rendered.markdown)

        raise RuntimeError("Unsupported Marker output format; expected markdown content.")
