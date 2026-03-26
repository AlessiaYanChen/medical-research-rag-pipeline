from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_project_root_on_path()

from src.app.ingestion.parser_factory import DEFAULT_PARSER_NAME, PARSER_CHOICES, build_parser  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse one PDF with the selected parser and export text/tables."
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF file.")
    parser.add_argument(
        "--parser",
        choices=PARSER_CHOICES,
        default=DEFAULT_PARSER_NAME,
        help="Parser used for the debug export.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/parsed_debug",
        help="Directory where markdown/json/csv outputs will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf)
    out_dir = Path(args.out_dir)

    if not pdf_path.exists():
        print(f"ERROR: PDF file not found: {pdf_path}")
        return 1

    document_parser = build_parser(args.parser)
    start_time = time.time()
    print(f"Starting parse: {pdf_path} (parser={args.parser})")
    try:
        parsed = document_parser.parse(pdf_path)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to parse PDF: {exc}")
        print(traceback.format_exc())
        return 1
    elapsed = time.time() - start_time
    print(f"Parse completed in {elapsed:.1f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem

    markdown_path = out_dir / f"{stem}.main_text.md"
    markdown_path.write_text(parsed.markdown_text, encoding="utf-8")
    print(f"Main text length: {len(parsed.markdown_text)} characters")

    for idx, table in enumerate(parsed.tables, start=1):
        table_json_path = out_dir / f"{stem}.table_{idx:02d}.json"
        table_csv_path = out_dir / f"{stem}.table_{idx:02d}.csv"

        table_json_path.write_text(
            json.dumps(
                {
                    "headers": table.headers,
                    "rows": table.rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        table_csv_path.write_text(table.csv, encoding="utf-8")
        print(
            "Table "
            f"{idx:02d}: headers={len(table.headers)} rows={len(table.rows)} "
            f"json={table_json_path.name} csv={table_csv_path.name}"
        )

    print(f"Parsed: {pdf_path}")
    print(f"Main text markdown: {markdown_path}")
    print(f"Tables extracted: {len(parsed.tables)}")
    if parsed.tables:
        print(f"Table files written under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

