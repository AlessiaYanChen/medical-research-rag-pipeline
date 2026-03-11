from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_project_root_on_path()

from src.adapters.parsing.marker_parser import MarkerParser  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse one PDF with MarkerParser and export text/tables."
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF file.")
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

    parser = MarkerParser()
    try:
        parsed = parser.parse(pdf_path)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to parse PDF: {exc}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem

    markdown_path = out_dir / f"{stem}.main_text.md"
    markdown_path.write_text(parsed.markdown_text, encoding="utf-8")

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

    print(f"Parsed: {pdf_path}")
    print(f"Main text markdown: {markdown_path}")
    print(f"Tables extracted: {len(parsed.tables)}")
    if parsed.tables:
        print(f"Table files written under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

