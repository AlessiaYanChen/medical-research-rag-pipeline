from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_project_root_on_path()

from src.app.tables.table_chunker import UnifiedChunker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UnifiedChunker against extracted markdown/table artifacts."
    )
    parser.add_argument(
        "--parsed-dir",
        required=True,
        help="Directory containing *.main_text.md and *.table_XX.(json|csv) files.",
    )
    parser.add_argument(
        "--doc-stem",
        help="Document stem prefix (e.g., 'RAPID'). If omitted, auto-detects when unique.",
    )
    parser.add_argument(
        "--doc-id",
        help="Document ID for chunk metadata. Defaults to doc stem.",
    )
    parser.add_argument(
        "--source-file",
        help="Source file name used in table context headers. Defaults to '<doc-stem>.pdf'.",
    )
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--overlap-paragraphs", type=int, default=1)
    parser.add_argument(
        "--preview",
        type=int,
        default=8,
        help="How many chunks to preview in stdout.",
    )
    parser.add_argument(
        "--save-json",
        help="Optional path to write chunk output as JSON.",
    )
    return parser.parse_args()


def _resolve_doc_stem(parsed_dir: Path, doc_stem: str | None) -> str:
    if doc_stem:
        return doc_stem

    md_files = sorted(parsed_dir.glob("*.main_text.md"))
    if len(md_files) != 1:
        raise ValueError(
            "Could not infer --doc-stem. Provide it explicitly when parsed-dir has 0 or multiple *.main_text.md files."
        )
    return md_files[0].name.removesuffix(".main_text.md")


def _load_table_artifacts(parsed_dir: Path, doc_stem: str) -> list[dict[str, Any]]:
    artifacts_by_index: dict[int, dict[str, Any]] = {}

    for csv_path in sorted(parsed_dir.glob(f"{doc_stem}.table_*.csv")):
        match = re.search(r"\.table_(\d+)\.csv$", csv_path.name)
        if not match:
            continue
        idx = int(match.group(1))
        artifacts_by_index.setdefault(idx, {})["csv"] = csv_path.read_text(encoding="utf-8")

    for json_path in sorted(parsed_dir.glob(f"{doc_stem}.table_*.json")):
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

    return [artifacts_by_index[idx] for idx in sorted(artifacts_by_index)]


def main() -> int:
    args = parse_args()
    parsed_dir = Path(args.parsed_dir)
    if not parsed_dir.exists():
        print(f"ERROR: parsed-dir does not exist: {parsed_dir}")
        return 1

    try:
        doc_stem = _resolve_doc_stem(parsed_dir, args.doc_stem)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1

    md_path = parsed_dir / f"{doc_stem}.main_text.md"
    if not md_path.exists():
        print(f"ERROR: markdown artifact not found: {md_path}")
        return 1

    markdown_text = md_path.read_text(encoding="utf-8")
    tables = _load_table_artifacts(parsed_dir, doc_stem)

    doc_id = args.doc_id or doc_stem
    source_file = args.source_file or f"{doc_stem}.pdf"
    chunker = UnifiedChunker(max_chars=args.max_chars, overlap_paragraphs=args.overlap_paragraphs)
    chunks = chunker.chunk_document(
        doc_id=doc_id,
        source_file=source_file,
        markdown_text=markdown_text,
        tables=tables,
    )

    text_count = sum(chunk.metadata.chunk_type == "text" for chunk in chunks)
    table_count = sum(chunk.metadata.chunk_type == "table" for chunk in chunks)
    print(f"Document stem: {doc_stem}")
    print(f"Chunks total: {len(chunks)}")
    print(f"Text chunks: {text_count}")
    print(f"Table chunks: {table_count}")
    print(f"Table artifacts loaded: {len(tables)}")

    if table_count != len(tables):
        print("WARNING: table chunk count does not match loaded table artifacts.")

    preview = max(0, args.preview)
    for idx, chunk in enumerate(chunks[:preview], start=1):
        content_preview = chunk.content.replace("\n", " ")[:220]
        print(
            f"[{idx}] id={chunk.id} type={chunk.metadata.chunk_type} "
            f"header='{chunk.metadata.parent_header}' page={chunk.metadata.page_number} :: {content_preview}"
        )

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_payload = [
            {
                "id": chunk.id,
                "doc_id": chunk.metadata.doc_id,
                "chunk_type": chunk.metadata.chunk_type,
                "parent_header": chunk.metadata.parent_header,
                "page_number": chunk.metadata.page_number,
                "content": chunk.content,
            }
            for chunk in chunks
        ]
        out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
        print(f"Saved chunk JSON: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
