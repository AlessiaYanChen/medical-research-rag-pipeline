from __future__ import annotations

import hashlib
from pathlib import Path

from src.app.ingestion.file_identity_utils import compute_file_identity


def test_compute_file_identity_returns_sha256_and_size(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    payload = b"example-pdf-bytes"
    pdf_path.write_bytes(payload)

    identity = compute_file_identity(pdf_path)

    assert identity == {
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "file_size_bytes": len(payload),
    }
