from __future__ import annotations

import hashlib
from pathlib import Path


def compute_file_identity(path: str | Path) -> dict[str, int | str]:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            if not block:
                break
            digest.update(block)
    return {
        "source_sha256": digest.hexdigest(),
        "file_size_bytes": file_path.stat().st_size,
    }
