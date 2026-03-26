from __future__ import annotations

from scripts.inspect_retrieval_candidates import _console_safe_text


def test_console_safe_text_preserves_encodable_text(monkeypatch) -> None:
    class Stdout:
        encoding = "utf-8"

    monkeypatch.setattr("scripts.inspect_retrieval_candidates.sys.stdout", Stdout())

    assert _console_safe_text("plain text") == "plain text"
    assert _console_safe_text("value ≥ threshold") == "value ≥ threshold"


def test_console_safe_text_replaces_unencodable_characters(monkeypatch) -> None:
    class Stdout:
        encoding = "cp1252"

    monkeypatch.setattr("scripts.inspect_retrieval_candidates.sys.stdout", Stdout())

    assert _console_safe_text("value ≥ threshold") == "value ? threshold"
