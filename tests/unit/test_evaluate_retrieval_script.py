from __future__ import annotations

from pathlib import Path

from scripts.evaluate_retrieval import resolve_output_paths


def test_resolve_output_paths_uses_dataset_specific_defaults_for_stable_dataset() -> None:
    json_out, csv_out = resolve_output_paths(
        dataset_path=Path("data/eval/sample_queries.json"),
        json_out="",
        csv_out="",
    )

    assert json_out == Path("data/eval/results/retrieval_eval_sample.json")
    assert csv_out == Path("data/eval/results/retrieval_eval_sample.csv")


def test_resolve_output_paths_uses_dataset_specific_defaults_for_expanded_dataset() -> None:
    json_out, csv_out = resolve_output_paths(
        dataset_path=Path("data/eval/expanded_queries.json"),
        json_out="",
        csv_out="",
    )

    assert json_out == Path("data/eval/results/retrieval_eval_expanded.json")
    assert csv_out == Path("data/eval/results/retrieval_eval_expanded.csv")


def test_resolve_output_paths_preserves_explicit_output_overrides() -> None:
    json_out, csv_out = resolve_output_paths(
        dataset_path=Path("data/eval/sample_queries.json"),
        json_out="custom/results.json",
        csv_out="custom/results.csv",
    )

    assert json_out == Path("custom/results.json")
    assert csv_out == Path("custom/results.csv")
