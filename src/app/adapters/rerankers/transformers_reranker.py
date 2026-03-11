from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from src.app.ports.re_ranker_port import ReRankerPort
from src.domain.models.chunk import Chunk


class TransformersReRanker(ReRankerPort):
    """
    Cross-encoder style re-ranker backed by Hugging Face transformers.

    By default this uses a text-classification pipeline over (query, chunk) pairs.
    A custom scorer can be injected for testing or alternative runtime behavior.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        scorer: Callable[[str, list[Chunk]], Sequence[float]] | None = None,
        device: int | str | None = None,
    ) -> None:
        self._model_name = model_name
        self._scorer = scorer or self._build_pipeline_scorer(model_name=model_name, device=device)

    def rank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        if not chunks or top_n <= 0:
            return []

        scores = list(self._scorer(query, chunks))
        if len(scores) != len(chunks):
            raise ValueError("Re-ranker scorer must return one score per chunk.")

        ranked_pairs = sorted(
            zip(chunks, scores),
            key=lambda item: item[1],
            reverse=True,
        )
        return [chunk for chunk, _ in ranked_pairs[:top_n]]

    @staticmethod
    def _build_pipeline_scorer(
        model_name: str,
        device: int | str | None,
    ) -> Callable[[str, list[Chunk]], Sequence[float]]:
        from transformers import pipeline  # type: ignore

        pipeline_kwargs: dict[str, Any] = {
            "task": "text-classification",
            "model": model_name,
            "tokenizer": model_name,
        }
        if device is not None:
            pipeline_kwargs["device"] = device

        classifier = pipeline(**pipeline_kwargs)

        def score(query: str, chunks: list[Chunk]) -> Sequence[float]:
            pairs = [
                {"text": query, "text_pair": chunk.content}
                for chunk in chunks
            ]
            outputs = classifier(
                pairs,
                truncation=True,
                max_length=512,
                batch_size=min(16, max(1, len(pairs))),
            )

            scores: list[float] = []
            for output in outputs:
                label = str(output.get("label", "")).upper()
                raw_score = float(output.get("score", 0.0))
                # Some cross-encoder checkpoints emit LABEL_0 as the relevant class.
                scores.append(-raw_score if label == "LABEL_0" else raw_score)
            return scores

        return score

