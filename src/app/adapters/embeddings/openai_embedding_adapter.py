from __future__ import annotations

from typing import Any


class OpenAIEmbeddingAdapter:
    """Embedding adapter for OpenAI and Azure OpenAI."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        provider: str = "openai",
        azure_endpoint: str | None = None,
        azure_api_version: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 64,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Embedding API key is required.")

        self._model = model
        self._provider = provider
        self._dimensions = dimensions
        self._batch_size = max(1, batch_size)

        if client is not None:
            self._client = client
            return

        if provider == "azure_openai":
            if not azure_endpoint:
                raise ValueError("Azure OpenAI endpoint is required for embeddings.")
            if not azure_api_version:
                raise ValueError("Azure OpenAI API version is required for embeddings.")

            from openai import AzureOpenAI  # type: ignore

            self._client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
            )
            return

        from openai import OpenAI  # type: ignore

        self._client = OpenAI(api_key=api_key)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            request: dict[str, Any] = {
                "model": self._model,
                "input": batch,
            }
            if self._dimensions is not None:
                request["dimensions"] = self._dimensions

            response = self._client.embeddings.create(**request)
            batch_vectors = [list(item.embedding) for item in response.data]
            vectors.extend(batch_vectors)

        return vectors
