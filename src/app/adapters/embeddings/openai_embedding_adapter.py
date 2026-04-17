from __future__ import annotations

import time
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
        max_retries: int = 3,
        retry_base_delay_seconds: float = 1.0,
        client: Any | None = None,
        sleeper: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Embedding API key is required.")

        self._model = model
        self._provider = provider
        self._dimensions = dimensions
        self._batch_size = max(1, batch_size)
        self._max_retries = max(1, int(max_retries))
        self._retry_base_delay_seconds = max(0.0, float(retry_base_delay_seconds))
        self._sleeper = sleeper or time.sleep

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

            response = self._create_embeddings_with_retry(request)
            batch_vectors = [list(item.embedding) for item in response.data]
            vectors.extend(batch_vectors)

        return vectors

    def _create_embeddings_with_retry(self, request: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return self._client.embeddings.create(**request)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self._max_retries or not self._is_retryable_embedding_error(exc):
                    raise
                self._sleeper(self._retry_base_delay_seconds * (2 ** (attempt - 1)))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Embedding request failed without raising an exception.")

    @staticmethod
    def _is_retryable_embedding_error(error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int) and status_code in {408, 409, 429}:
            return True
        if isinstance(status_code, int) and status_code >= 500:
            return True

        error_type_name = error.__class__.__name__.lower()
        return error_type_name in {
            "internalservererror",
            "apiconnectionerror",
            "apitimeouterror",
            "ratelimiterror",
        }
