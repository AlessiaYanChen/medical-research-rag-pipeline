from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.app.adapters.embeddings.openai_embedding_adapter import OpenAIEmbeddingAdapter


class FakeEmbeddingsClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        inputs = kwargs["input"]
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[float(index), float(index) + 0.5])
                for index, _ in enumerate(inputs)
            ]
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsClient()


class FakeRetryableError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status={status_code}")
        self.status_code = status_code


def test_openai_embedding_adapter_batches_and_forwards_dimensions() -> None:
    client = FakeOpenAIClient()
    adapter = OpenAIEmbeddingAdapter(
        api_key="test-key",
        model="text-embedding-3-large",
        dimensions=1024,
        batch_size=2,
        client=client,
    )

    vectors = adapter(["one", "two", "three"])

    assert len(client.embeddings.calls) == 2
    assert client.embeddings.calls[0]["model"] == "text-embedding-3-large"
    assert client.embeddings.calls[0]["dimensions"] == 1024
    assert client.embeddings.calls[0]["input"] == ["one", "two"]
    assert client.embeddings.calls[1]["input"] == ["three"]
    assert vectors == [[0.0, 0.5], [1.0, 1.5], [0.0, 0.5]]


def test_openai_embedding_adapter_requires_api_key() -> None:
    with pytest.raises(ValueError, match="Embedding API key is required."):
        OpenAIEmbeddingAdapter(api_key="")


def test_openai_embedding_adapter_requires_azure_fields() -> None:
    with pytest.raises(ValueError, match="Azure OpenAI endpoint is required"):
        OpenAIEmbeddingAdapter(
            api_key="test-key",
            provider="azure_openai",
            azure_api_version="2024-10-21",
        )

    with pytest.raises(ValueError, match="Azure OpenAI API version is required"):
        OpenAIEmbeddingAdapter(
            api_key="test-key",
            provider="azure_openai",
            azure_endpoint="https://example.openai.azure.com",
        )


def test_openai_embedding_adapter_retries_transient_errors() -> None:
    sleep_calls: list[float] = []

    class FlakyEmbeddingsClient:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise FakeRetryableError(status_code=500)
            inputs = kwargs["input"]
            return SimpleNamespace(
                data=[
                    SimpleNamespace(embedding=[float(index), float(index) + 0.25])
                    for index, _ in enumerate(inputs)
                ]
            )

    client = SimpleNamespace(embeddings=FlakyEmbeddingsClient())
    adapter = OpenAIEmbeddingAdapter(
        api_key="test-key",
        client=client,
        sleeper=sleep_calls.append,
        retry_base_delay_seconds=0.5,
        max_retries=3,
    )

    vectors = adapter(["one"])

    assert client.embeddings.calls == 3
    assert sleep_calls == [0.5, 1.0]
    assert vectors == [[0.0, 0.25]]


def test_openai_embedding_adapter_does_not_retry_non_retryable_errors() -> None:
    sleep_calls: list[float] = []

    class FatalEmbeddingsClient:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise ValueError("bad request")

    client = SimpleNamespace(embeddings=FatalEmbeddingsClient())
    adapter = OpenAIEmbeddingAdapter(
        api_key="test-key",
        client=client,
        sleeper=sleep_calls.append,
        max_retries=3,
    )

    with pytest.raises(ValueError, match="bad request"):
        adapter(["one"])

    assert client.embeddings.calls == 1
    assert sleep_calls == []
