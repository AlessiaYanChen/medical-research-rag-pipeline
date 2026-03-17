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
