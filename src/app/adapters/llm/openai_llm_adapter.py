from __future__ import annotations

from typing import Any

from src.app.ports.llm_port import LLMPort


class OpenAILLMAdapter(LLMPort):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        azure_endpoint: str | None = None,
        azure_api_version: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("LLM API key is required.")

        self._model = model
        self._provider = provider
        if client is not None:
            self._client = client
        else:
            if provider == "azure_openai":
                if not azure_endpoint:
                    raise ValueError("Azure OpenAI endpoint is required.")
                if not azure_api_version:
                    raise ValueError("Azure OpenAI API version is required.")

                from openai import AzureOpenAI  # type: ignore

                self._client = AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=azure_endpoint,
                    api_version=azure_api_version,
                )
            else:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        if self._provider == "azure_openai":
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical research assistant.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            return response.choices[0].message.content.strip()

        response = self._client.responses.create(
            model=self._model,
            input=prompt,
        )
        return getattr(response, "output_text", "").strip()
