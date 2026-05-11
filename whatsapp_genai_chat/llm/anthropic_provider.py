import os
from whatsapp_genai_chat.llm.base import EmbeddingProvider, LLMProvider


class AnthropicLLM(LLMProvider):
    def __init__(self):
        import anthropic
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self._max_tokens = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "2048"))

    def complete(self, system: str, user: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text


class AnthropicEmbedding(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "Anthropic does not provide an embedding API. "
            "Use a different PROVIDER for embeddings (e.g. azure_openai or openai)."
        )
