import os
from openai import OpenAI
from whatsapp_genai_chat.llm.base import EmbeddingProvider, LLMProvider


class OpenAIEmbedding(EmbeddingProvider):
    def __init__(self):
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]


class OpenAILLM(LLMProvider):
    def __init__(self):
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = os.environ.get("OPENAI_MODEL", "gpt-4o")

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content
