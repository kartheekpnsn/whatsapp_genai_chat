import os
import google.generativeai as genai
from whatsapp_genai_chat.llm.base import EmbeddingProvider, LLMProvider


class GoogleLLM(LLMProvider):
    def __init__(self):
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self._model = genai.GenerativeModel(os.environ.get("GOOGLE_MODEL", "gemini-2.0-flash"))

    def complete(self, system: str, user: str) -> str:
        prompt = f"{system}\n\n{user}"
        response = self._model.generate_content(prompt)
        return response.text


class GoogleEmbedding(EmbeddingProvider):
    def __init__(self):
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self._model = os.environ.get("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004")

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            result = genai.embed_content(model=self._model, content=text)
            results.append(result["embedding"])
        return results
