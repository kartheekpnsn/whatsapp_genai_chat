import os
from whatsapp_genai_chat.llm.base import EmbeddingProvider, LLMProvider


class GoogleLLM(LLMProvider):
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self._model_name = os.environ.get("GOOGLE_MODEL", "gemini-2.0-flash")
        self._genai = genai

    def complete(self, system: str, user: str) -> str:
        model = self._genai.GenerativeModel(self._model_name, system_instruction=system)
        response = model.generate_content(user)
        return response.text


class GoogleEmbedding(EmbeddingProvider):
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self._model = os.environ.get("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004")
        self._genai = genai

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            result = self._genai.embed_content(model=self._model, content=text)
            results.append(result["embedding"])
        return results
