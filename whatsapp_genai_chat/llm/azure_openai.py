import os
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from whatsapp_genai_chat.llm.base import EmbeddingProvider, LLMProvider


def _make_client() -> AzureOpenAI:
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,
        api_version=os.environ.get("AZURE_OPENAI_VERSION", "2024-02-01"),
    )


class AzureOpenAIEmbedding(EmbeddingProvider):
    def __init__(self):
        self._client = _make_client()
        self._deployment = os.environ["AZURE_EMBEDDING_DEPLOYMENT_NAME"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._deployment)
        return [item.embedding for item in response.data]


class AzureOpenAILLM(LLMProvider):
    def __init__(self):
        self._client = _make_client()
        self._deployment = os.environ["AZURE_LLM_DEPLOYMENT_NAME"]

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content
