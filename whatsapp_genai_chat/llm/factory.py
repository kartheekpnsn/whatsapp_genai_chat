import os
from dotenv import load_dotenv
from whatsapp_genai_chat.llm.base import EmbeddingProvider, LLMProvider

load_dotenv()


def _provider() -> str:
    return os.environ.get("PROVIDER", "azure_openai").lower()


def get_llm_provider() -> LLMProvider:
    p = _provider()
    if p == "azure_openai":
        from whatsapp_genai_chat.llm.azure_openai import AzureOpenAILLM
        return AzureOpenAILLM()
    if p == "openai":
        from whatsapp_genai_chat.llm.openai_provider import OpenAILLM
        return OpenAILLM()
    if p == "anthropic":
        from whatsapp_genai_chat.llm.anthropic_provider import AnthropicLLM
        return AnthropicLLM()
    if p == "google":
        from whatsapp_genai_chat.llm.google_provider import GoogleLLM
        return GoogleLLM()
    raise ValueError(f"Unknown provider: '{p}'. Must be one of: azure_openai, openai, anthropic, google")


def get_embedding_provider() -> EmbeddingProvider:
    p = _provider()
    if p == "azure_openai":
        from whatsapp_genai_chat.llm.azure_openai import AzureOpenAIEmbedding
        return AzureOpenAIEmbedding()
    if p == "openai":
        from whatsapp_genai_chat.llm.openai_provider import OpenAIEmbedding
        return OpenAIEmbedding()
    if p == "anthropic":
        from whatsapp_genai_chat.llm.anthropic_provider import AnthropicEmbedding
        return AnthropicEmbedding()
    if p == "google":
        from whatsapp_genai_chat.llm.google_provider import GoogleEmbedding
        return GoogleEmbedding()
    raise ValueError(f"Unknown provider: '{p}'. Must be one of: azure_openai, openai, anthropic, google")
