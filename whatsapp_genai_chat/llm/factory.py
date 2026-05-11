import os
import importlib
from dotenv import load_dotenv
from whatsapp_genai_chat.llm.base import EmbeddingProvider, LLMProvider

load_dotenv()

_REGISTRY: dict[str, tuple[str, str, str]] = {
    # provider_name: (module_path, LLMClass, EmbeddingClass)
    "azure_openai": (
        "whatsapp_genai_chat.llm.azure_openai",
        "AzureOpenAILLM",
        "AzureOpenAIEmbedding",
    ),
    "openai": (
        "whatsapp_genai_chat.llm.openai_provider",
        "OpenAILLM",
        "OpenAIEmbedding",
    ),
    "anthropic": (
        "whatsapp_genai_chat.llm.anthropic_provider",
        "AnthropicLLM",
        "AnthropicEmbedding",
    ),
    "google": (
        "whatsapp_genai_chat.llm.google_provider",
        "GoogleLLM",
        "GoogleEmbedding",
    ),
}

_VALID = ", ".join(_REGISTRY)


def _provider() -> str:
    return os.environ.get("PROVIDER", "azure_openai").lower()


def _resolve(class_index: int):
    p = _provider()
    if p not in _REGISTRY:
        raise ValueError(f"Unknown provider: '{p}'. Must be one of: {_VALID}")
    module_path, llm_cls, emb_cls = _REGISTRY[p]
    cls_name = llm_cls if class_index == 0 else emb_cls
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)()


def get_llm_provider() -> LLMProvider:
    return _resolve(0)


def get_embedding_provider() -> EmbeddingProvider:
    return _resolve(1)
