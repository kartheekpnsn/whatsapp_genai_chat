import os
import pytest
from unittest.mock import patch, MagicMock
from whatsapp_genai_chat.llm.factory import get_llm_provider, get_embedding_provider
from whatsapp_genai_chat.llm.base import LLMProvider, EmbeddingProvider


def test_factory_returns_llm_provider_instance():
    with patch.dict(os.environ, {"PROVIDER": "openai", "OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o"}):
        with patch("whatsapp_genai_chat.llm.openai_provider.OpenAI"):
            provider = get_llm_provider()
            assert isinstance(provider, LLMProvider)


def test_factory_returns_embedding_provider_instance():
    with patch.dict(os.environ, {"PROVIDER": "openai", "OPENAI_API_KEY": "test-key", "OPENAI_EMBEDDING_MODEL": "text-embedding-3-large"}):
        with patch("whatsapp_genai_chat.llm.openai_provider.OpenAI"):
            provider = get_embedding_provider()
            assert isinstance(provider, EmbeddingProvider)


def test_factory_raises_on_unknown_provider():
    with patch.dict(os.environ, {"PROVIDER": "unknown_provider"}):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_llm_provider()


def test_anthropic_embedding_raises_not_implemented():
    with patch.dict(os.environ, {"PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"}):
        with patch("whatsapp_genai_chat.llm.anthropic_provider.anthropic.Anthropic"):
            provider = get_embedding_provider()
            with pytest.raises(NotImplementedError):
                provider.embed(["test"])
