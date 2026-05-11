from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        ...
