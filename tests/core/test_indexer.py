import pytest
import pickle
import numpy as np
from pathlib import Path
from whatsapp_genai_chat.core.indexer import build_index, save_index, load_index, IndexData


def make_fake_embeddings(n=10, dim=8):
    return np.random.rand(n, dim).astype("float32")


def test_build_index_returns_index_data():
    texts = [f"message {i}" for i in range(10)]
    embeddings = make_fake_embeddings(10)
    idx = build_index(texts=texts, embeddings=embeddings, user1="Alice", user2="Bob")
    assert isinstance(idx, IndexData)
    assert idx.user1 == "Alice"
    assert idx.user2 == "Bob"
    assert len(idx.texts) == 10


def test_save_and_load_roundtrip(tmp_path):
    texts = [f"msg {i}" for i in range(5)]
    embeddings = make_fake_embeddings(5)
    idx = build_index(texts=texts, embeddings=embeddings, user1="A", user2="B")
    pkl_path = tmp_path / "test.pkl"
    save_index(idx, pkl_path)
    loaded = load_index(pkl_path)
    assert loaded.user1 == "A"
    assert loaded.user2 == "B"
    assert len(loaded.texts) == 5


def test_load_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_index(Path("indexes/nonexistent.pkl"))
