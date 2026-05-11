import pytest
import pickle
import numpy as np
import faiss
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
    assert idx.row_indices == []  # no DataFrame provided


def test_build_index_row_indices_with_dataframe():
    import pandas as pd
    texts = ["hello", "goodbye"]
    df = pd.DataFrame([
        {"sender": "Bob", "message": "hello"},
        {"sender": "Alice", "message": "hi"},
        {"sender": "Bob", "message": "goodbye"},
        {"sender": "Alice", "message": "bye"},
    ])
    embeddings = make_fake_embeddings(2, dim=8)
    idx = build_index(texts=texts, embeddings=embeddings, user1="Alice", user2="Bob", df=df)
    assert idx.row_indices == [0, 2]


def test_save_and_load_roundtrip(tmp_path):
    texts = [f"msg {i}" for i in range(5)]
    embeddings = make_fake_embeddings(5, dim=8)
    idx = build_index(texts=texts, embeddings=embeddings, user1="A", user2="B")
    pkl_path = tmp_path / "test.pkl"
    save_index(idx, pkl_path)
    loaded = load_index(pkl_path)
    assert loaded.user1 == "A"
    assert loaded.user2 == "B"
    assert len(loaded.texts) == 5
    assert loaded.faiss_index.ntotal == 5
    query = make_fake_embeddings(1, dim=8)
    faiss.normalize_L2(query)
    distances, indices = loaded.faiss_index.search(query, k=1)
    assert indices.shape == (1, 1)


def test_load_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_index(Path("indexes/nonexistent.pkl"))


def test_build_index_raises_on_empty_texts():
    embeddings = make_fake_embeddings(0, dim=8)
    with pytest.raises(ValueError, match="must not be empty"):
        build_index(texts=[], embeddings=embeddings, user1="A", user2="B")


def test_build_index_raises_on_mismatched_lengths():
    texts = ["a", "b", "c"]
    embeddings = make_fake_embeddings(5, dim=8)  # 5 rows, 3 texts
    with pytest.raises(ValueError, match="must match"):
        build_index(texts=texts, embeddings=embeddings, user1="A", user2="B")
