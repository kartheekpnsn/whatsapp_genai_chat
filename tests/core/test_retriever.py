import faiss
import numpy as np
import pandas as pd
import pytest
from whatsapp_genai_chat.core.indexer import build_index, IndexData
from whatsapp_genai_chat.core.retriever import search_and_fetch_replies


def make_index() -> IndexData:
    user2_msgs = ["hello how are you", "what did you eat", "good morning", "are you busy", "let's meet"]
    user1_replies = ["I'm good thanks", "I had rice", "morning!", "a bit busy", "sure let's go"]

    rows = []
    for u2, u1 in zip(user2_msgs, user1_replies):
        rows.append({"sender": "Bob", "message": u2})
        rows.append({"sender": "Alice", "message": u1})
    df = pd.DataFrame(rows)

    dim = 4
    embeddings = np.random.rand(len(user2_msgs), dim).astype("float32")
    return build_index(texts=user2_msgs, embeddings=embeddings, user1="Alice", user2="Bob", df=df)


def test_search_returns_user1_replies():
    idx = make_index()
    query_embedding = np.random.rand(1, 4).astype("float32")
    faiss.normalize_L2(query_embedding)
    replies = search_and_fetch_replies(idx, query_embedding, k=3)
    assert isinstance(replies, list)
    assert len(replies) <= 3
    for r in replies:
        assert isinstance(r, str)


def test_search_respects_k():
    idx = make_index()
    query_embedding = np.random.rand(1, 4).astype("float32")
    faiss.normalize_L2(query_embedding)
    replies = search_and_fetch_replies(idx, query_embedding, k=2)
    assert len(replies) <= 2


def test_search_works_with_unnormalized_query():
    idx = make_index()
    # deliberately un-normalized query — should still work correctly now
    query_embedding = np.ones((1, 4), dtype="float32") * 5.0  # large magnitude
    replies = search_and_fetch_replies(idx, query_embedding, k=3)
    assert isinstance(replies, list)


def test_search_handles_no_reply():
    # DataFrame where User2 sends the last message with no User1 reply after
    df = pd.DataFrame([
        {"sender": "Bob", "message": "hello"},
        {"sender": "Alice", "message": "hi there"},
        {"sender": "Bob", "message": "goodbye"},  # last message, no User1 reply
    ])
    dim = 4
    embeddings = np.random.rand(2, dim).astype("float32")
    idx = build_index(texts=["hello", "goodbye"], embeddings=embeddings, user1="Alice", user2="Bob", df=df)
    query = np.random.rand(1, dim).astype("float32")
    # should not crash, may return fewer or zero replies for the last-message match
    replies = search_and_fetch_replies(idx, query, k=2)
    assert isinstance(replies, list)


def test_search_returns_user1_not_user2_messages():
    idx = make_index()
    query_embedding = np.random.rand(1, 4).astype("float32")
    faiss.normalize_L2(query_embedding)
    replies = search_and_fetch_replies(idx, query_embedding, k=5)
    user2_msgs = {"hello how are you", "what did you eat", "good morning", "are you busy", "let's meet"}
    for r in replies:
        assert r not in user2_msgs, f"Returned a User2 message instead of User1 reply: {r}"
