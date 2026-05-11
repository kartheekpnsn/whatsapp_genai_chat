import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from whatsapp_genai_chat.core.indexer import IndexData, build_index


def make_test_index() -> IndexData:
    user2_msgs = ["hello", "how are you", "what did you eat"]
    user1_replies = ["hey!", "I'm good", "I had rice"]
    rows = []
    for u2, u1 in zip(user2_msgs, user1_replies):
        rows.append({"sender": "Bob", "message": u2})
        rows.append({"sender": "Alice", "message": u1})
    df = pd.DataFrame(rows)
    embeddings = np.random.rand(3, 4).astype("float32")
    return build_index(texts=user2_msgs, embeddings=embeddings, user1="Alice", user2="Bob", df=df)


@pytest.fixture
def client():
    test_index = make_test_index()
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Hey! I'm good."
    mock_embedding = MagicMock()
    mock_embedding.embed.return_value = [np.random.rand(4).tolist()]

    import whatsapp_genai_chat.api.main as main_module
    import whatsapp_genai_chat.api.routes as routes_module

    with patch.object(main_module, "index_data", test_index), \
         patch.object(routes_module, "get_llm_provider", return_value=mock_llm), \
         patch.object(routes_module, "get_embedding_provider", return_value=mock_embedding):
        from whatsapp_genai_chat.api.main import app
        yield TestClient(app)


def test_health_returns_user_names(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["user1"] == "Alice"
    assert data["user2"] == "Bob"


def test_chat_returns_reply(client):
    response = client.post("/chat", json={"message": "hello there"})
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "user1" in data
    assert data["user1"] == "Alice"


def test_chat_requires_message_field(client):
    response = client.post("/chat", json={})
    assert response.status_code == 422
