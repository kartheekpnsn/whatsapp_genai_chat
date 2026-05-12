import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from whatsapp_genai_chat.core.indexer import IndexData, build_index


def make_test_index() -> IndexData:
    np.random.seed(42)
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
    # return an embedding identical to index row 0 — guaranteed FAISS hit
    first_vec = np.ones(4, dtype="float32")
    first_vec /= np.linalg.norm(first_vec)
    mock_embedding.embed.return_value = [first_vec.tolist()]

    import whatsapp_genai_chat.api.main as main_module

    with patch.object(main_module, "index_data", test_index), \
         patch.object(main_module, "llm_provider", mock_llm), \
         patch.object(main_module, "embedding_provider", mock_embedding), \
         patch.object(main_module, "memory_store", None):
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


def test_health_returns_503_when_index_not_loaded():
    import whatsapp_genai_chat.api.main as main_module
    with patch.object(main_module, "index_data", None):
        from whatsapp_genai_chat.api.main import app
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/health")
        assert response.status_code == 503


def test_chat_injects_memory_into_prompt(tmp_path):
    from datetime import datetime, timedelta, timezone
    from unittest.mock import patch, MagicMock
    import whatsapp_genai_chat.api.main as main_module
    from whatsapp_genai_chat.core.memory import MemoryStore

    test_index = make_test_index()
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Nice to hear!"
    mock_embedding = MagicMock()
    first_vec = __import__("numpy").ones(4, dtype="float32")
    first_vec /= __import__("numpy").linalg.norm(first_vec)
    mock_embedding.embed.return_value = [first_vec.tolist()]

    # pre-populate memory CSV with one recent turn
    csv_path = tmp_path / "memory.csv"
    import csv as _csv
    now = datetime.now(timezone.utc)
    with csv_path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["timestamp", "query", "response"])
        w.writeheader()
        w.writerow({
            "timestamp": (now - timedelta(minutes=5)).isoformat(),
            "query": "previous question",
            "response": "previous answer",
        })

    mock_memory = MemoryStore(csv_path, window_minutes=30)

    with patch.object(main_module, "index_data", test_index), \
         patch.object(main_module, "llm_provider", mock_llm), \
         patch.object(main_module, "embedding_provider", mock_embedding), \
         patch.object(main_module, "memory_store", mock_memory):
        from whatsapp_genai_chat.api.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.post("/chat", json={"message": "new question"})

    assert response.status_code == 200
    call_args = mock_llm.complete.call_args
    user_prompt = call_args.kwargs.get("user") or call_args.args[1]
    assert "Recent conversation history" in user_prompt
    assert "previous question" in user_prompt
    assert "previous answer" in user_prompt
