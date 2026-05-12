import csv
import pytest
from pathlib import Path
from whatsapp_genai_chat.core.memory import MemoryStore


def test_append_creates_csv_with_header(tmp_path):
    csv_path = tmp_path / "persona" / "memory.csv"
    store = MemoryStore(csv_path, window_minutes=30)
    store.append("hello", "hey!")

    assert csv_path.exists()
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["query"] == "hello"
    assert rows[0]["response"] == "hey!"
    assert "timestamp" in rows[0]


def test_append_multiple_rows(tmp_path):
    csv_path = tmp_path / "memory.csv"
    store = MemoryStore(csv_path, window_minutes=30)
    store.append("q1", "r1")
    store.append("q2", "r2")

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[1]["query"] == "q2"
