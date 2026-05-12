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


from datetime import datetime, timedelta, timezone


def _write_rows(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "query", "response"])
        writer.writeheader()
        writer.writerows(rows)


def test_load_recent_returns_empty_when_no_file(tmp_path):
    store = MemoryStore(tmp_path / "memory.csv", window_minutes=30)
    assert store.load_recent() == []


def test_load_recent_returns_rows_within_window(tmp_path):
    csv_path = tmp_path / "memory.csv"
    now = datetime.now(timezone.utc)
    rows = [
        {"timestamp": (now - timedelta(minutes=10)).isoformat(), "query": "q1", "response": "r1"},
        {"timestamp": (now - timedelta(minutes=5)).isoformat(), "query": "q2", "response": "r2"},
    ]
    _write_rows(csv_path, rows)

    store = MemoryStore(csv_path, window_minutes=30)
    result = store.load_recent()
    assert len(result) == 2
    assert result[0]["query"] == "q1"
    assert result[1]["query"] == "q2"


def test_load_recent_clears_csv_and_returns_empty_when_session_expired(tmp_path):
    csv_path = tmp_path / "memory.csv"
    now = datetime.now(timezone.utc)
    rows = [
        {"timestamp": (now - timedelta(minutes=60)).isoformat(), "query": "old", "response": "stale"},
    ]
    _write_rows(csv_path, rows)

    store = MemoryStore(csv_path, window_minutes=30)
    result = store.load_recent()

    assert result == []
    assert not csv_path.exists()


def test_load_recent_excludes_rows_outside_window(tmp_path):
    csv_path = tmp_path / "memory.csv"
    now = datetime.now(timezone.utc)
    rows = [
        {"timestamp": (now - timedelta(minutes=45)).isoformat(), "query": "old", "response": "stale"},
        {"timestamp": (now - timedelta(minutes=10)).isoformat(), "query": "recent", "response": "fresh"},
    ]
    _write_rows(csv_path, rows)

    store = MemoryStore(csv_path, window_minutes=30)
    result = store.load_recent()
    assert len(result) == 1
    assert result[0]["query"] == "recent"


def test_load_recent_returns_empty_on_malformed_csv(tmp_path):
    csv_path = tmp_path / "memory.csv"
    csv_path.write_text("not,valid,csv\nbad data here")
    store = MemoryStore(csv_path, window_minutes=30)
    assert store.load_recent() == []
