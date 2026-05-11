import pytest
from pathlib import Path
from whatsapp_genai_chat.core.parser import parse_chat

SAMPLE = Path(__file__).parent.parent.parent / "data" / "chat-sample.txt"

def test_parse_returns_dataframe():
    df = parse_chat(SAMPLE)
    assert not df.empty

def test_parse_has_required_columns():
    df = parse_chat(SAMPLE)
    assert "sender" in df.columns
    assert "message" in df.columns

def test_parse_extracts_senders():
    df = parse_chat(SAMPLE)
    assert df["sender"].nunique() >= 1

def test_parse_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        parse_chat(Path("data/nonexistent.txt"))
