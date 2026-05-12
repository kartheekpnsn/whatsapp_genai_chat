# Memory Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-index, time-windowed CSV memory layer so the LLM has context of recent conversation turns within the same session.

**Architecture:** A new `MemoryStore` class in `core/memory.py` manages reading/writing a `memory.csv` file co-located with each `.pkl` index. The `/chat` route loads recent turns before the LLM call and appends the exchange after. Session expiry is determined by the time gap between now and the last stored entry — if it exceeds `MEMORY_WINDOW_MINUTES`, the CSV is cleared.

**Tech Stack:** Python stdlib (`csv`, `pathlib`, `datetime`), FastAPI, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `whatsapp_genai_chat/core/memory.py` | Create | `MemoryStore` class — read/write/expiry logic |
| `tests/core/test_memory.py` | Create | Unit tests for `MemoryStore` |
| `whatsapp_genai_chat/api/main.py` | Modify | Add `memory_store` global, instantiate in `lifespan` |
| `whatsapp_genai_chat/api/routes.py` | Modify | Inject memory into LLM prompt, append after reply |
| `tests/api/test_routes.py` | Modify | Patch `memory_store` in existing route tests + new memory injection test |
| `.env.example` | Modify | Document `MEMORY_WINDOW_MINUTES=30` |

---

## Task 1: Create `MemoryStore` with `append`

**Files:**
- Create: `whatsapp_genai_chat/core/memory.py`
- Create: `tests/core/test_memory.py`

- [ ] **Step 1: Write the failing test for `append`**

Create `tests/core/test_memory.py`:

```python
import csv
import pytest
from pathlib import Path
from whatsapp_genai_chat.core.memory import MemoryStore


def test_append_creates_csv_with_header(tmp_path):
    csv_path = tmp_path / "persona" / "memory.csv"
    store = MemoryStore(csv_path, window_minutes=30)
    store.append("hello", "hey!")

    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    assert rows[0]["query"] == "hello"
    assert rows[0]["response"] == "hey!"
    assert "timestamp" in rows[0]


def test_append_multiple_rows(tmp_path):
    csv_path = tmp_path / "memory.csv"
    store = MemoryStore(csv_path, window_minutes=30)
    store.append("q1", "r1")
    store.append("q2", "r2")

    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 2
    assert rows[1]["query"] == "q2"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/kartheek/Documents/kartheek/Work/Engagements/Personal/whatsapp_genai_chat
.venv/bin/pytest tests/core/test_memory.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `memory.py` does not exist yet.

- [ ] **Step 3: Implement `MemoryStore` with `append`**

Create `whatsapp_genai_chat/core/memory.py`:

```python
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_FIELDS = ["timestamp", "query", "response"]


class MemoryStore:
    def __init__(self, csv_path: Path, window_minutes: int = 30) -> None:
        self.csv_path = Path(csv_path)
        self.window_minutes = window_minutes

    def append(self, query: str, response: str) -> None:
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self.csv_path.exists()
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "query": query,
                    "response": response,
                })
        except Exception:
            logger.warning("MemoryStore.append failed", exc_info=True)

    def load_recent(self) -> list[dict]:
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/core/test_memory.py -v
```

Expected: Both `test_append_*` tests PASS.

- [ ] **Step 5: Commit**

```bash
git add whatsapp_genai_chat/core/memory.py tests/core/test_memory.py
git commit -m "feat: add MemoryStore with append"
```

---

## Task 2: Implement `load_recent` with session expiry

**Files:**
- Modify: `whatsapp_genai_chat/core/memory.py`
- Modify: `tests/core/test_memory.py`

- [ ] **Step 1: Write failing tests for `load_recent`**

Append to `tests/core/test_memory.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/core/test_memory.py -v -k "load_recent"
```

Expected: All 5 `test_load_recent_*` tests FAIL (returns `[]` always, logic not implemented).

- [ ] **Step 3: Implement `load_recent`**

Replace the `load_recent` stub in `whatsapp_genai_chat/core/memory.py`:

```python
def load_recent(self) -> list[dict]:
    if not self.csv_path.exists():
        return []
    try:
        with self.csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return []
        now = datetime.now(timezone.utc)
        last_ts = datetime.fromisoformat(rows[-1]["timestamp"])
        if (now - last_ts).total_seconds() > self.window_minutes * 60:
            self.csv_path.unlink(missing_ok=True)
            return []
        cutoff = now.timestamp() - self.window_minutes * 60
        return [
            r for r in rows
            if datetime.fromisoformat(r["timestamp"]).timestamp() >= cutoff
        ]
    except Exception:
        logger.warning("MemoryStore.load_recent failed", exc_info=True)
        return []
```

- [ ] **Step 4: Run all memory tests**

```bash
.venv/bin/pytest tests/core/test_memory.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add whatsapp_genai_chat/core/memory.py tests/core/test_memory.py
git commit -m "feat: implement MemoryStore.load_recent with session expiry"
```

---

## Task 3: Wire `MemoryStore` into app startup

**Files:**
- Modify: `whatsapp_genai_chat/api/main.py`

- [ ] **Step 1: Add `memory_store` global and instantiate in `lifespan`**

Open `whatsapp_genai_chat/api/main.py`. Make these changes:

Add import after the existing imports:
```python
from whatsapp_genai_chat.core.memory import MemoryStore
```

Add the global declaration (after `embedding_provider: EmbeddingProvider | None = None`):
```python
memory_store: MemoryStore | None = None
```

In the `lifespan` function, add after `index_data = load_index(index_path)`:
```python
window_minutes = int(os.environ.get("MEMORY_WINDOW_MINUTES", "30"))
memory_csv = index_path.with_suffix("") / "memory.csv"
memory_store = MemoryStore(memory_csv, window_minutes=window_minutes)
```

Also add `memory_store` to the `global` declaration line at the top of `lifespan`:
```python
global index_data, llm_provider, embedding_provider, memory_store
```

- [ ] **Step 2: Verify the app still starts cleanly**

```bash
cd /Users/kartheek/Documents/kartheek/Work/Engagements/Personal/whatsapp_genai_chat
.venv/bin/pytest tests/ -v
```

Expected: All existing tests PASS (no regressions).

- [ ] **Step 3: Commit**

```bash
git add whatsapp_genai_chat/api/main.py
git commit -m "feat: instantiate MemoryStore in app lifespan"
```

---

## Task 4: Inject memory into `/chat` route

**Files:**
- Modify: `whatsapp_genai_chat/api/routes.py`
- Modify: `tests/api/test_routes.py`

- [ ] **Step 1: Write failing test for memory injection**

Append to `tests/api/test_routes.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/api/test_routes.py::test_chat_injects_memory_into_prompt -v
```

Expected: FAIL — memory block not yet in the prompt.

- [ ] **Step 3: Update `routes.py` to add memory helper and inject context**

Open `whatsapp_genai_chat/api/routes.py`.

Add import at top (after existing imports):
```python
import whatsapp_genai_chat.api.main as main_module
```
(already present — skip if so)

Add helper function after `_require_llm_provider`:
```python
def _get_memory_store():
    return main_module.memory_store  # may be None — memory is non-critical
```

In the `chat` handler, add memory loading right after `replies = search_and_fetch_replies(...)`:

```python
recent_turns = []
ms = _get_memory_store()
if ms is not None:
    try:
        recent_turns = ms.load_recent()
    except Exception:
        logger.warning("Failed to load memory", exc_info=True)
```

Build the memory block and update `user_prompt`. Replace the existing `user_prompt` assignment:

```python
memory_block = ""
if recent_turns:
    lines = []
    for turn in recent_turns:
        lines.append(f"- {idx.user2}: {turn['query']}")
        lines.append(f"  {idx.user1}: {turn['response']}")
    memory_block = "\nRecent conversation history (most recent last):\n" + "\n".join(lines) + "\n"

user_prompt = (
    f"Here are past responses from {idx.user1} in similar contexts:\n"
    + "\n".join(f"- {r}" for r in replies)
    + memory_block
    + f"\n\nUsing only the above responses as your answer source, reply to this message from {idx.user2}: {req.message}"
)
```

After `reply = llm_provider.complete(...)`, add memory append:

```python
ms = _get_memory_store()
if ms is not None:
    try:
        ms.append(req.message, reply)
    except Exception:
        logger.warning("Failed to append to memory", exc_info=True)
```

- [ ] **Step 4: Run all route tests**

```bash
.venv/bin/pytest tests/api/test_routes.py -v
```

Expected: All tests PASS including the new `test_chat_injects_memory_into_prompt`.

- [ ] **Step 5: Update existing route test fixture to patch `memory_store`**

The existing `client` fixture in `tests/api/test_routes.py` patches `index_data`, `llm_provider`, and `embedding_provider` but not `memory_store`. This means it uses whatever `memory_store` is set globally — which may write real files. Patch it to `None` so existing tests are isolated:

Find the `client` fixture and add `patch.object(main_module, "memory_store", None)` to the context manager:

```python
@pytest.fixture
def client():
    test_index = make_test_index()
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Hey! I'm good."
    mock_embedding = MagicMock()
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
```

- [ ] **Step 6: Run full test suite**

```bash
.venv/bin/pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add whatsapp_genai_chat/api/routes.py tests/api/test_routes.py
git commit -m "feat: inject memory context into LLM prompt in /chat route"
```

---

## Task 5: Document env var in `.env.example`

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Add `MEMORY_WINDOW_MINUTES` to `.env.example`**

Open `.env.example` and append:

```
# Minutes of inactivity after which the conversation memory is cleared (default: 30)
MEMORY_WINDOW_MINUTES=30
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: document MEMORY_WINDOW_MINUTES env var"
```

---

## Task 6: Final verification

- [ ] **Step 1: Run full test suite**

```bash
.venv/bin/pytest tests/ -v
```

Expected: All tests PASS with no warnings about unpatched globals.

- [ ] **Step 2: Verify memory folder is gitignored**

```bash
grep -n "indexes/" .gitignore 2>/dev/null || echo "not ignored"
```

If `indexes/` or `*.csv` is not already gitignored, add to `.gitignore`:

```
indexes/*/memory.csv
```

Then commit:

```bash
git add .gitignore
git commit -m "chore: gitignore per-index memory CSV files"
```
