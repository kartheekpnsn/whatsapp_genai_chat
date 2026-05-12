# Memory Layer Design

**Date:** 2026-05-12  
**Status:** Approved

## Problem

Each `/chat` request is stateless. The LLM has no awareness of what was said in the same conversation session, making responses feel disconnected across turns.

## Goal

Add a per-index, time-windowed memory layer that injects recent conversation history into the LLM context, so responses are aware of what was said earlier in the same session.

---

## Storage Layout

Each pickle file gets a sibling directory with a `memory.csv`:

```
indexes/
  chat-amma.pkl
  chat-amma/
    memory.csv
  chat-dad.pkl
  chat-dad/
    memory.csv
```

The CSV has three columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | ISO-8601 string (UTC) | When the exchange occurred |
| `query` | string | user2's message |
| `response` | string | user1's generated reply |

The directory is created lazily on first write. No CSV header is written initially — it is written on first append.

---

## `MemoryStore` (`core/memory.py`)

A single class that manages one persona's memory file.

### Constructor

```python
MemoryStore(csv_path: Path, window_minutes: int = 30)
```

- `csv_path`: path to the `memory.csv` file (e.g. `indexes/chat-amma/memory.csv`)
- `window_minutes`: session expiry threshold

### `load_recent() → list[dict]`

1. If the CSV does not exist or is empty, return `[]`.
2. Read all rows.
3. Compute the gap between `now (UTC)` and the **last row's timestamp**.
4. If the gap exceeds `window_minutes`: delete (clear) the CSV, return `[]`.
5. Otherwise: return all rows whose timestamps fall within `window_minutes` of now, preserving order (oldest first).

### `append(query: str, response: str) → None`

1. Ensure the parent directory exists.
2. Write a new row `(utcnow().isoformat(), query, response)` to the CSV.
3. If the file does not exist, write the header row first.

---

## Configuration

`MEMORY_WINDOW_MINUTES` env var (int, default `30`) controls the session window. Loaded in `main.py` at startup, passed into `MemoryStore` at construction.

---

## Startup Changes (`api/main.py`)

- Add `memory_store: MemoryStore | None = None` global alongside `index_data`, `llm_provider`, `embedding_provider`.
- In `lifespan`, after loading the index, derive the memory CSV path from the index pickle path:
  ```
  pkl_path = indexes/chat-amma.pkl
  csv_path = indexes/chat-amma/memory.csv
  ```
- Instantiate `MemoryStore(csv_path, window_minutes)` and assign to `memory_store`.

---

## Route Changes (`api/routes.py`)

The `/chat` handler gains two steps around the LLM call:

**Before LLM call:**
1. Call `memory_store.load_recent()` → list of `{timestamp, query, response}` dicts.
2. If non-empty, format as a block and append to `user_prompt` (after FAISS examples, before the final instruction):

```
Recent conversation history (most recent last):
- {user2}: <query>
  {user1}: <response>
...
```

**After LLM call:**
1. Call `memory_store.append(req.message, reply)` to persist the exchange.

A `_require_memory_store()` helper is added (mirrors existing `_require_*` helpers), though memory is non-critical — if `memory_store` is `None`, the route proceeds without memory context rather than raising 503.

---

## Data Flow

```
User message
     │
     ▼
FAISS retrieval (existing) → reply examples
     │
     ▼
MemoryStore.load_recent() → recent turns (or [] if session expired/new)
     │
     ▼
Build LLM prompt:
  system_prompt (unchanged)
  user_prompt = FAISS examples + memory block + instruction
     │
     ▼
LLM.complete()
     │
     ▼
MemoryStore.append(query, reply)
     │
     ▼
Return ChatResponse
```

---

## Session Expiry Behaviour

- Session boundary is determined by the gap between **now** and the **last stored entry's timestamp**.
- If the gap > `MEMORY_WINDOW_MINUTES`: CSV is cleared, the incoming message starts a fresh session.
- There is no explicit "session ID" — time continuity is the sole signal.

---

## Error Handling

- If the CSV is malformed/unreadable, `load_recent()` returns `[]` (logs a warning, does not crash the request).
- If `append()` fails (e.g. disk full), it logs a warning and swallows the error — memory is best-effort, not critical path.

---

## Files Changed

| File | Change |
|------|--------|
| `whatsapp_genai_chat/core/memory.py` | New — `MemoryStore` class |
| `whatsapp_genai_chat/api/main.py` | Add `memory_store` global, instantiate in `lifespan` |
| `whatsapp_genai_chat/api/routes.py` | Inject memory context before LLM call, append after |
| `.env.example` | Add `MEMORY_WINDOW_MINUTES=30` |
