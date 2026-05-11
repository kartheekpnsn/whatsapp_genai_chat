# WhatsApp GenAI Chat

Chat with an AI that responds like one participant from a WhatsApp export.

The app parses a two-person WhatsApp `.txt` export, builds a local FAISS index over one participant's messages, retrieves similar historical turns for a new message, and asks an LLM to synthesize a reply in the selected participant's style. It includes a FastAPI backend and a small WhatsApp-like Vite frontend.

## Features

- Parse WhatsApp text exports with `whatsappchattodf`
- Select which of the two chat participants the bot should imitate
- Build and persist a local FAISS vector index as a `.pkl` file
- Retrieve semantically similar messages and the selected participant's follow-up replies
- Generate persona-style responses with Azure OpenAI, OpenAI, Google Gemini, or Anthropic LLMs
- Run a local FastAPI API on port `8003`
- Run a Vite frontend on port `5174`
- Use one `Makefile` to build indexes and run the full app

## How It Works

1. A WhatsApp export is parsed into a dataframe with normalized `sender` and `message` columns.
2. During indexing, you choose the participant to simulate.
3. The app embeds the other participant's messages and stores them in a FAISS index.
4. At chat time, your message is embedded and searched against that index.
5. For the top matches, the app fetches the simulated participant's immediate replies from the original chat.
6. The configured LLM receives those replies as style examples and writes a new response.

In the code, `user1` is the participant being simulated by the bot, and `user2` is the human user.

## Project Structure

```text
.
├── frontend/                       # Vite frontend
│   ├── index.html
│   ├── main.js
│   └── style.css
├── scripts/
│   └── build_index.py              # Interactive FAISS index builder
├── tests/                          # Unit and API tests
├── whatsapp_genai_chat/
│   ├── api/                        # FastAPI app and routes
│   ├── core/                       # Parser, indexer, retriever
│   └── llm/                        # Provider interfaces and implementations
├── data/                           # Local WhatsApp exports
├── indexes/                        # Generated FAISS pickle indexes
├── Makefile
├── pyproject.toml
└── REQUIREMENT.md
```

`data/*.txt`, generated indexes, `.env`, and other local artifacts are ignored by git so private chats and credentials do not get committed.

## Requirements

- Python `>=3.12`
- `uv` for Python environment and package management
- Node.js and npm for the Vite frontend
- A two-person WhatsApp chat export as `.txt`
- Access to at least one supported LLM and embedding provider

Python 3.12 is recommended for the smoothest FAISS wheel availability.

## Setup

Install Python dependencies:

```bash
uv sync --extra dev
```

Install frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

Create your local environment file:

```bash
cp .env.example .env
```

Then edit `.env` for the provider you want to use.

## Provider Configuration

Select the active provider with:

```env
PROVIDER=azure_openai
```

Supported values:

| Provider | LLM | Embeddings | Notes |
| --- | --- | --- | --- |
| `azure_openai` | Yes | Yes | Uses `DefaultAzureCredential`; no API key required |
| `openai` | Yes | Yes | Uses `OPENAI_API_KEY` |
| `google` | Yes | Yes | Uses `GOOGLE_API_KEY` |
| `anthropic` | Yes | No | Anthropic embeddings are not implemented |

Because the current factory uses the same `PROVIDER` for both LLMs and embeddings, `PROVIDER=anthropic` cannot build an index or answer chat requests by itself.

### Azure OpenAI

```env
PROVIDER=azure_openai
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_VERSION=2024-02-01
AZURE_LLM_DEPLOYMENT_NAME=gpt-4.1
AZURE_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large
```

Azure authentication uses `DefaultAzureCredential`, so authenticate locally with a supported Azure identity method, for example:

```bash
az login
```

### OpenAI

```env
PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

### Google

```env
PROVIDER=google
GOOGLE_API_KEY=...
GOOGLE_MODEL=gemini-2.0-flash
GOOGLE_EMBEDDING_MODEL=models/text-embedding-004
```

### Anthropic

```env
PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-sonnet-4-6
ANTHROPIC_MAX_TOKENS=2048
```

Anthropic can generate replies, but this project does not currently provide a separate embedding-provider setting. Use `azure_openai`, `openai`, or `google` when building indexes or running the app.

## Prepare a WhatsApp Export

Export a two-person chat from WhatsApp as a `.txt` file and place it under `data/`.

Example:

```text
data/chat-sample.txt
```

The parser expects `whatsappchattodf` to return at least these columns:

- `User`
- `Message`

They are normalized internally to:

- `sender`
- `message`

## Build an Index

Run:

```bash
make index FILE=data/chat-sample.txt
```

The script will:

1. Parse the chat export.
2. Confirm there are exactly two senders.
3. Ask which user the bot should simulate.
4. Embed the other user's messages in batches.
5. Save the generated index to `indexes/<chat-file-stem>.pkl`.

If more than one `.pkl` exists in `indexes/`, set `INDEX_PATH` in `.env`:

```env
INDEX_PATH=/absolute/path/to/indexes/chat-sample.pkl
```

Relative paths work too, but absolute paths are clearer when switching between multiple indexes.

## Run the App

Start backend and frontend together:

```bash
make dev
```

To run the app with a specific index for this session, pass `FILE`:

```bash
make dev FILE=indexes/chat-sample.pkl
```

This starts:

- FastAPI backend: `http://localhost:8003`
- Vite frontend: `http://localhost:5174`

Run them separately if needed:

```bash
make backend
make frontend
```

Open the frontend at:

```text
http://localhost:5174
```

## API

### `GET /health`

Returns the loaded index status and participant names.

Example response:

```json
{
  "status": "ok",
  "user1": "Alice",
  "user2": "Bob"
}
```

### `POST /chat`

Request:

```json
{
  "message": "hello there"
}
```

Response:

```json
{
  "reply": "Hey! I'm good.",
  "user1": "Alice"
}
```

The `message` field is required and is limited to 2,000 characters.

## Make Targets

```bash
make help
```

Available targets:

| Target | Description |
| --- | --- |
| `make dev` | Run backend and frontend together |
| `make dev FILE=indexes/chat-sample.pkl` | Run backend and frontend with a specific index for this session |
| `make backend` | Run FastAPI on port `8003` |
| `make frontend` | Run Vite on port `5174` |
| `make index FILE=data/chat-sample.txt` | Build a FAISS index from a WhatsApp export |
| `make help` | Show available targets |

## Testing

Run the test suite:

```bash
uv run pytest
```

The tests cover:

- WhatsApp parsing
- FAISS index construction and persistence
- Reply retrieval from matched chat turns
- FastAPI route behavior
- Provider factory behavior

## Troubleshooting

### No index found

If the backend fails with:

```text
No index found in indexes/
```

build an index first:

```bash
make index FILE=data/chat-sample.txt
```

### Multiple indexes found

If there are multiple `.pkl` files in `indexes/`, set:

```env
INDEX_PATH=/absolute/path/to/the-index.pkl
```

Or pass the index directly when running the full app:

```bash
make dev FILE=indexes/chat-sample.pkl
```

### Backend offline in the frontend

The frontend calls `http://localhost:8003`. Make sure the backend is running:

```bash
make backend
```

### CORS issues

By default, the backend allows `http://localhost:5174`. Override with:

```env
CORS_ORIGINS=http://localhost:5174,http://127.0.0.1:5174
```

### Azure authentication errors

Make sure you are logged in and have access to the Azure OpenAI resource:

```bash
az login
```

Also confirm that deployment names in `.env` match the deployments in Azure OpenAI.

### Missing FAISS or parser dependencies

Install or refresh the Python environment:

```bash
uv sync --extra dev
```

If FAISS wheels are unavailable for your Python version or platform, try Python 3.12.

## Privacy Notes

WhatsApp exports are private personal data. Keep raw chat files in `data/`, keep generated indexes in `indexes/`, and do not commit `.env`, `.txt`, `.pkl`, or generated local artifacts.

The app sends retrieved message examples and your current prompt to the configured LLM provider. Review your provider's data handling policy before using real personal chats.

## License

MIT. See [LICENSE](LICENSE).
