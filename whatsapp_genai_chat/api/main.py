import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from whatsapp_genai_chat.core.indexer import IndexData, load_index

load_dotenv()

index_data: IndexData = None  # populated at startup


def _resolve_index_path() -> Path:
    explicit = os.environ.get("INDEX_PATH", "").strip()
    if explicit:
        return Path(explicit)
    # anchor to project root (this file is at whatsapp_genai_chat/api/main.py)
    indexes_dir = Path(__file__).parent.parent.parent / "indexes"
    pkls = list(indexes_dir.glob("*.pkl"))
    if not pkls:
        raise RuntimeError(
            "No index found in indexes/. Run: make index FILE=data/chat-sample.txt"
        )
    if len(pkls) > 1:
        raise RuntimeError(
            f"Multiple indexes found: {[p.name for p in pkls]}. "
            "Set INDEX_PATH in .env to specify which one to use."
        )
    return pkls[0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global index_data
    index_path = _resolve_index_path()
    index_data = load_index(index_path)
    print(f"Loaded index: {index_path} | user1={index_data.user1} | user2={index_data.user2}")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from whatsapp_genai_chat.api import routes  # noqa: E402
app.include_router(routes.router)
