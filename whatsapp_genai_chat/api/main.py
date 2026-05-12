import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from whatsapp_genai_chat.core.indexer import IndexData, load_index
from whatsapp_genai_chat.llm.base import LLMProvider, EmbeddingProvider
from whatsapp_genai_chat.core.memory import MemoryStore

load_dotenv()

index_data: IndexData | None = None        # populated at startup
llm_provider: LLMProvider | None = None    # populated at startup
embedding_provider: EmbeddingProvider | None = None  # populated at startup
memory_store: MemoryStore | None = None


def _resolve_index_path() -> Path:
    explicit = os.environ.get("INDEX_PATH", "").strip()
    indexes_dir = Path(__file__).parent.parent.parent / "indexes"
    if explicit:
        resolved = Path(explicit).resolve()
        # security: restrict to project indexes dir unless absolute path given explicitly
        return resolved
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
    global index_data, llm_provider, embedding_provider, memory_store
    from whatsapp_genai_chat.llm.factory import get_llm_provider, get_embedding_provider
    index_path = _resolve_index_path()
    index_data = load_index(index_path)
    window_minutes = int(os.environ.get("MEMORY_WINDOW_MINUTES", "30"))
    memory_csv = index_path.with_suffix("") / "memory.csv"
    memory_store = MemoryStore(memory_csv, window_minutes=window_minutes)
    llm_provider = get_llm_provider()
    embedding_provider = get_embedding_provider()
    print(f"Loaded index: {index_path} | user1={index_data.user1} | user2={index_data.user2}")
    yield


app = FastAPI(lifespan=lifespan)

cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:5174").split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

from whatsapp_genai_chat.api import routes  # noqa: E402
app.include_router(routes.router)
