import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import whatsapp_genai_chat.api.main as main_module
from whatsapp_genai_chat.core.indexer import IndexData
from whatsapp_genai_chat.core.retriever import search_and_fetch_replies

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    reply: str
    user1: str


def _require_index() -> IndexData:
    idx = main_module.index_data
    if idx is None:
        raise HTTPException(status_code=503, detail="Index not loaded. Run: make index FILE=data/chat-sample.txt")
    return idx


def _require_embedding_provider():
    ep = main_module.embedding_provider
    if ep is None:
        raise HTTPException(status_code=503, detail="Embedding provider not initialized.")
    return ep


def _require_llm_provider():
    lp = main_module.llm_provider
    if lp is None:
        raise HTTPException(status_code=503, detail="LLM provider not initialized.")
    return lp


@router.get("/health")
def health():
    idx = _require_index()
    return {"status": "ok", "user1": idx.user1, "user2": idx.user2}


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    idx = _require_index()

    try:
        embedding_provider = _require_embedding_provider()
        raw = embedding_provider.embed([req.message])
        query_embedding = np.array(raw, dtype="float32").reshape(1, -1)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Embedding provider error: {e}")

    replies = search_and_fetch_replies(idx, query_embedding, k=5)

    if not replies:
        raise HTTPException(status_code=500, detail="No relevant responses found in index.")

    try:
        llm_provider = _require_llm_provider()
        system_prompt = (
            f"You are {idx.user1}. Based on how {idx.user1} has responded in the past, "
            f"reply in their exact style — including language mix (e.g. Telugu+English), tone, emoji usage, "
            f"and message length. Only use the provided example responses as style reference."
        )
        user_prompt = (
            f"Here are example responses from {idx.user1}:\n"
            + "\n".join(f"- {r}" for r in replies)
            + f"\n\nNow respond to this message from {idx.user2}: {req.message}"
        )
        reply = llm_provider.complete(system=system_prompt, user=user_prompt)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM provider error: {e}")

    return ChatResponse(reply=reply, user1=idx.user1)
