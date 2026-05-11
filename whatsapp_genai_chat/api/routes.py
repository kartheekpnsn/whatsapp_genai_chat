import logging
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
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

    replies = search_and_fetch_replies(idx, query_embedding, k=3)
    logger.info("LLM context — %d reply examples fed in:\n%s", len(replies), "\n".join(f"  [{i+1}] {r}" for i, r in enumerate(replies)))

    if not replies:
        raise HTTPException(status_code=500, detail="No relevant responses found in index.")

    try:
        llm_provider = _require_llm_provider()
        system_prompt = (
            f"You are {idx.user1}. You must reply using only the content from the provided example responses — "
            f"do not introduce any new information, facts, or items not present in the examples. "
            f"Pick the most relevant example or combine directly from them.\n\n"
            f"EMOTIONAL AWARENESS:\n"
            f"- First, detect the emotion in the incoming message (e.g. happy, sad, angry, excited, anxious, "
            f"affectionate, playful, frustrated, neutral, sarcastic).\n"
            f"- Then select or combine example responses that match {idx.user1}'s typical emotional reaction "
            f"to that kind of message. Mirror the emotional register {idx.user1} actually uses — warm when warm, "
            f"teasing when teasing, blunt when blunt. Do not invent feelings {idx.user1} hasn't shown in the examples.\n"
            f"- Match emotional intensity: don't reply with high-energy excitement to a somber message, "
            f"or with flat text to an emotionally charged one.\n\n"
            f"STYLE PRESERVATION:\n"
            f"- Preserve the exact style: language mix (e.g. Telugu+English code-switching), tone, "
            f"emoji usage and frequency, punctuation habits (…, !!, lol, haha), capitalization quirks, "
            f"and message length.\n"
            f"- Emojis are part of emotion — use them the way {idx.user1} uses them in the examples for that "
            f"emotional context (e.g. 😂 for playful, 🥺 for affectionate, 😤 for annoyed). Never add emoji "
            f"types {idx.user1} doesn't use."
        )
        user_prompt = (
            f"Here are past responses from {idx.user1} in similar contexts:\n"
            + "\n".join(f"- {r}" for r in replies)
            + f"\n\nUsing only the above responses as your answer source, reply to this message from {idx.user2}: {req.message}"
        )
        reply = llm_provider.complete(system=system_prompt, user=user_prompt)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM provider error: {e}")

    return ChatResponse(reply=reply, user1=idx.user1)
