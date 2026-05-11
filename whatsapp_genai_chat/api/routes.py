import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import whatsapp_genai_chat.api.main as main_module
from whatsapp_genai_chat.core.retriever import search_and_fetch_replies
from whatsapp_genai_chat.llm.factory import get_llm_provider, get_embedding_provider

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    user1: str


@router.get("/health")
def health():
    idx = main_module.index_data
    return {"status": "ok", "user1": idx.user1, "user2": idx.user2}


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    idx = main_module.index_data
    embedding_provider = get_embedding_provider()
    llm_provider = get_llm_provider()

    query_embedding = np.array(embedding_provider.embed([req.message]), dtype="float32")
    replies = search_and_fetch_replies(idx, query_embedding, k=5)

    if not replies:
        raise HTTPException(status_code=500, detail="No relevant responses found in index.")

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
    return ChatResponse(reply=reply, user1=idx.user1)
