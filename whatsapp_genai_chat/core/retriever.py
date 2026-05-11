import logging
import faiss
import numpy as np
from whatsapp_genai_chat.core.indexer import IndexData

logger = logging.getLogger(__name__)


def search_and_fetch_replies(index_data: IndexData, query_embedding: np.ndarray, k: int = 5) -> list[str]:
    query = query_embedding.copy()
    faiss.normalize_L2(query)
    _, indices = index_data.faiss_index.search(query, k)

    df = index_data.df
    user1 = index_data.user1
    row_indices = index_data.row_indices

    replies: list[str] = []
    seen: set[int] = set()

    logger.info("Top-%d FAISS indices returned: %s", k, indices[0].tolist())

    for idx in indices[0]:
        if idx < 0 or idx >= len(row_indices):
            continue
        iloc_pos = row_indices[idx]
        if iloc_pos in seen:
            continue
        seen.add(iloc_pos)
        matched_msg = df.iloc[iloc_pos]["message"]
        logger.info("Matched message [iloc=%d]: %s", iloc_pos, matched_msg)
        # advance past any remaining user2 messages to find the start of user1's reply
        user2 = index_data.user2
        j = iloc_pos + 1
        while j < len(df) and df.iloc[j]["sender"] == user2:
            j += 1
        # collect consecutive user1 messages
        while j < len(df) and df.iloc[j]["sender"] == user1:
            replies.append(str(df.iloc[j]["message"]))
            j += 1

    logger.info("Fetched %d reply candidates: %s", len(replies), replies)
    return replies
