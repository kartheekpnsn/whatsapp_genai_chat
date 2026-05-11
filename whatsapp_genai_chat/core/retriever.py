import faiss
import numpy as np
from whatsapp_genai_chat.core.indexer import IndexData


def search_and_fetch_replies(index_data: IndexData, query_embedding: np.ndarray, k: int = 5) -> list[str]:
    # query_embedding must already be L2-normalized (caller's responsibility)
    _, indices = index_data.faiss_index.search(query_embedding, k)
    df = index_data.df
    user1 = index_data.user1
    user2_texts = index_data.texts

    replies = []
    seen_positions = set()

    for idx in indices[0]:
        if idx < 0 or idx >= len(user2_texts):
            continue
        matched_msg = user2_texts[idx]
        mask = (df["sender"] == index_data.user2) & (df["message"] == matched_msg)
        matching_rows = df[mask]
        if matching_rows.empty:
            continue
        row_pos = matching_rows.index[0]
        if row_pos in seen_positions:
            continue
        seen_positions.add(row_pos)
        # collect consecutive User1 messages immediately after this User2 message
        j = row_pos + 1
        while j < len(df) and df.loc[j, "sender"] == user1:
            replies.append(str(df.loc[j, "message"]))
            j += 1

    return replies
