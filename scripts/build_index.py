#!/usr/bin/env python3
"""
Build a FAISS index from a WhatsApp chat export.

Usage:
    uv run scripts/build_index.py data/chat-sample.txt
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from whatsapp_genai_chat.core.parser import parse_chat
from whatsapp_genai_chat.core.indexer import build_index, save_index
from whatsapp_genai_chat.llm.factory import get_embedding_provider


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run scripts/build_index.py <path_to_chat.txt>")
        sys.exit(1)

    chat_path = Path(sys.argv[1])
    print(f"Parsing {chat_path}...")
    df = parse_chat(chat_path)

    senders = df["sender"].dropna().unique().tolist()
    if len(senders) != 2:
        print(f"Error: expected 2 users, found {len(senders)}: {senders}")
        sys.exit(1)

    print("\nFound 2 users:")
    for i, name in enumerate(senders, 1):
        print(f"  {i}. {name}")

    while True:
        choice = input("\nSelect the user to simulate (1 or 2): ").strip()
        if choice in ("1", "2"):
            break
        print("Please enter 1 or 2.")

    user1 = senders[int(choice) - 1]
    user2 = senders[0] if choice == "2" else senders[1]
    print(f"\nBot will simulate: {user1}")
    print(f"You will act as: {user2}")

    user2_msgs = df[df["sender"] == user2]["message"].dropna().tolist()
    print(f"\nEmbedding {len(user2_msgs)} messages from {user2}...")

    embedding_provider = get_embedding_provider()
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(user2_msgs), batch_size):
        batch = user2_msgs[i:i + batch_size]
        embeddings = embedding_provider.embed(batch)
        all_embeddings.extend(embeddings)
        print(f"  Embedded {min(i + batch_size, len(user2_msgs))}/{len(user2_msgs)}")

    embeddings_array = np.array(all_embeddings, dtype="float32")

    index_data = build_index(
        texts=user2_msgs,
        embeddings=embeddings_array,
        user1=user1,
        user2=user2,
        df=df,
    )

    output_path = Path("indexes") / f"{chat_path.stem}.pkl"
    save_index(index_data, output_path)
    print(f"\nIndex saved to {output_path}")


if __name__ == "__main__":
    main()
