import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd


@dataclass
class IndexData:
    faiss_index: Any  # faiss.Index subtype (IndexFlatIP after normalization)
    texts: list[str]          # User2 messages, aligned with index rows
    user1: str
    user2: str
    df: pd.DataFrame          # full chat DataFrame


def build_index(
    texts: list[str],
    embeddings: np.ndarray,
    user1: str,
    user2: str,
    df: pd.DataFrame | None = None,
) -> IndexData:
    if len(texts) == 0:
        raise ValueError("texts must not be empty")
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be a 2-D array, got shape {embeddings.shape}")
    if len(texts) != embeddings.shape[0]:
        raise ValueError(
            f"texts length ({len(texts)}) must match embeddings row count ({embeddings.shape[0]})"
        )
    dim = embeddings.shape[1]
    embeddings = embeddings.copy()  # don't mutate caller's array
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return IndexData(
        faiss_index=index,
        texts=texts,
        user1=user1,
        user2=user2,
        df=df if df is not None else pd.DataFrame(),
    )


def save_index(data: IndexData, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_index(path: Path) -> IndexData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Index not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)
