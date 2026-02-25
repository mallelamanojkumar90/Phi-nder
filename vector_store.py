"""
Phi-nder – FAISS Vector Store
==============================
Manages the FAISS index for similarity search and persists
both the index and the associated chunk metadata to disk.

Uses **cosine similarity** via IndexFlatIP on L2-normalized
embeddings.  Similarity scores range from 0.0 (unrelated)
to 1.0 (identical).  Higher = more relevant.
"""

import os
import pickle
from typing import List, Dict, Tuple

import faiss
import numpy as np

from config import FAISS_DIR, TOP_K


INDEX_FILE    = os.path.join(FAISS_DIR, "index.faiss")
METADATA_FILE = os.path.join(FAISS_DIR, "metadata.pkl")


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row so inner product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    return (vectors / norms).astype("float32")


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a cosine-similarity FAISS index from an embedding matrix.

    Embeddings are L2-normalized before indexing so that inner
    product equals cosine similarity.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(n, dim)`` float32 matrix.

    Returns
    -------
    faiss.IndexFlatIP
    """
    embeddings = _normalize(embeddings)
    dim = embeddings.shape[1]
    print(f"\n{'─'*60}")
    print(f"🔨 Building FAISS index (Cosine/IP, dim={dim}, vectors={embeddings.shape[0]})")
    print(f"{'─'*60}")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  ✅ Index built with {index.ntotal} vectors")
    return index


def save_index(index: faiss.IndexFlatIP, metadata: List[Dict]) -> None:
    """Persist the FAISS index and chunk metadata to disk."""
    print(f"\n💾 Saving index to disk ...")
    os.makedirs(FAISS_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)
    print(f"  ✅ Saved: {INDEX_FILE}")
    print(f"  ✅ Saved: {METADATA_FILE}")
    print(f"\n{'='*60}")
    print(f"🎉 INGESTION COMPLETE!")
    print(f"{'='*60}\n")


def load_index() -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    """
    Load a previously saved FAISS index and metadata.

    Returns
    -------
    tuple[faiss.IndexFlatIP, list[dict]]

    Raises
    ------
    FileNotFoundError
        If no saved index exists.
    """
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(
            "No FAISS index found. Please ingest PDFs first."
        )

    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


def search(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    top_k: int = TOP_K,
) -> Tuple[List[int], List[float]]:
    """
    Search the FAISS index for the most similar vectors.

    The query embedding is L2-normalized before searching so
    that scores represent cosine similarity (0.0–1.0).

    Parameters
    ----------
    index : faiss.IndexFlatIP
    query_embedding : np.ndarray
        Shape ``(1, dim)`` or ``(dim,)`` float32 vector.
    top_k : int

    Returns
    -------
    tuple[list[int], list[float]]
        ``(indices, similarities)`` of the top-k most similar vectors.
        Similarities are cosine similarities in range [0, 1].
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    query_embedding = _normalize(query_embedding)

    similarities, indices = index.search(query_embedding, top_k)
    return indices[0].tolist(), similarities[0].tolist()
