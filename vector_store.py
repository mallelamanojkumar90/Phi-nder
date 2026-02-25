"""
Phi-nder – FAISS Vector Store
==============================
Manages the FAISS index for similarity search and persists
both the index and the associated chunk metadata to disk.
"""

import os
import pickle
from typing import List, Dict, Tuple

import faiss
import numpy as np

from config import FAISS_DIR, TOP_K


INDEX_FILE    = os.path.join(FAISS_DIR, "index.faiss")
METADATA_FILE = os.path.join(FAISS_DIR, "metadata.pkl")


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build a flat L2 FAISS index from an embedding matrix.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(n, dim)`` float32 matrix.

    Returns
    -------
    faiss.IndexFlatL2
    """
    dim = embeddings.shape[1]
    print(f"\n{'─'*60}")
    print(f"🔨 Building FAISS index (L2, dim={dim}, vectors={embeddings.shape[0]})")
    print(f"{'─'*60}")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"  ✅ Index built with {index.ntotal} vectors")
    return index


def save_index(index: faiss.IndexFlatL2, metadata: List[Dict]) -> None:
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


def load_index() -> Tuple[faiss.IndexFlatL2, List[Dict]]:
    """
    Load a previously saved FAISS index and metadata.

    Returns
    -------
    tuple[faiss.IndexFlatL2, list[dict]]

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
    index: faiss.IndexFlatL2,
    query_embedding: np.ndarray,
    top_k: int = TOP_K,
) -> Tuple[List[int], List[float]]:
    """
    Search the FAISS index for the closest vectors.

    Parameters
    ----------
    index : faiss.IndexFlatL2
    query_embedding : np.ndarray
        Shape ``(1, dim)`` or ``(dim,)`` float32 vector.
    top_k : int

    Returns
    -------
    tuple[list[int], list[float]]
        ``(indices, distances)`` of the top-k nearest neighbours.
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    distances, indices = index.search(query_embedding, top_k)
    return indices[0].tolist(), distances[0].tolist()
