"""
Phi-nder – Embedding Generation
================================
Generates vector embeddings by calling the Ollama REST API
with the configured embedding model (default: nomic-embed-text).
"""

from typing import List

import numpy as np
import requests

from config import OLLAMA_BASE_URL, EMBED_MODEL


def get_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector for a single piece of text.

    Parameters
    ----------
    text : str
        The input text to embed.

    Returns
    -------
    list[float]
        The embedding vector.

    Raises
    ------
    ConnectionError
        If Ollama is not reachable.
    RuntimeError
        If the API returns an error.
    """
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {"model": EMBED_MODEL, "prompt": text}

    try:
        response = requests.post(url, json=payload, timeout=120)
    except requests.ConnectionError:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Please make sure Ollama is running."
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama embeddings API error ({response.status_code}): "
            f"{response.text}"
        )

    data = response.json()
    return data["embedding"]


def get_embeddings_batch(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Parameters
    ----------
    texts : list[str]
        The texts to embed.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(len(texts), embedding_dim)``.
    """
    print(f"\n{'─'*60}")
    print(f"🧮 Generating embeddings for {len(texts)} chunks")
    print(f"{'─'*60}")

    embeddings = []
    for i, text in enumerate(texts, 1):
        print(f"  ⚡ Embedding chunk {i}/{len(texts)} ...", end=" ", flush=True)
        emb = get_embedding(text)
        embeddings.append(emb)
        print(f"✅ (dim={len(emb)})")

    print(f"\n✅ All {len(texts)} embeddings generated!")
    return np.array(embeddings, dtype="float32")
