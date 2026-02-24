"""
Phi-nder – PDF Ingestion & Chunking
====================================
Reads every PDF in a directory, extracts text page-by-page,
and splits the text into overlapping chunks for embedding.
"""

import os
import glob
from typing import List, Dict

from PyPDF2 import PdfReader

from config import CHUNK_SIZE, CHUNK_OVERLAP, PDF_DIR


# ── Public API ──────────────────────────────────────────

def load_pdfs(directory: str = PDF_DIR) -> List[Dict[str, str]]:
    """
    Read all PDF files in *directory*.

    Returns
    -------
    list[dict]
        Each dict has keys ``source`` (filename) and ``text`` (full extracted text).
    """
    documents: List[Dict[str, str]] = []
    pdf_paths = glob.glob(os.path.join(directory, "*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in '{directory}'. "
            "Please add at least one PDF to the pdfs/ folder."
        )

    for path in sorted(pdf_paths):
        reader = PdfReader(path)
        pages_text: List[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        full_text = "\n".join(pages_text)
        if full_text.strip():
            documents.append({
                "source": os.path.basename(path),
                "text": full_text,
            })

    return documents


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split *text* into overlapping chunks using a sliding window.

    Parameters
    ----------
    text : str
        Raw document text.
    chunk_size : int
        Maximum number of characters per chunk.
    overlap : int
        Number of overlapping characters between consecutive chunks.

    Returns
    -------
    list[str]
    """
    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


def ingest(directory: str = PDF_DIR) -> List[Dict]:
    """
    Full ingestion pipeline: load PDFs → chunk text.

    Returns
    -------
    list[dict]
        Each dict: ``{"text": str, "source": str, "chunk_id": int}``
    """
    documents = load_pdfs(directory)
    all_chunks: List[Dict] = []
    chunk_id = 0

    for doc in documents:
        chunks = chunk_text(doc["text"])
        for c in chunks:
            all_chunks.append({
                "text": c,
                "source": doc["source"],
                "chunk_id": chunk_id,
            })
            chunk_id += 1

    return all_chunks
