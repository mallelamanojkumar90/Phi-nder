"""
Phi-nder Configuration
======================
Central settings for the local RAG pipeline.
All paths are relative to the project root.
"""

import os

# ── Ollama ──────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL       = os.getenv("LLM_MODEL", "phi")

# ── Chunking ────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Retrieval ───────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "5"))

# RAG-first strategy: Minimum cosine similarity for retrieved
# chunks to be considered relevant.  Range: 0.0 to 1.0.
# A LOWER value means RAG is more likely to be used (more permissive).
# 1.0 = only exact matches, 0.0 = always use RAG.
# The system always tries RAG first; only if the best chunk's
# similarity is BELOW this threshold does it fall back to the
# model's own training data.
# Default 0.3 = strongly favor RAG answers.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.3"))

# ── Paths ───────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PDF_DIR    = os.path.join(BASE_DIR, "pdfs")
FAISS_DIR  = os.path.join(BASE_DIR, "faiss_index")

# Ensure directories exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)
