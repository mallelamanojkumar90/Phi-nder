"""
Phi-nder – RAG Pipeline Orchestrator
======================================
Ties ingestion, embedding, FAISS, and LLM together
into a single easy-to-use pipeline class.

Uses a **RAG-first** answering strategy:
  1. ALWAYS search the knowledge base first when an index exists.
  2. If relevant context is found → answer from RAG (grounded in docs).
  3. ONLY if the knowledge base has NO relevant context → fall back
     to the model's own training data.

This ensures the uploaded documents are always the primary source
of truth for answering questions.
"""

from typing import List, Dict, Generator, Tuple

import numpy as np

from config import TOP_K, PDF_DIR, RELEVANCE_THRESHOLD
from ingestion import ingest
from embeddings import get_embedding, get_embeddings_batch
from vector_store import build_index, save_index, load_index, search
from llm import (
    generate_answer,
    generate_answer_stream,
    generate_model_answer,
    generate_model_answer_stream,
)

# Source-type constants – used by the UI to display badges
SOURCE_RAG   = "rag"
SOURCE_MODEL = "model"


class PhiNderPipeline:
    """
    End-to-end RAG pipeline with automatic fall-through to model
    knowledge when the knowledge base doesn't cover the question.

    Usage
    -----
    >>> p = PhiNderPipeline()
    >>> p.ingest_pdfs()            # one-time PDF ingestion
    >>> answer = p.query("What is …?")
    """

    def __init__(self) -> None:
        self.index = None
        self.metadata: List[Dict] = []

    # ── Ingestion ───────────────────────────────────────

    def ingest_pdfs(self, pdf_dir: str = PDF_DIR) -> int:
        """
        Ingest all PDFs, generate embeddings, build & save the FAISS index.

        Returns
        -------
        int
            Number of chunks indexed.
        """
        chunks = ingest(pdf_dir)
        if not chunks:
            raise ValueError("No text chunks were produced from the PDFs.")

        texts = [c["text"] for c in chunks]
        embeddings = get_embeddings_batch(texts)

        self.index = build_index(embeddings)
        self.metadata = chunks
        save_index(self.index, self.metadata)

        return len(chunks)

    # ── Retrieval ───────────────────────────────────────

    def _ensure_index(self) -> None:
        """Load the FAISS index from disk if not already in memory."""
        if self.index is None:
            self.index, self.metadata = load_index()

    def retrieve(self, question: str, top_k: int = TOP_K) -> List[Dict]:
        """
        Retrieve the most relevant chunks for a question.

        Returns
        -------
        list[dict]
            Each dict has ``text``, ``source``, ``chunk_id``, ``distance``.
        """
        self._ensure_index()

        q_emb = np.array(get_embedding(question), dtype="float32")
        indices, distances = search(self.index, q_emb, top_k)

        results = []
        for idx, dist in zip(indices, distances):
            if 0 <= idx < len(self.metadata):
                entry = dict(self.metadata[idx])
                entry["distance"] = float(dist)
                results.append(entry)

        return results

    # ── Relevance check ─────────────────────────────────

    @staticmethod
    def _is_context_relevant(context_chunks: List[Dict]) -> bool:
        """
        Return ``True`` if the retrieved chunks are relevant enough
        to answer the question from the RAG knowledge base.

        **RAG-first strategy:** This method is intentionally permissive.
        We use BOTH the best distance AND the average distance of the
        top chunks to decide relevance. RAG is preferred whenever
        there is *any* reasonable match in the knowledge base.

        Uses the *best* (smallest) L2 distance among the top-k
        chunks and compares against ``RELEVANCE_THRESHOLD``.
        """
        if not context_chunks:
            return False

        best_distance = min(c["distance"] for c in context_chunks)

        # Primary check: if the best chunk is within threshold → RAG
        if best_distance <= RELEVANCE_THRESHOLD:
            return True

        # Secondary check: if at least 2 chunks have reasonable
        # distances (within 1.5x threshold), the query likely has
        # partial coverage in the knowledge base → still use RAG
        moderate_threshold = RELEVANCE_THRESHOLD * 1.5
        moderate_matches = sum(
            1 for c in context_chunks if c["distance"] <= moderate_threshold
        )
        if moderate_matches >= 2:
            return True

        return False

    # ── Query (RAG-first with model fallback) ────────────

    def query(self, question: str, top_k: int = TOP_K) -> Tuple[str, str]:
        """
        Run the **RAG-first** pipeline.

        Strategy:
          1. Always retrieve from the knowledge base first.
          2. If relevant context found → answer from RAG.
          3. Only if context is irrelevant → fall back to model.

        Returns
        -------
        tuple[str, str]
            ``(answer, source_type)`` where *source_type* is
            ``"rag"`` or ``"model"``.
        """
        # Step 1: Always try RAG first
        context_chunks = self.retrieve(question, top_k)

        # Step 2: Use RAG if context is relevant
        if self._is_context_relevant(context_chunks):
            return generate_answer(question, context_chunks), SOURCE_RAG

        # Step 3: Only fall back to model if RAG has nothing useful
        return generate_model_answer(question), SOURCE_MODEL

    def query_stream(
        self, question: str, top_k: int = TOP_K
    ) -> Tuple[List[Dict], Generator[str, None, None], str]:
        """
        Streaming version of ``query()``.

        **RAG-first:** Always attempts retrieval from knowledge base
        before falling back to model knowledge.

        Returns
        -------
        tuple[list[dict], Generator, str]
            ``(context_chunks, token_generator, source_type)``
            *context_chunks* is empty when *source_type* is ``"model"``.
        """
        # Step 1: Always try RAG first
        context_chunks = self.retrieve(question, top_k)

        # Step 2: Use RAG if context is relevant
        if self._is_context_relevant(context_chunks):
            token_gen = generate_answer_stream(question, context_chunks)
            return context_chunks, token_gen, SOURCE_RAG

        # Step 3: Only fall back to model if RAG has nothing useful
        token_gen = generate_model_answer_stream(question)
        return [], token_gen, SOURCE_MODEL

    def query_stream_no_index(
        self, question: str,
    ) -> Tuple[List[Dict], Generator[str, None, None], str]:
        """
        Answer from model knowledge when no FAISS index exists at all.

        Returns
        -------
        tuple[list[dict], Generator, str]
            ``([], token_generator, "model")``
        """
        token_gen = generate_model_answer_stream(question)
        return [], token_gen, SOURCE_MODEL
