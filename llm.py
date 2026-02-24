"""
Phi-nder – LLM Answer Generation
==================================
Sends a retrieval-augmented prompt to the Phi model
running on Ollama and returns the generated answer.

Supports two modes:
  1. **RAG mode** – answer grounded in retrieved document context.
  2. **Model mode** – answer from the model's own training data
     when the knowledge base doesn't contain relevant information.
"""

from typing import List, Dict, Generator

import requests

from config import OLLAMA_BASE_URL, LLM_MODEL


# ── System prompts ──────────────────────────────────────

RAG_SYSTEM_PROMPT = (
    "You are Phi-nder, an intelligent assistant. "
    "Your PRIMARY job is to answer questions based on the provided document context. "
    "Always prioritize information from the context over your own knowledge. "
    "Use the context thoroughly — extract relevant details, summarize, "
    "and synthesize the information to give a comprehensive answer. "
    "Only say the context does not contain enough information if it is "
    "genuinely unrelated to the question. Do not make up facts beyond "
    "what the context provides."
)

MODEL_SYSTEM_PROMPT = (
    "You are Phi-nder, an intelligent assistant. "
    "The user's question is NOT covered by the uploaded documents, "
    "so answer it using your own knowledge from your training data. "
    "Be helpful, accurate, and concise. "
    "If you are unsure about something, say so honestly."
)

# ── Prompt templates ────────────────────────────────────

RAG_PROMPT_TEMPLATE = """Use the following context to answer the question.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}

Answer:"""

MODEL_PROMPT_TEMPLATE = """Answer the following question using your own knowledge.

Question: {question}

Answer:"""


# ── Helpers ─────────────────────────────────────────────

def _build_rag_prompt(question: str, context_chunks: List[Dict]) -> str:
    """Assemble the RAG prompt from context chunks and the question."""
    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    return RAG_PROMPT_TEMPLATE.format(context=context_text, question=question)


def _build_model_prompt(question: str) -> str:
    """Assemble a prompt for the model to answer from its training data."""
    return MODEL_PROMPT_TEMPLATE.format(question=question)


def _post_ollama(
    prompt: str,
    system: str,
    stream: bool = False,
) -> requests.Response:
    """Send a request to the Ollama generate API."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": stream,
    }
    try:
        resp = requests.post(url, json=payload, timeout=300, stream=stream)
    except requests.ConnectionError:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Please make sure Ollama is running."
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama generate API error ({resp.status_code}): {resp.text}"
        )
    return resp


# ── RAG-mode generation ────────────────────────────────

def generate_answer(question: str, context_chunks: List[Dict]) -> str:
    """
    Generate an answer grounded in the retrieved context (non-streaming).
    """
    prompt = _build_rag_prompt(question, context_chunks)
    resp = _post_ollama(prompt, RAG_SYSTEM_PROMPT, stream=False)
    return resp.json().get("response", "").strip()


def generate_answer_stream(
    question: str, context_chunks: List[Dict]
) -> Generator[str, None, None]:
    """
    Stream a RAG-grounded answer token-by-token.
    """
    prompt = _build_rag_prompt(question, context_chunks)
    resp = _post_ollama(prompt, RAG_SYSTEM_PROMPT, stream=True)
    import json
    for line in resp.iter_lines(decode_unicode=True):
        if line:
            data = json.loads(line)
            token = data.get("response", "")
            if token:
                yield token
            if data.get("done", False):
                break


# ── Model-mode generation (no RAG context) ─────────────

def generate_model_answer(question: str) -> str:
    """
    Generate an answer purely from the model's training data (non-streaming).
    """
    prompt = _build_model_prompt(question)
    resp = _post_ollama(prompt, MODEL_SYSTEM_PROMPT, stream=False)
    return resp.json().get("response", "").strip()


def generate_model_answer_stream(
    question: str,
) -> Generator[str, None, None]:
    """
    Stream an answer from the model's own knowledge, token-by-token.
    """
    prompt = _build_model_prompt(question)
    resp = _post_ollama(prompt, MODEL_SYSTEM_PROMPT, stream=True)
    import json
    for line in resp.iter_lines(decode_unicode=True):
        if line:
            data = json.loads(line)
            token = data.get("response", "")
            if token:
                yield token
            if data.get("done", False):
                break
