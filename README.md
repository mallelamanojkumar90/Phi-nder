# 🔍 Phi-nder – Local RAG System

A fully offline Retrieval-Augmented Generation (RAG) application that answers questions from your PDF documents using the **Phi** model via **Ollama**.

## Architecture

```
PDF Files → Ingestion & Chunking → Embeddings (Ollama) → FAISS Index
                                                              ↓
User Question → Embed Query → FAISS Search → Top-K Chunks → Phi LLM → Answer
```

## Prerequisites

1. **Python 3.10+**
2. **Ollama** — [Install Ollama](https://ollama.com/download)
3. Pull the required models:
   ```bash
   ollama pull phi3
   ollama pull nomic-embed-text
   ```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Add your PDFs
# Copy PDF files into the pdfs/ folder

# Launch the app
streamlit run app.py
```

## Usage

1. Place your PDF files in the `pdfs/` folder.
2. Open the app in your browser (Streamlit will show the URL).
3. Click **🔄 Ingest PDFs** in the sidebar.
4. Type your question in the chat input and press Enter.
5. Phi-nder will retrieve relevant context and generate an answer!

## Project Structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `config.py` | Central configuration |
| `ingestion.py` | PDF loading & text chunking |
| `embeddings.py` | Ollama embedding generation |
| `vector_store.py` | FAISS index management |
| `llm.py` | Phi answer generation |
| `rag_pipeline.py` | End-to-end RAG orchestrator |

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `phi3` | LLM model name |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks retrieved |
