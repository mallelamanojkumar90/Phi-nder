"""
Phi-nder – Streamlit Query Interface
======================================
A polished, dark-themed Streamlit app for the local RAG pipeline.
Run with:  streamlit run app.py
"""

import os
import streamlit as st
from rag_pipeline import PhiNderPipeline, SOURCE_RAG, SOURCE_MODEL
from config import PDF_DIR, FAISS_DIR, LLM_MODEL, EMBED_MODEL

# ── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="Phi-nder – Local RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ──────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hero Header ─────────────── */
    .hero {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }

    /* ── Sidebar ─────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #0f172a 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #a78bfa;
    }

    /* ── Cards ────────────────────── */
    .source-card {
        background: rgba(30, 27, 75, 0.5);
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        backdrop-filter: blur(8px);
        transition: border-color 0.2s;
    }
    .source-card:hover {
        border-color: rgba(167, 139, 250, 0.6);
    }
    .source-card .src-name {
        color: #a78bfa;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.4rem;
    }
    .source-card .src-text {
        color: #cbd5e1;
        font-size: 0.82rem;
        line-height: 1.55;
    }

    /* ── Stats pills ─────────────── */
    .stat-pill {
        display: inline-block;
        background: rgba(167, 139, 250, 0.15);
        color: #a78bfa;
        border-radius: 999px;
        padding: 0.3rem 0.9rem;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }

    /* ── Answer area ─────────────── */
    .answer-block {
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #a78bfa;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        color: #e2e8f0;
        line-height: 1.7;
        font-size: 0.95rem;
    }

    /* ── Source badge ─────────────── */
    .source-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.25rem 0.85rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        letter-spacing: 0.02em;
    }
    .source-badge.rag {
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.2), rgba(96, 165, 250, 0.2));
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    .source-badge.model {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(245, 158, 11, 0.2));
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
</style>
""", unsafe_allow_html=True)


def _source_badge_html(source_type: str) -> str:
    """Return an HTML badge indicating whether the answer is from RAG or Model."""
    if source_type == SOURCE_RAG:
        return '<span class="source-badge rag">📚 Answered from Knowledge Base (RAG)</span>'
    else:
        return '<span class="source-badge model">🧠 Answered from Model Knowledge</span>'


# ── Session State ───────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = PhiNderPipeline()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Phi-nder Settings")
    st.divider()

    # PDF count
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")] if os.path.isdir(PDF_DIR) else []
    st.markdown(f'<span class="stat-pill">📄 {len(pdf_files)} PDFs found</span>', unsafe_allow_html=True)

    # Index status
    index_exists = os.path.exists(os.path.join(FAISS_DIR, "index.faiss"))
    status_emoji = "✅" if index_exists else "⚠️"
    status_text = "Index ready" if index_exists else "Not indexed"
    st.markdown(f'<span class="stat-pill">{status_emoji} {status_text}</span>', unsafe_allow_html=True)

    st.markdown(f'<span class="stat-pill">🤖 {LLM_MODEL}</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="stat-pill">📐 {EMBED_MODEL}</span>', unsafe_allow_html=True)

    st.divider()

    # Ingest button
    if st.button("🔄 Ingest PDFs", use_container_width=True, type="primary"):
        if not pdf_files:
            st.error("No PDFs found! Add PDF files to the `pdfs/` folder.")
        else:
            with st.spinner("📚 Reading PDFs & generating embeddings…"):
                try:
                    count = st.session_state.pipeline.ingest_pdfs()
                    st.success(f"✅ Indexed **{count}** chunks from **{len(pdf_files)}** PDFs!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")

    st.divider()

    # Clear history
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown(
        "<div style='text-align:center; color:#475569; font-size:0.75rem; margin-top:2rem;'>"
        "Phi-nder v1.1 · Hybrid RAG + Model"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Hero Header ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔍 Phi-nder</h1>
    <p>Your local, offline RAG assistant — powered by Phi & FAISS</p>
</div>
""", unsafe_allow_html=True)


# ── Chat History ────────────────────────────────────────
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant", avatar="🔍"):
        # Source badge
        st.markdown(_source_badge_html(entry["source_type"]), unsafe_allow_html=True)
        st.markdown(f'<div class="answer-block">{entry["answer"]}</div>', unsafe_allow_html=True)
        if entry["sources"]:
            with st.expander(f"📑 Sources ({len(entry['sources'])} chunks)"):
                for src in entry["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="src-name">📄 {src["source"]}</div>'
                        f'<div class="src-text">{src["text"][:300]}{"…" if len(src["text"]) > 300 else ""}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ── Query Input ─────────────────────────────────────────
question = st.chat_input("Ask Phi-nder anything — documents or general knowledge…")

if question:
    # Display user message immediately
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant", avatar="🔍"):
        with st.spinner("🧠 Thinking…"):
            try:
                # Decide: do we have an index?
                if index_exists:
                    context_chunks, token_gen, source_type = (
                        st.session_state.pipeline.query_stream(question)
                    )
                else:
                    # No index – fall through to model knowledge
                    context_chunks, token_gen, source_type = (
                        st.session_state.pipeline.query_stream_no_index(question)
                    )

                # Show the source badge
                badge_placeholder = st.empty()
                badge_placeholder.markdown(
                    _source_badge_html(source_type),
                    unsafe_allow_html=True,
                )

                # Streaming answer
                answer_placeholder = st.empty()
                full_answer = ""
                for token in token_gen:
                    full_answer += token
                    answer_placeholder.markdown(
                        f'<div class="answer-block">{full_answer}▌</div>',
                        unsafe_allow_html=True,
                    )

                # Final render without cursor
                answer_placeholder.markdown(
                    f'<div class="answer-block">{full_answer}</div>',
                    unsafe_allow_html=True,
                )

                # Show sources (only for RAG answers)
                if context_chunks:
                    with st.expander(f"📑 Sources ({len(context_chunks)} chunks)"):
                        for src in context_chunks:
                            st.markdown(
                                f'<div class="source-card">'
                                f'<div class="src-name">📄 {src["source"]}</div>'
                                f'<div class="src-text">{src["text"][:300]}{"…" if len(src["text"]) > 300 else ""}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                # Save to history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": full_answer,
                    "sources": context_chunks,
                    "source_type": source_type,
                })

            except FileNotFoundError:
                # No index – fall through to model knowledge
                try:
                    context_chunks, token_gen, source_type = (
                        st.session_state.pipeline.query_stream_no_index(question)
                    )

                    badge_placeholder = st.empty()
                    badge_placeholder.markdown(
                        _source_badge_html(source_type),
                        unsafe_allow_html=True,
                    )

                    answer_placeholder = st.empty()
                    full_answer = ""
                    for token in token_gen:
                        full_answer += token
                        answer_placeholder.markdown(
                            f'<div class="answer-block">{full_answer}▌</div>',
                            unsafe_allow_html=True,
                        )

                    answer_placeholder.markdown(
                        f'<div class="answer-block">{full_answer}</div>',
                        unsafe_allow_html=True,
                    )

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": full_answer,
                        "sources": [],
                        "source_type": source_type,
                    })

                except ConnectionError as e:
                    st.error(f"🔌 {e}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

            except ConnectionError as e:
                st.error(f"🔌 {e}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
