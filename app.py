import os
import sys
import json
import time
import re
import hashlib
from pathlib import Path
import sys
from pathlib import Path

# Universal root resolution for Streamlit Cloud & local execution
file_path = Path(__file__).resolve()
if (file_path.parent / "src").exists():
    ROOT_DIR = file_path.parent
elif (file_path.parent.parent / "src").exists():
    ROOT_DIR = file_path.parent.parent
elif (file_path.parent.parent.parent / "src").exists():
    ROOT_DIR = file_path.parent.parent.parent
else:
    ROOT_DIR = file_path.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Execute main UI
from src.ui import app
# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import SAMPLE_DIR, DEFAULT_TOP_K, HYBRID_ALPHA, GEMINI_API_KEY
from src.analytics.pdf_parser import PDFParser, ExtractedDocument
from src.analytics.text_profiler import TextProfiler, DocumentAnalytics
from src.rag.chatbot import SmartPDFChatbot, ChatResponse, Citation

# Configure Streamlit Page
st.set_page_config(
    page_title="SmartPdf Chatbot — Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sky Blue & Biscuit Color Theme Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Background: Deep Dark Slate with Biscuit Undertones */
    .stApp {
        background: linear-gradient(135deg, #090D16 0%, #0F172A 50%, #1C1917 100%);
    }

    /* Hero Banner with Sky Blue & Biscuit Glow */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #292524 100%);
        border-radius: 20px;
        padding: 28px 36px;
        margin-bottom: 24px;
        border: 1.5px solid #D4A373;
        box-shadow: 0 15px 35px -10px rgba(56, 189, 248, 0.25), 0 10px 25px -5px rgba(212, 163, 115, 0.2);
    }

    /* Gradient Title: Sky Blue to Biscuit */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #7DD3FC 40%, #E6CCB2 75%, #D4A373 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        color: #E2E8F0 !important;
        font-size: 1.05rem;
        margin-top: 8px;
        font-weight: 400;
    }

    /* Biscuit & Sky Blue Welcome Container */
    .welcome-card {
        background: linear-gradient(145deg, #131B2E 0%, #1C1917 100%);
        border: 1px solid #D4A373;
        border-radius: 16px;
        padding: 26px 30px;
        box-shadow: 0 12px 25px -5px rgba(0, 0, 0, 0.4);
        margin-bottom: 22px;
    }

    .welcome-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #38BDF8 !important;
        margin-bottom: 10px;
    }

    .welcome-text {
        font-size: 1rem;
        color: #E6CCB2 !important;
        line-height: 1.6;
        margin-bottom: 16px;
    }

    /* Feature Badges in Biscuit & Sky Blue */
    .stat-badge {
        display: inline-block;
        background: rgba(212, 163, 115, 0.15);
        color: #FAEDCD !important;
        border: 1px solid #D4A373;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 4px 6px 4px 0;
    }

    .stat-badge-sky {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #BAE6FD !important;
        border: 1px solid #38BDF8;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 4px 6px 4px 0;
    }

    /* KPI Summary Tiles */
    .kpi-tile {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(212, 163, 115, 0.4);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #38BDF8;
    }
    .kpi-label {
        font-size: 0.82rem;
        color: #D4A373;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
        font-weight: 600;
    }

    /* Interactive Concept Pills */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #D4A373;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #38BDF8;
        color: #38BDF8;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "parser" not in st.session_state:
    st.session_state.parser = PDFParser()
if "profiler" not in st.session_state:
    st.session_state.profiler = TextProfiler()
if "chatbot" not in st.session_state:
    st.session_state.chatbot = SmartPDFChatbot()
if "current_doc" not in st.session_state:
    st.session_state.current_doc = None
if "analytics" not in st.session_state:
    st.session_state.analytics = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = GEMINI_API_KEY
if "model_name" not in st.session_state:
    st.session_state.model_name = "gemini-2.5-flash"
if "loaded_file_hash" not in st.session_state:
    st.session_state.loaded_file_hash = ""
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def process_uploaded_bytes(file_bytes: bytes, filename: str) -> bool:
    """Processes, indexes, and loads PDF bytes into session."""
    if not file_bytes:
        st.error("Uploaded file appears to be empty.")
        return False

    file_hash = hashlib.md5(file_bytes).hexdigest()
    if st.session_state.loaded_file_hash == file_hash and st.session_state.current_doc is not None:
        return False

    try:
        with st.spinner(f"✨ Ingesting & Building Vector Embeddings for '{filename}'..."):
            doc = st.session_state.parser.parse(file_bytes, filename=filename)
            st.session_state.current_doc = doc
            st.session_state.chatbot.load_document(doc)
            st.session_state.analytics = st.session_state.profiler.profile_document(doc)
            st.session_state.messages = []
            st.session_state.loaded_file_hash = file_hash
        return True
    except Exception as e:
        st.error(f"❌ Failed to parse PDF: {str(e)}")
        return False


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color:#38BDF8; margin-bottom:0;'>📄 SmartPdf Chatbot</h2>", unsafe_allow_html=True)
    st.caption("AI-Powered Document Intelligence & RAG")
    st.markdown("---")

    # AI Model Selector
    st.subheader("🤖 AI Model Engine")
    model_choice = st.selectbox(
        "Model Selection",
        options=[
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-1.5-flash",
            "extractive-nlp-engine"
        ],
        index=0,
        format_func=lambda x: {
            "gemini-2.5-flash": "⚡ Gemini 2.5 Flash (Fastest)",
            "gemini-2.5-pro": "🧠 Gemini 2.5 Pro (Deep Reasoning)",
            "gemini-1.5-flash": "🚀 Gemini 1.5 Flash",
            "extractive-nlp-engine": "⚙️ Local Extractive NLP (100% Offline)"
        }.get(x, x)
    )
    st.session_state.model_name = model_choice

    user_api_key = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        help="Provide your Gemini API key for conversational generative responses. Leave blank for instant offline extractive mode."
    )
    st.session_state.api_key = user_api_key

    st.session_state.chatbot.set_config(
        api_key=st.session_state.api_key,
        model_name=st.session_state.model_name
    )

    st.markdown("---")
    st.subheader("📁 Upload Document")

    sidebar_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"],
        key="sidebar_uploader",
        help="Upload any PDF to analyze."
    )
    if sidebar_file is not None:
        if process_uploaded_bytes(sidebar_file.getvalue(), sidebar_file.name):
            st.rerun()

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📑 Sample Report", use_container_width=True):
            sample_pdf_path = SAMPLE_DIR / "ai_in_healthcare_report.pdf"
            if not sample_pdf_path.exists():
                from scripts.create_sample_pdf import generate_sample_report
                generate_sample_report(str(sample_pdf_path))
            with open(sample_pdf_path, "rb") as f:
                if process_uploaded_bytes(f.read(), "ai_in_healthcare_report.pdf"):
                    st.rerun()

    with col_btn2:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.current_doc = None
            st.session_state.analytics = None
            st.session_state.messages = []
            st.session_state.loaded_file_hash = ""
            st.session_state.chatbot.clear_history()
            st.rerun()

    st.markdown("---")
    st.subheader("⚙️ Retrieval Parameters")
    top_k = st.slider("Context Chunks (Top-K)", min_value=1, max_value=8, value=DEFAULT_TOP_K)
    hybrid_alpha = st.slider(
        "Hybrid Search Balance (Dense vs Sparse)",
        min_value=0.0,
        max_value=1.0,
        value=HYBRID_ALPHA,
        help="1.0 = Pure Semantic Vector Search | 0.0 = Pure BM25 Keyword Search"
    )
    st.session_state.chatbot.retriever.alpha = hybrid_alpha
    st.session_state.chatbot.top_k = top_k


# ==========================================
# HERO BANNER
# ==========================================
st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">SmartPdf Chatbot</h1>
    <div class="hero-subtitle">
        AI-Powered Document Intelligence • Hybrid Dense + BM25 Retrieval • Statistical NLP Analytics
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# LANDING SCREEN (WHEN NO DOCUMENT LOADED)
# ==========================================
if not st.session_state.current_doc:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-title">👋 Welcome to SmartPdf Chatbot!</div>
        <div class="welcome-text">
            Upload your PDF document below or select <b>"📑 Sample Report"</b> from the sidebar for an instant demonstration.
        </div>
        <div>
            <span class="stat-badge-sky">⚡ Hybrid Vector + BM25 Retrieval</span>
            <span class="stat-badge">📊 Statistical NLP & Readability Scoring</span>
            <span class="stat-badge-sky">📚 Grounded Source Citations</span>
            <span class="stat-badge">🔒 100% Offline Extractive Engine</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📤 Drop your PDF file below to start:")
    main_drop = st.file_uploader(
        "Drag and drop PDF here",
        type=["pdf"],
        key="main_dropzone",
        label_visibility="collapsed"
    )
    if main_drop is not None:
        if process_uploaded_bytes(main_drop.getvalue(), main_drop.name):
            st.rerun()

    st.stop()


# ==========================================
# ACTIVE DOCUMENT METRICS BAR
# ==========================================
doc = st.session_state.current_doc
an = st.session_state.analytics

col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
with col_k1:
    st.markdown(f'<div class="kpi-tile"><div class="kpi-val">{doc.filename[:15]}..</div><div class="kpi-label">Active PDF</div></div>', unsafe_allow_html=True)
with col_k2:
    st.markdown(f'<div class="kpi-tile"><div class="kpi-val">{doc.total_pages}</div><div class="kpi-label">Pages</div></div>', unsafe_allow_html=True)
with col_k3:
    st.markdown(f'<div class="kpi-tile"><div class="kpi-val">{doc.total_words:,}</div><div class="kpi-label">Total Words</div></div>', unsafe_allow_html=True)
with col_k4:
    st.markdown(f'<div class="kpi-tile"><div class="kpi-val">{len(st.session_state.chatbot.all_chunks)}</div><div class="kpi-label">Vector Chunks</div></div>', unsafe_allow_html=True)
with col_k5:
    st.markdown(f'<div class="kpi-tile"><div class="kpi-val">{doc.estimated_reading_time_minutes}m</div><div class="kpi-label">Reading Time</div></div>', unsafe_allow_html=True)

if doc.extraction_warning:
    st.warning(f"⚠️ {doc.extraction_warning}")

st.markdown("---")

# ==========================================
# INTERACTIVE MULTI-TAB DASHBOARD
# ==========================================
tab_chat, tab_analytics, tab_inspector, tab_search = st.tabs([
    "💬 Interactive Smart Chat",
    "📊 NLP Analytics & Readability",
    "🔍 Vector Chunk Inspector",
    "🔎 Real-Time Text Search"
])


# ==========================================
# TAB 1: INTERACTIVE SMART CHAT
# ==========================================
with tab_chat:
    col_act1, col_act2 = st.columns([1.2, 1])
    with col_act1:
        if st.button("📝 Summarize Document", use_container_width=True):
            with st.spinner("Generating executive summary..."):
                t0 = time.time()
                summary_resp = st.session_state.chatbot.summarize_document()
                latency = round((time.time() - t0) * 1000, 1)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": summary_resp.answer,
                    "model": summary_resp.model_used,
                    "latency": latency,
                    "citations": []
                })

    with col_act2:
        if st.session_state.messages:
            chat_transcript = "# SmartPdf Chatbot Transcript\n\n"
            for m in st.session_state.messages:
                chat_transcript += f"### {m['role'].title()}\n{m['content']}\n\n"
            st.download_button(
                "📥 Export Chat",
                data=chat_transcript,
                file_name=f"smartpdf_chat_{doc.filename}.md",
                mime="text/markdown",
                use_container_width=True
            )

    # Interactive Concept Pills
    if an and an.top_keyphrases:
        st.markdown("**💡 Interactive Topics (Click to ask about a topic):**")
        pill_cols = st.columns(min(len(an.top_keyphrases[:5]), 5))
        for idx, kp in enumerate(an.top_keyphrases[:5]):
            topic_prompt = f"What does the document say about {kp.phrase}?"
            if pill_cols[idx].button(f"🔍 {kp.phrase.title()}", key=f"kp_pill_{idx}", use_container_width=True):
                st.session_state.pending_query = topic_prompt
                st.rerun()

    st.markdown("---")

    # Render Conversation History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📚 Grounded Sources ({len(msg['citations'])} excerpts)"):
                    for idx, cit in enumerate(msg["citations"], 1):
                        st.markdown(f"**Source {idx} — Page {cit.page_number}** `(Relevance: {cit.score * 100:.1f}%)`")
                        st.caption(f"\"{cit.excerpt}\"")
            if msg.get("latency"):
                st.caption(f"⚡ Generated in {msg['latency']}ms via `{msg.get('model', 'RAG')}`")

    # Chat Input Box
    prompt = st.chat_input("Ask a question about this document...")
    if st.session_state.pending_query:
        prompt = st.session_state.pending_query
        st.session_state.pending_query = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing document and formulating answer..."):
                t0 = time.time()
                response = st.session_state.chatbot.answer_question(prompt, top_k=top_k)
                latency = round((time.time() - t0) * 1000, 1)
                
                st.markdown(response.answer)
                if response.citations:
                    with st.expander(f"📚 Grounded Sources ({len(response.citations)} excerpts)"):
                        for idx, cit in enumerate(response.citations, 1):
                            st.markdown(f"**Source {idx} — Page {cit.page_number}** `(Relevance: {cit.score * 100:.1f}%)`")
                            st.caption(f"\"{cit.excerpt}\"")
                st.caption(f"⚡ Generated in {latency}ms via `{response.model_used}`")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "model": response.model_used,
                    "latency": latency,
                    "citations": response.citations
                })


# ==========================================
# TAB 2: NLP ANALYTICS & READABILITY
# ==========================================
with tab_analytics:
    if an:
        st.subheader("📈 Statistical NLP Overview")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unique Vocabulary", f"{an.unique_words:,}")
        c2.metric("Lexical Diversity (TTR)", f"{an.lexical_diversity_ttr:.3f}")
        c3.metric("Flesch Reading Ease", f"{an.readability.flesch_reading_ease}/100", an.readability.reading_ease_level)
        c4.metric("FK Grade Level", f"Grade {an.readability.flesch_kincaid_grade}")

        st.markdown("---")
        col_g1, col_g2 = st.columns(2)
        custom_color_scale = [[0, "#38BDF8"], [0.5, "#DDB892"], [1, "#D4A373"]]

        with col_g1:
            st.subheader("📄 Interactive Page Density & Reading Velocity")
            df_pages = pd.DataFrame([
                {
                    "Page": f"Page {pm.page_number}",
                    "Word Count": pm.word_count,
                    "Reading Time (s)": pm.reading_time_sec,
                    "Density (%)": pm.density_pct
                }
                for pm in an.page_metrics
            ])
            fig_pages = px.bar(
                df_pages,
                x="Page",
                y="Word Count",
                color="Density (%)",
                color_continuous_scale=custom_color_scale,
                title="Page Word Distribution"
            )
            fig_pages.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pages, use_container_width=True)

        with col_g2:
            st.subheader("🔑 Top TF-IDF Keyphrases & Salience Scores")
            if an.top_keyphrases:
                df_kp = pd.DataFrame([
                    {"Keyphrase": kp.phrase.title(), "Salience Score": kp.score, "Occurrences": kp.frequency}
                    for kp in an.top_keyphrases[:10]
                ])
                fig_kp = px.bar(
                    df_kp,
                    x="Salience Score",
                    y="Keyphrase",
                    orientation="h",
                    color="Occurrences",
                    color_continuous_scale=custom_color_scale,
                    title="Salient Concept Importance"
                )
                fig_kp.update_layout(yaxis=dict(autorange="reversed"), template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_kp, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Automated Extractive Executive Summary")
        for idx, sentence in enumerate(an.extractive_summary, 1):
            st.markdown(f"**{idx}.** {sentence}")


# ==========================================
# TAB 3: VECTOR CHUNK INSPECTOR
# ==========================================
with tab_inspector:
    st.subheader("🔍 Vector Index & Chunk Inspector")
    st.caption("Inspect how text chunks are created, tokenized, and retrieved.")

    all_chunks = st.session_state.chatbot.all_chunks
    st.markdown(f"**Total Indexed Chunks:** `{len(all_chunks)}`")

    test_q = st.text_input("Test Hybrid Retrieval Query:", value="radiology medical imaging sensitivity")
    if test_q:
        t0 = time.time()
        results = st.session_state.chatbot.retriever.retrieve(test_q, top_k=top_k)
        ret_ms = round((time.time() - t0) * 1000, 2)
        
        st.markdown(f"**Found {len(results)} matching chunks in {ret_ms}ms:**")
        for idx, res in enumerate(results, 1):
            with st.container():
                st.markdown(f"##### Hit #{idx} — Chunk ID: `{res.chunk.chunk_id}` | Score: `{res.score:.4f}` | Page: `{res.chunk.page_number}`")
                st.info(res.chunk.text)

    st.markdown("---")
    st.subheader("📦 All Indexed Document Chunks")
    if all_chunks:
        df_chunks = pd.DataFrame([
            {
                "Chunk ID": c.chunk_id,
                "Page": c.page_number,
                "Words": c.word_count,
                "Characters": c.char_length,
                "Excerpt": c.text[:140] + "..."
            }
            for c in all_chunks
        ])
        st.dataframe(df_chunks, use_container_width=True)


# ==========================================
# TAB 4: REAL-TIME TEXT SEARCH
# ==========================================
with tab_search:
    st.subheader("🔎 In-Document Text Search & Match Finder")
    st.caption("Search for any exact word or phrase across all pages in the active document.")

    search_term = st.text_input("Enter search keyword or phrase:", placeholder="e.g. market, clinical, HIPAA")
    if search_term and search_term.strip():
        term = search_term.strip().lower()
        matched_pages = []
        for page in doc.pages:
            if term in page.text.lower():
                count = len(re.findall(r"\b" + re.escape(term) + r"\b", page.text.lower()))
                matched_pages.append((page.page_number, count, page.text))

        if matched_pages:
            st.success(f"Found matches on {len(matched_pages)} page(s):")
            for page_num, count, text in matched_pages:
                with st.expander(f"📄 Page {page_num} ({count} occurrence{'s' if count > 1 else ''})"):
                    highlighted = re.sub(
                        r"(?i)\b(" + re.escape(term) + r")\b",
                        r"**:\1:**",
                        text
                    )
                    st.markdown(highlighted)
        else:
            st.warning(f"No exact matches found for '{search_term}'.")
