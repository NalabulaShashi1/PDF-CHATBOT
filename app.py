import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Smart PDF Chatbot", page_icon="📄", layout="wide")

# -----------------------------
# SKY BLUE & WHITE THEME
# -----------------------------
st.markdown(
    """
    <style>
    :root {
        --sky-blue: #0ea5e9;
        --sky-blue-light: #e0f2fe;
        --sky-blue-mid: #7dd3fc;
        --sky-blue-dark: #0369a1;
    }

    /* App background */
    .stApp {
        background: linear-gradient(180deg, #f0f9ff 0%, #ffffff 45%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 2px solid var(--sky-blue);
    }
    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid var(--sky-blue);
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--sky-blue-dark) !important;
    }

    /* Titles */
    h1, h2, h3 {
        color: #000000 !important;
    }

    /* Body text stays black for readability */
    .stApp, .stApp p, .stApp li, .stApp span, .stMarkdown {
        color: #000000;
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        background-color: #ffffff;
        color: #000000;
        border: 2px solid var(--sky-blue);
        border-radius: 10px;
        font-weight: 600;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: var(--sky-blue);
        color: #ffffff;
        border-color: var(--sky-blue-dark);
    }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        background-color: #ffffff;
        border: 1px solid var(--sky-blue-mid);
        border-radius: 14px;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 4px rgba(14, 165, 233, 0.12);
        color: #000000;
    }

    /* File uploader */
    section[data-testid="stFileUploaderDropzone"] {
        background-color: var(--sky-blue-light);
        border: 2px dashed var(--sky-blue);
        border-radius: 12px;
    }

    /* Info / success boxes */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Expander */
    details {
        background-color: var(--sky-blue-light);
        border-radius: 10px;
        border: 1px solid var(--sky-blue-mid);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# SESSION STATE
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []          # list of dicts: {question, answer, sources}
if "chunks" not in st.session_state:
    st.session_state.chunks = []           # list of dicts: {text, source, page}
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []  # names of files already indexed

# -----------------------------
# SIDEBAR: SETTINGS
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Anthropic API key (optional)",
        type="password",
        help="Paste a key to get a real generated answer written from the retrieved "
             "passages. Leave blank to just see the best-matching passages instead.",
    )
    model_name = st.selectbox(
        "Model",
        ["claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5-20251001"],
        index=0,
        disabled=not api_key,
    )

    st.divider()
    st.subheader("Retrieval")
    chunk_size = st.slider("Chunk size (characters)", 300, 1500, 600, 50)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, 200, 25)
    top_k = st.slider("Passages to retrieve", 1, 10, 6)
    min_similarity = st.slider(
        "Minimum relevance", 0.0, 1.0, 0.7, 0.01,
        help="Passages scoring below this are ignored. TF-IDF similarity scores "
             "are naturally on the low side — lower this if you get no results.",
    )

    st.divider()
    if st.button("🗑️ Clear everything", use_container_width=True):
        st.session_state.history = []
        st.session_state.chunks = []
        st.session_state.vectorizer = None
        st.session_state.vectors = None
        st.session_state.processed_files = []
        st.rerun()

st.title("📄 Smart PDF Chatbot")
st.caption("Upload one or more PDFs, then chat with them. Retrieval is TF-IDF based; "
           "answer generation uses Claude if you supply an API key.")

# -----------------------------
# FUNCTIONS: TEXT EXTRACTION & CHUNKING
# -----------------------------
def extract_pdf_pages(file):
    """Return list of (page_number, page_text)."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = re.sub(r"\s+", " ", page_text).strip()
        if page_text:
            pages.append((i, page_text))
    return pages


def chunk_pages(pages, source_name, size, overlap):
    """Sliding-window chunking that keeps track of source file + page number."""
    chunks = []
    for page_num, text in pages:
        if len(text) <= size:
            chunks.append({"text": text, "source": source_name, "page": page_num})
            continue
        start = 0
        while start < len(text):
            end = start + size
            piece = text[start:end].strip()
            if len(piece) > 30:
                chunks.append({"text": piece, "source": source_name, "page": page_num})
            if end >= len(text):
                break
            start = end - overlap
    return chunks


def build_index(chunks):
    corpus = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), sublinear_tf=True
    )
    vectors = vectorizer.fit_transform(corpus)
    return vectorizer, vectors


def retrieve(question, vectorizer, vectors, chunks, k, min_sim):
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, vectors)[0]

    # Grab a wider pool than k, then merge neighboring chunks (same source,
    # adjacent index) into single richer passages so answers get fuller,
    # more accurate context instead of disconnected fragments.
    pool_size = min(len(chunks), max(k * 3, 15))
    ranked_idx = np.argsort(sims)[::-1][:pool_size]
    candidate_idx = sorted(i for i in ranked_idx if sims[i] >= min_sim)

    merged = []
    used = set()
    for idx in candidate_idx:
        if idx in used:
            continue
        group = [idx]
        used.add(idx)
        # absorb the immediate neighbor if it's from the same source and
        # also scored above threshold, giving continuous context
        if (idx + 1) < len(chunks) and (idx + 1) in candidate_idx \
                and chunks[idx + 1]["source"] == chunks[idx]["source"]:
            group.append(idx + 1)
            used.add(idx + 1)

        text = " ".join(chunks[i]["text"] for i in group)
        score = float(max(sims[i] for i in group))
        merged.append({
            "text": text,
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"],
            "score": score,
        })

    merged.sort(key=lambda p: p["score"], reverse=True)
    return merged[:k]


def generate_answer_with_claude(question, passages, api_key, model):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    context_block = "\n\n".join(
        f"[Passage {i+1} | {p['source']} p.{p['page']}]\n{p['text']}"
        for i, p in enumerate(passages)
    )

    system_prompt = (
        "You are a precise research assistant. Answer the user's question using ONLY "
        "the passages provided. Be direct and precise:\n"
        "- Get straight to the point, no filler or restating the question.\n"
        "- Prefer short sentences or a tight bullet list over long paragraphs.\n"
        "- Keep it as brief as possible while still fully answering the question.\n"
        "- Cite passages inline like [1], [2] matching the passage numbers given.\n"
        "- If the passages do not contain the answer, say so in one short sentence "
        "instead of guessing."
    )

    message = client.messages.create(
        model=model,
        max_tokens=400,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"Passages:\n\n{context_block}\n\nQuestion: {question}\n\n"
                           f"Answer precisely and concisely.",
            }
        ],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def generate_answer_extractive(passages):
    """Fallback when no API key is supplied: show short, precise snippets."""
    if not passages:
        return "No sufficiently relevant passage was found for that question."

    def trim(text, limit=220):
        text = text.strip()
        if len(text) <= limit:
            return text
        cut = text[:limit].rsplit(" ", 1)[0]
        return cut + " …"

    lines = ["**Top matching passages** (no API key set, so this is extractive, not generated):\n"]
    for i, p in enumerate(passages, start=1):
        lines.append(
            f"**[{i}] {p['source']} p.{p['page']}** (relevance {p['score']:.2f}) — {trim(p['text'])}"
        )
    return "\n\n".join(lines)


# -----------------------------
# FILE UPLOAD & INDEXING
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload PDF(s)", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    if new_files:
        with st.spinner(f"Reading and indexing {len(new_files)} file(s)..."):
            for f in new_files:
                pages = extract_pdf_pages(f)
                if not pages:
                    st.warning(f"⚠️ Couldn't extract any text from **{f.name}** "
                               f"(it may be a scanned/image-only PDF).")
                    continue
                new_chunks = chunk_pages(pages, f.name, chunk_size, chunk_overlap)
                st.session_state.chunks.extend(new_chunks)
                st.session_state.processed_files.append(f.name)

        if st.session_state.chunks:
            st.session_state.vectorizer, st.session_state.vectors = build_index(
                st.session_state.chunks
            )

    if st.session_state.processed_files:
        st.success(
            f"✅ Indexed {len(st.session_state.processed_files)} file(s), "
            f"{len(st.session_state.chunks)} chunks: "
            f"{', '.join(st.session_state.processed_files)}"
        )

has_index = st.session_state.vectorizer is not None and st.session_state.chunks

# -----------------------------
# CHAT INTERFACE
# -----------------------------
st.divider()

# Replay existing history as a chat log
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        if turn["sources"]:
            with st.expander("📌 Sources"):
                for i, p in enumerate(turn["sources"], start=1):
                    st.markdown(f"**[{i}] {p['source']} — p.{p['page']}** "
                                f"(relevance {p['score']:.2f})")
                    st.info(p["text"])

question = st.chat_input(
    "Ask a question about your PDF(s)..." if has_index else "Upload a PDF to get started"
)

if question:
    if not has_index:
        st.warning("Please upload at least one PDF first.")
    else:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant passages..."):
                passages = retrieve(
                    question,
                    st.session_state.vectorizer,
                    st.session_state.vectors,
                    st.session_state.chunks,
                    top_k,
                    min_similarity,
                )

            if api_key:
                with st.spinner("Generating answer with Claude..."):
                    try:
                        answer = generate_answer_with_claude(
                            question, passages, api_key, model_name
                        )
                    except Exception as e:
                        answer = (f"⚠️ Couldn't reach the Anthropic API ({e}). "
                                  f"Showing extractive passages instead.\n\n"
                                  + generate_answer_extractive(passages))
            else:
                answer = generate_answer_extractive(passages)

            st.markdown(answer)
            if passages:
                with st.expander("📌 Sources"):
                    for i, p in enumerate(passages, start=1):
                        st.markdown(f"**[{i}] {p['source']} — p.{p['page']}** "
                                    f"(relevance {p['score']:.2f})")
                        st.info(p["text"])

        st.session_state.history.append(
            {"question": question, "answer": answer, "sources": passages}
        )

# -----------------------------
# DOWNLOAD CHAT HISTORY
# -----------------------------
if st.session_state.history:
    transcript = "\n\n".join(
        f"Q: {t['question']}\nA: {t['answer']}" for t in st.session_state.history
    )
    st.sidebar.divider()
    st.sidebar.download_button(
        "⬇️ Download chat transcript",
        data=transcript,
        file_name="chat_transcript.txt",
        mime="text/plain",
        use_container_width=True,
    )
