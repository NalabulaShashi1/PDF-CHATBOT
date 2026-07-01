import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Smart PDF Chatbot", page_icon="📄")

st.title("📄 Smart PDF Chatbot")
st.write("Upload a PDF and ask questions from it.")

# -----------------------------
# SESSION STATE (CHAT HISTORY)
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# FUNCTION: READ PDF
# -----------------------------
def extract_pdf_text(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text

# -----------------------------
# FUNCTION: CREATE CHUNKS
# -----------------------------
def create_chunks(text):
    sentences = text.split(".")
    chunks = []

    for s in sentences:
        s = s.strip()
        if len(s) > 20:
            chunks.append(s)

    return chunks

# -----------------------------
# FUNCTION: RETRIEVE BEST MATCH
# -----------------------------
def retrieve_answer(chunks, question):
    docs = chunks + [question]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(docs)

    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    best_index = similarity.argmax()

    return chunks[best_index]

# -----------------------------
# FUNCTION: GENERATE FINAL ANSWER
# -----------------------------
def generate_answer(context, question):
    return f"""
📌 Based on the document:

{context}

💡 Answer:
This section explains that {context.lower()}.
"""

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading PDF..."):
        text = extract_pdf_text(uploaded_file)

    chunks = create_chunks(text)

    st.success("PDF loaded successfully!")

    question = st.text_input("Ask your question:")

    if question:
        with st.spinner("Finding best answer..."):
            context = retrieve_answer(chunks, question)

        answer = generate_answer(context, question)

        # Save history
        st.session_state.history.append((question, answer))

        # Display answer
        st.subheader("🤖 Answer")
        st.write(answer)

        st.subheader("📌 Source Text")
        st.info(context)

# -----------------------------
# CHAT HISTORY
# -----------------------------
if st.session_state.history:
    st.subheader("🕘 Previous Questions")

    for q, a in reversed(st.session_state.history):
        st.write(f"**Q:** {q}")
        st.write(f"**A:** {a}")
        st.write("---")