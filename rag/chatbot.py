import os
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.config import GEMINI_API_KEY, DEFAULT_TOP_K, HYBRID_ALPHA
from src.analytics.pdf_parser import ExtractedDocument
from src.rag.chunker import DocumentChunker, TextChunk
from src.rag.vector_store import VectorStore, SearchResult
from src.rag.hybrid_retriever import HybridRetriever


class Citation(BaseModel):
    page_number: int
    chunk_id: str
    score: float
    excerpt: str


class ChatMessage(BaseModel):
    role: str
    content: str
    citations: List[Citation] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    model_used: str
    tokens_used: Optional[int] = None
    retrieval_count: int


class SmartPDFChatbot:
    def __init__(self, api_key: Optional[str] = None, top_k: int = DEFAULT_TOP_K, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.top_k = top_k
        self.chunker = DocumentChunker()
        self.vector_store = VectorStore()
        self.retriever = HybridRetriever(self.vector_store, alpha=HYBRID_ALPHA)
        
        self.current_doc: Optional[ExtractedDocument] = None
        self.all_chunks: List[TextChunk] = []
        self.history: List[ChatMessage] = []
        self._gemini_client = None

        self._init_llm_client()

    def set_config(self, api_key: Optional[str] = None, model_name: Optional[str] = None, top_k: Optional[int] = None):
        if api_key is not None:
            self.api_key = api_key.strip()
        if model_name is not None:
            self.model_name = model_name
        if top_k is not None:
            self.top_k = top_k
        self._init_llm_client()

    def _init_llm_client(self):
        if self.api_key and self.api_key.strip() and self.model_name != "extractive-nlp-engine":
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.api_key)
            except Exception:
                self._gemini_client = None
        else:
            self._gemini_client = None

    def load_document(self, doc: ExtractedDocument) -> int:
        self.current_doc = doc
        self.all_chunks = self.chunker.chunk_document(doc)
        self.retriever.index(self.all_chunks)
        self.history = []
        return len(self.all_chunks)

    def answer_question(self, question: str, top_k: Optional[int] = None) -> ChatResponse:
        k = top_k or self.top_k
        question = question.strip()
        if not question:
            return ChatResponse(answer="Please enter a valid question.", citations=[], model_used="none", retrieval_count=0)

        if not self.all_chunks:
            return ChatResponse(answer="No document has been loaded yet. Please upload a PDF first.", citations=[], model_used="none", retrieval_count=0)

        search_hits = self.retriever.retrieve(question, top_k=k)
        citations = self._build_citations(search_hits)

        if self._gemini_client and self.model_name != "extractive-nlp-engine":
            try:
                answer = self._generate_with_gemini(question, search_hits)
                model_used = self.model_name
            except Exception as e:
                answer = self._generate_extractive(question, search_hits, fallback_msg=str(e))
                model_used = "extractive-fallback"
        else:
            answer = self._generate_extractive(question, search_hits)
            model_used = "extractive-nlp-engine"

        self.history.append(ChatMessage(role="user", content=question))
        self.history.append(ChatMessage(role="assistant", content=answer, citations=citations))

        return ChatResponse(
            answer=answer,
            citations=citations,
            model_used=model_used,
            retrieval_count=len(search_hits)
        )

    def summarize_document(self, focus: str = "general") -> ChatResponse:
        if not self.current_doc:
            return ChatResponse(answer="No document loaded to summarize.", citations=[], model_used="none", retrieval_count=0)

        sample_text = self.current_doc.full_text[:4000]
        if self._gemini_client and self.model_name != "extractive-nlp-engine":
            try:
                prompt = (
                    f"You are an expert AI document analyst. Provide a structured, clear summary of this document.\n"
                    f"Include: 1) Executive Summary, 2) Key Takeaways, 3) Important Data Points.\n\n"
                    f"Document excerpt:\n{sample_text}"
                )
                res = self._gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                return ChatResponse(
                    answer=res.text or "Summary generated.",
                    citations=[],
                    model_used=self.model_name,
                    retrieval_count=len(self.all_chunks)
                )
            except Exception:
                pass

        from src.analytics.text_profiler import TextProfiler
        profiler = TextProfiler()
        analytics = profiler.profile_document(self.current_doc)
        
        summary_lines = []
        if analytics.top_keyphrases:
            topics = ", ".join([kp.phrase.title() for kp in analytics.top_keyphrases[:5]])
            summary_lines.append(f"**Key Topics:** {topics}\n")

        summary_lines.append("**Summary Overview:**")
        for sentence in analytics.extractive_summary:
            summary_lines.append(f"• {sentence}")

        return ChatResponse(
            answer="\n".join(summary_lines),
            citations=[],
            model_used="extractive-nlp-profiler",
            retrieval_count=len(self.all_chunks)
        )

    def _generate_with_gemini(self, question: str, hits: List[SearchResult]) -> str:
        context_blocks = [f"[Page {h.chunk.page_number}]: {h.chunk.text}" for h in hits]
        context_str = "\n\n".join(context_blocks)

        prompt = (
            "You are an intelligent document assistant. "
            "Provide a direct, clear, and comprehensive answer to the question using ONLY the provided document context below.\n"
            "Do not describe how you extracted the information or include technical debug details. "
            "Simply answer the user's question directly in a natural and professional tone.\n\n"
            f"--- DOCUMENT CONTEXT ---\n{context_str}\n\n"
            f"--- QUESTION ---\n{question}\n\n"
            "--- ANSWER ---"
        )
        response = self._gemini_client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text.strip() if response.text else "Unable to generate answer from the provided document."

    def _generate_extractive(self, question: str, hits: List[SearchResult], fallback_msg: str = "") -> str:
        if not hits:
            return "No relevant information found in the document for your query."

        q_terms = set(re.findall(r"\b\w{3,}\b", question.lower()))
        collected_sentences = []
        seen_sentences = set()

        for hit in hits:
            text = hit.chunk.text.strip()
            sentences = re.split(r"(?<=[.!?])\s+", text)
            
            scored_sentences = []
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean.split()) < 4 or s_clean in seen_sentences:
                    continue
                s_terms = set(re.findall(r"\b\w{3,}\b", s_clean.lower()))
                match_count = len(q_terms.intersection(s_terms))
                if match_count > 0:
                    scored_sentences.append((match_count, s_clean))
            
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            for _, s_text in scored_sentences[:2]:
                if s_text not in seen_sentences:
                    seen_sentences.add(s_text)
                    collected_sentences.append(s_text)

        if not collected_sentences:
            top_text = hits[0].chunk.text.strip()
            first_sentence = re.split(r"(?<=[.!?])\s+", top_text)[0]
            collected_sentences.append(first_sentence)

        if len(collected_sentences) == 1:
            return collected_sentences[0]
        else:
            return "\n\n".join(collected_sentences)

    def _build_citations(self, hits: List[SearchResult]) -> List[Citation]:
        citations = []
        for h in hits:
            excerpt = h.chunk.text[:180] + "..." if len(h.chunk.text) > 180 else h.chunk.text
            citations.append(Citation(
                page_number=h.chunk.page_number,
                chunk_id=h.chunk.chunk_id,
                score=round(h.score, 4),
                excerpt=excerpt
            ))
        return citations

    def get_history(self) -> List[ChatMessage]:
        return self.history

    def clear_history(self):
        self.history = []