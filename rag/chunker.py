import re
from typing import List
from pydantic import BaseModel
from src.analytics.pdf_parser import ExtractedDocument


class TextChunk(BaseModel):
    chunk_id: str
    filename: str
    page_number: int
    text: str
    word_count: int
    char_length: int


class DocumentChunker:
    """Intelligent sentence-aware document chunker with sliding window overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = max(chunk_size, 150)
        self.chunk_overlap = min(chunk_overlap, self.chunk_size // 2)

    def chunk_document(self, doc: ExtractedDocument) -> List[TextChunk]:
        chunks: List[TextChunk] = []
        global_chunk_idx = 0

        for page in doc.pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            page_chunks = self._chunk_page(page_text, page.page_number, doc.filename, global_chunk_idx)
            chunks.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

        return chunks

    def _chunk_page(self, text: str, page_num: int, filename: str, start_idx: int) -> List[TextChunk]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        page_chunks: List[TextChunk] = []
        
        current_chunk_sentences = []
        current_len = 0
        local_idx = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            s_len = len(sentence)
            if current_len + s_len > self.chunk_size and current_chunk_sentences:
                chunk_str = " ".join(current_chunk_sentences)
                words = re.findall(r"\b\w+\b", chunk_str)
                chunk_id = f"{filename}_p{page_num}_c{start_idx + local_idx}"
                
                page_chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    filename=filename,
                    page_number=page_num,
                    text=chunk_str,
                    word_count=len(words),
                    char_length=len(chunk_str)
                ))
                local_idx += 1

                overlap_sentences = []
                overlap_len = 0
                for prev_s in reversed(current_chunk_sentences):
                    if overlap_len + len(prev_s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, prev_s)
                        overlap_len += len(prev_s)
                    else:
                        break
                
                current_chunk_sentences = overlap_sentences
                current_len = overlap_len

            current_chunk_sentences.append(sentence)
            current_len += s_len

        if current_chunk_sentences:
            chunk_str = " ".join(current_chunk_sentences)
            words = re.findall(r"\b\w+\b", chunk_str)
            chunk_id = f"{filename}_p{page_num}_c{start_idx + local_idx}"
            page_chunks.append(TextChunk(
                chunk_id=chunk_id,
                filename=filename,
                page_number=page_num,
                text=chunk_str,
                word_count=len(words),
                char_length=len(chunk_str)
            ))

        return page_chunks