import io
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import pypdf

try:
    import pikepdf
    HAS_PIKEPDF = True
except ImportError:
    HAS_PIKEPDF = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False


class PageContent(BaseModel):
    page_number: int
    text: str
    char_count: int
    word_count: int
    line_count: int
    reading_time_seconds: float
    is_scanned: bool = False


class ExtractedDocument(BaseModel):
    filename: str
    total_pages: int
    total_words: int
    total_characters: int
    estimated_reading_time_minutes: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    pages: List[PageContent] = Field(default_factory=list)
    full_text: str = ""
    is_mostly_scanned: bool = False
    extraction_warning: Optional[str] = None


class PDFParser:
    """Enterprise-Grade Multi-Tier PDF Parser with Auto-Repair and Decryption."""

    def __init__(self, avg_wpm: int = 200):
        self.avg_wpm = avg_wpm

    def parse(self, source: Union[str, Path, bytes, io.BytesIO], filename: str = "document.pdf") -> ExtractedDocument:
        raw_bytes = self._to_bytes(source)
        if not raw_bytes or len(raw_bytes) < 10:
            raise ValueError("The provided PDF file is empty (0 bytes).")

        errors = []

        # Tier 1: Direct PyPDF parse
        try:
            doc = self._parse_with_pypdf(raw_bytes, filename)
            if doc.total_pages > 0:
                return doc
        except Exception as e:
            errors.append(f"PyPDF: {str(e)}")

        # Tier 2: Auto-repair corrupted xref with PikePDF
        if HAS_PIKEPDF:
            try:
                repaired_bytes = self._repair_with_pikepdf(raw_bytes)
                doc = self._parse_with_pypdf(repaired_bytes, filename)
                if doc.total_pages > 0:
                    return doc
            except Exception as e:
                errors.append(f"PikePDF Repair: {str(e)}")

        # Tier 3: PyPDF2 fallback
        if HAS_PYPDF2:
            try:
                doc = self._parse_with_pypdf2(raw_bytes, filename)
                if doc.total_pages > 0:
                    return doc
            except Exception as e:
                errors.append(f"PyPDF2: {str(e)}")

        raise ValueError(f"Unable to parse PDF after multiple recovery attempts. ({'; '.join(errors)})")

    def _to_bytes(self, source: Union[str, Path, bytes, io.BytesIO]) -> bytes:
        if isinstance(source, (str, Path)):
            with open(source, "rb") as f:
                return f.read()
        elif isinstance(source, bytes):
            return source
        elif isinstance(source, io.BytesIO):
            source.seek(0)
            return source.read()
        raise ValueError("Unsupported source format for PDF.")

    def _repair_with_pikepdf(self, raw_bytes: bytes) -> bytes:
        stream_in = io.BytesIO(raw_bytes)
        with pikepdf.open(stream_in, allow_overwriting_input=True) as pdf:
            stream_out = io.BytesIO()
            pdf.save(stream_out)
            return stream_out.getvalue()

    def _parse_with_pypdf(self, raw_bytes: bytes, filename: str) -> ExtractedDocument:
        stream = io.BytesIO(raw_bytes)
        reader = pypdf.PdfReader(stream, strict=False)

        if reader.is_encrypted:
            for pwd in ["", b"", " ", "owner", "admin"]:
                try:
                    reader.decrypt(pwd)
                    break
                except Exception:
                    pass

        total_pages = len(reader.pages)
        raw_metadata = reader.metadata or {}
        
        clean_metadata = {}
        for k, v in raw_metadata.items():
            clean_key = str(k).lstrip("/").lower()
            clean_metadata[clean_key] = str(v) if v is not None else ""

        pages_list: List[PageContent] = []
        full_text_parts: List[str] = []
        total_words = 0
        total_chars = 0
        blank_pages = 0

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            raw_text = ""
            try:
                raw_text = page.extract_text() or ""
            except Exception:
                try:
                    raw_text = page.extract_text(extraction_mode="layout") or ""
                except Exception:
                    raw_text = ""

            cleaned_text = self._clean_text(raw_text)
            words = re.findall(r"\b\w+\b", cleaned_text)
            w_count = len(words)
            c_count = len(cleaned_text)
            l_count = len([line for line in cleaned_text.splitlines() if line.strip()])
            read_time_sec = round((w_count / max(self.avg_wpm, 1)) * 60, 1)
            is_blank = (w_count < 5)
            if is_blank:
                blank_pages += 1

            page_obj = PageContent(
                page_number=page_num,
                text=cleaned_text,
                char_count=c_count,
                word_count=w_count,
                line_count=l_count,
                reading_time_seconds=read_time_sec,
                is_scanned=is_blank
            )
            pages_list.append(page_obj)
            if cleaned_text:
                full_text_parts.append(cleaned_text)
            total_words += w_count
            total_chars += c_count

        full_text = "\n\n".join(full_text_parts)
        est_read_min = round(total_words / max(self.avg_wpm, 1), 2)
        
        is_scanned = (blank_pages == total_pages and total_pages > 0) or (total_words < 15 and total_pages > 0)
        warning_msg = None
        if is_scanned:
            warning_msg = "This PDF appears to be a scanned image or photo without selectable digital text."

        return ExtractedDocument(
            filename=filename,
            total_pages=total_pages,
            total_words=total_words,
            total_characters=total_chars,
            estimated_reading_time_minutes=est_read_min,
            metadata=clean_metadata,
            pages=pages_list,
            full_text=full_text,
            is_mostly_scanned=is_scanned,
            extraction_warning=warning_msg
        )

    def _parse_with_pypdf2(self, raw_bytes: bytes, filename: str) -> ExtractedDocument:
        stream = io.BytesIO(raw_bytes)
        reader = PyPDF2.PdfReader(stream)
        
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                pass

        total_pages = len(reader.pages)
        pages_list: List[PageContent] = []
        full_text_parts: List[str] = []
        total_words = 0
        total_chars = 0

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            raw_text = page.extract_text() or ""
            cleaned_text = self._clean_text(raw_text)
            words = re.findall(r"\b\w+\b", cleaned_text)
            w_count = len(words)
            c_count = len(cleaned_text)
            l_count = len([line for line in cleaned_text.splitlines() if line.strip()])
            read_time_sec = round((w_count / max(self.avg_wpm, 1)) * 60, 1)

            page_obj = PageContent(
                page_number=page_num,
                text=cleaned_text,
                char_count=c_count,
                word_count=w_count,
                line_count=l_count,
                reading_time_seconds=read_time_sec,
                is_scanned=(w_count < 5)
            )
            pages_list.append(page_obj)
            if cleaned_text:
                full_text_parts.append(cleaned_text)
            total_words += w_count
            total_chars += c_count

        return ExtractedDocument(
            filename=filename,
            total_pages=total_pages,
            total_words=total_words,
            total_characters=total_chars,
            estimated_reading_time_minutes=round(total_words / max(self.avg_wpm, 1), 2),
            metadata={},
            pages=pages_list,
            full_text="\n\n".join(full_text_parts),
            is_mostly_scanned=(total_words < 15 and total_pages > 0)
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        text = text.replace("\xa0", " ").replace("\u200b", "")
        text = re.sub(r"\n{3,}", "\n\n", text)
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()