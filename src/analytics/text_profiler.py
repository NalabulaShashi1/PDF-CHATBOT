import math
import re
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.analytics.pdf_parser import ExtractedDocument


class ReadabilityScores(BaseModel):
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    reading_ease_level: str
    avg_words_per_sentence: float
    avg_syllables_per_word: float


class Keyphrase(BaseModel):
    phrase: str
    score: float
    frequency: int


class PageMetric(BaseModel):
    page_number: int
    word_count: int
    char_count: int
    reading_time_sec: float
    density_pct: float


class DocumentAnalytics(BaseModel):
    filename: str
    total_pages: int
    total_words: int
    total_sentences: int
    unique_words: int
    lexical_diversity_ttr: float
    readability: ReadabilityScores
    top_keyphrases: List[Keyphrase]
    page_metrics: List[PageMetric]
    extractive_summary: List[str]
    word_frequency_top20: Dict[str, int]


class TextProfiler:
    """Statistical NLP and Data Science Profiler for documents."""

    def __init__(self, top_n_keywords: int = 15, summary_sentences: int = 4):
        self.top_n_keywords = top_n_keywords
        self.summary_sentences = summary_sentences

    def profile_document(self, doc: ExtractedDocument) -> DocumentAnalytics:
        full_text = doc.full_text.strip()
        if not full_text:
            return self._empty_analytics(doc.filename)

        sentences = self._split_sentences(full_text)
        words = re.findall(r"\b[a-zA-Z]{2,}\b", full_text.lower())
        total_words = len(words)
        total_sentences = max(len(sentences), 1)

        unique_words = len(set(words))
        ttr = round((unique_words / max(total_words, 1)), 4)

        readability = self._calculate_readability(words, sentences)

        word_counts = pd.Series(words).value_counts()
        stopwords = self._get_stopwords()
        filtered_word_counts = word_counts[~word_counts.index.isin(stopwords)]
        top_20_words = filtered_word_counts.head(20).to_dict()

        keyphrases = self._extract_keyphrases(doc)

        page_metrics: List[PageMetric] = []
        for p in doc.pages:
            density_pct = round((p.word_count / max(doc.total_words, 1)) * 100, 2)
            page_metrics.append(PageMetric(
                page_number=p.page_number,
                word_count=p.word_count,
                char_count=p.char_count,
                reading_time_sec=p.reading_time_seconds,
                density_pct=density_pct
            ))

        summary = self._generate_extractive_summary(sentences, self.summary_sentences)

        return DocumentAnalytics(
            filename=doc.filename,
            total_pages=doc.total_pages,
            total_words=total_words,
            total_sentences=total_sentences,
            unique_words=unique_words,
            lexical_diversity_ttr=ttr,
            readability=readability,
            top_keyphrases=keyphrases,
            page_metrics=page_metrics,
            extractive_summary=summary,
            word_frequency_top20={str(k): int(v) for k, v in top_20_words.items()}
        )

    def _extract_keyphrases(self, doc: ExtractedDocument) -> List[Keyphrase]:
        corpus = [p.text for p in doc.pages if len(p.text.strip()) > 20]
        if not corpus:
            corpus = [doc.full_text]

        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 3),
                max_features=200,
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b"
            )
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            
            top_indices = scores.argsort()[::-1][:self.top_n_keywords]
            
            full_lower = doc.full_text.lower()
            keyphrases = []
            for idx in top_indices:
                phrase = str(feature_names[idx])
                score = round(float(scores[idx]), 4)
                count = len(re.findall(r"\b" + re.escape(phrase) + r"\b", full_lower))
                keyphrases.append(Keyphrase(phrase=phrase, score=score, frequency=count))
            return keyphrases
        except Exception:
            return []

    def _calculate_readability(self, words: List[str], sentences: List[str]) -> ReadabilityScores:
        total_words = max(len(words), 1)
        total_sentences = max(len(sentences), 1)

        syllables = sum(self._count_syllables(w) for w in words)
        avg_wps = round(total_words / total_sentences, 2)
        avg_spw = round(syllables / total_words, 2)

        flesch_ease = 206.835 - (1.015 * avg_wps) - (84.6 * avg_spw)
        flesch_ease = round(max(0.0, min(100.0, flesch_ease)), 1)

        fk_grade = (0.39 * avg_wps) + (11.8 * avg_spw) - 15.59
        fk_grade = round(max(0.0, fk_grade), 1)

        if flesch_ease >= 90:
            level = "Very Easy (5th Grade)"
        elif flesch_ease >= 80:
            level = "Easy (6th Grade)"
        elif flesch_ease >= 70:
            level = "Fairly Easy (7th Grade)"
        elif flesch_ease >= 60:
            level = "Standard / Plain English (8th-9th Grade)"
        elif flesch_ease >= 50:
            level = "Fairly Difficult (10th-12th Grade)"
        elif flesch_ease >= 30:
            level = "Difficult (College Level)"
        else:
            level = "Very Difficult / Academic (Graduate Level)"

        return ReadabilityScores(
            flesch_reading_ease=flesch_ease,
            flesch_kincaid_grade=fk_grade,
            reading_ease_level=level,
            avg_words_per_sentence=avg_wps,
            avg_syllables_per_word=avg_spw
        )

    def _generate_extractive_summary(self, sentences: List[str], n_sentences: int) -> List[str]:
        valid_sentences = [s.strip() for s in sentences if len(s.split()) >= 6 and len(s) < 350]
        if len(valid_sentences) <= n_sentences:
            return valid_sentences

        try:
            vec = TfidfVectorizer(stop_words="english")
            matrix = vec.fit_transform(valid_sentences)
            sentence_scores = np.asarray(matrix.sum(axis=1)).ravel()
            ranked_indices = sentence_scores.argsort()[::-1][:n_sentences]
            ranked_indices = sorted(ranked_indices)
            return [valid_sentences[i] for i in ranked_indices]
        except Exception:
            return valid_sentences[:n_sentences]

    @staticmethod
    def _count_syllables(word: str) -> int:
        word = word.lower()
        if len(word) <= 3:
            return 1
        word = re.sub(r"(?:[^laeiouy]|ed|es|e)$", "", word)
        word = re.sub(r"^y", "", word)
        matches = re.findall(r"[aeiouy]{1,2}", word)
        return max(len(matches), 1)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text)
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _get_stopwords() -> set:
        return {
            "the", "of", "and", "a", "to", "in", "is", "you", "that", "it", "he", "was",
            "for", "on", "are", "as", "with", "his", "they", "i", "at", "be", "this",
            "have", "from", "or", "one", "had", "by", "word", "but", "not", "what",
            "all", "were", "we", "when", "your", "can", "said", "there", "use", "an"
        }

    def _empty_analytics(self, filename: str) -> DocumentAnalytics:
        return DocumentAnalytics(
            filename=filename,
            total_pages=0,
            total_words=0,
            total_sentences=0,
            unique_words=0,
            lexical_diversity_ttr=0.0,
            readability=ReadabilityScores(
                flesch_reading_ease=0.0,
                flesch_kincaid_grade=0.0,
                reading_ease_level="Empty Document",
                avg_words_per_sentence=0.0,
                avg_syllables_per_word=0.0
            ),
            top_keyphrases=[],
            page_metrics=[],
            extractive_summary=[],
            word_frequency_top20={}
        )