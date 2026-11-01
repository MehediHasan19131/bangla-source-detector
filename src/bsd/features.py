"""The 15 interpretable surface/structure features from the CSE437 paper."""

import re
import unicodedata

import numpy as np

from .clean import bangla_ratio

SENTENCE_END = re.compile(r"[।.!?]+")
DIGIT = re.compile(r"[0-9০-৯]")
LATIN = re.compile(r"[A-Za-z]")
QUOTES = set("\"'“”‘’")

FEATURE_NAMES = [
    "char_len",
    "word_count",
    "avg_word_len",
    "sentence_count",
    "avg_sentence_len",
    "type_token_ratio",
    "punct_ratio",
    "danda_count",
    "period_count",
    "comma_count",
    "question_count",
    "quote_count",
    "digit_ratio",
    "latin_ratio",
    "bangla_ratio",
]


def extract(text: str) -> dict[str, float]:
    words = text.split()
    n_words = len(words) or 1
    n_chars = len(text) or 1
    non_space = sum(1 for c in text if not c.isspace()) or 1
    sentences = [s for s in SENTENCE_END.split(text) if s.strip()]
    n_sent = len(sentences) or 1
    return {
        "char_len": len(text),
        "word_count": len(words),
        "avg_word_len": sum(len(w) for w in words) / n_words,
        "sentence_count": len(sentences),
        "avg_sentence_len": len(words) / n_sent,
        "type_token_ratio": len(set(words)) / n_words,
        "punct_ratio": sum(1 for c in text if unicodedata.category(c).startswith("P")) / n_chars,
        "danda_count": text.count("।"),
        "period_count": text.count("."),
        "comma_count": text.count(","),
        "question_count": text.count("?"),
        "quote_count": sum(1 for c in text if c in QUOTES),
        "digit_ratio": len(DIGIT.findall(text)) / non_space,
        "latin_ratio": len(LATIN.findall(text)) / non_space,
        "bangla_ratio": bangla_ratio(text),
    }


def matrix(texts: list[str]) -> np.ndarray:
    return np.array([[extract(t)[name] for name in FEATURE_NAMES] for t in texts], dtype=float)
