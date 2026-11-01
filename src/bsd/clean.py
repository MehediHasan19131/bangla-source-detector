"""Text cleaning shared by every data source (same rules as the CSE437 paper)."""

import re
import unicodedata
from collections.abc import Iterable

BANGLA_CHAR = re.compile(r"[ঀ-৿]")
WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Unicode NFC normalization plus whitespace collapsing."""
    return WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def bangla_ratio(text: str) -> float:
    """Share of non-space characters that are in the Bangla Unicode block."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if BANGLA_CHAR.match(c)) / len(chars)


def is_valid(text: str, min_chars: int = 100, max_chars: int = 3000, min_bangla: float = 0.7) -> bool:
    return min_chars <= len(text) <= max_chars and bangla_ratio(text) >= min_bangla


def clean_texts(texts: Iterable[str], **filters) -> list[str]:
    """Normalize, filter out noisy/non-Bangla texts and drop exact duplicates (order preserved)."""
    seen: set[str] = set()
    out = []
    for raw in texts:
        if not raw:
            continue
        text = normalize(raw)
        if text in seen or not is_valid(text, **filters):
            continue
        seen.add(text)
        out.append(text)
    return out
