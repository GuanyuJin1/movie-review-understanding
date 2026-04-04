import html
import re
import string
from typing import Iterable

WHITESPACE_RE = re.compile(r"\s+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
PUNCT_TRANSLATION_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """Apply lightweight normalization for course-project sentiment analysis."""
    normalized = html.unescape(str(text))
    normalized = normalized.lower()
    normalized = HTML_TAG_RE.sub(" ", normalized)
    normalized = normalized.translate(PUNCT_TRANSLATION_TABLE)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def batch_clean_text(texts: Iterable[str]) -> list[str]:
    """Apply preprocessing to a collection of texts."""
    return [clean_text(text) for text in texts]
