"""Transcript normalization, sentence-aware chunking, and paragraphing."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
_WHITESPACE = re.compile(r"\s+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")
_WORD = re.compile(r"\S+")
_NUMBER = re.compile(r"\d[\d,.]*")
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_TERMINAL_PUNCTUATION = set(".!?…\"'»”’)]}")


def normalize_transcript(text: str) -> str:
    """Normalize a transcript into a single readable line.

    Removes zero-width characters and BOMs, normalizes line endings, collapses
    all whitespace (including cue line breaks), and trims the result.
    """

    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    cleaned = _ZERO_WIDTH.sub("", cleaned)
    return _WHITESPACE.sub(" ", cleaned).strip()


def normalize_cleaned_text(text: str) -> str:
    """Normalize LLM output while preserving intentional paragraph breaks."""

    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    cleaned = _ZERO_WIDTH.sub("", cleaned)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.split("\n")]
    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in lines:
        if line:
            buffer.append(line)
            continue
        if buffer:
            paragraphs.append(" ".join(buffer))
            buffer = []
    if buffer:
        paragraphs.append(" ".join(buffer))
    return "\n\n".join(paragraphs).strip()


def word_count(text: str) -> int:
    """Count whitespace-delimited words."""

    return len(_WORD.findall(text or ""))


def chunk_text(text: str, max_words: int = 2500) -> list[str]:
    """Split text into chunks of at most ``max_words`` words.

    Paragraph and sentence boundaries are preferred; only an individual
    oversized sentence is split on word boundaries.  A tiny final chunk is
    rebalanced with its predecessor so the LLM never receives a fragment.
    """

    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    if not text or not text.strip():
        return []

    pieces = _split_into_pieces(text, max_words)
    chunks = _pack_pieces(pieces, max_words)
    return _rebalance_tail(chunks, max_words)


def to_paragraphs(text: str, *, sentences_per_paragraph: int = 5) -> list[str]:
    """Convert cleaned text to display paragraphs.

    Blank-line separated blocks are kept together.  A block longer than
    ``sentences_per_paragraph`` sentences is split into groups of that size;
    blocks without sentence punctuation fall back to roughly 180-word groups.
    """

    if not text or not text.strip():
        return []

    normalized = normalize_cleaned_text(text)
    if normalized:
        blocks = [block.strip() for block in normalized.split("\n\n") if block.strip()]
    else:
        blocks = []

    paragraphs: list[str] = []
    for block in blocks:
        sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(block) if part.strip()]
        if len(sentences) > sentences_per_paragraph:
            for start in range(0, len(sentences), sentences_per_paragraph):
                paragraphs.append(" ".join(sentences[start : start + sentences_per_paragraph]))
            continue
        if len(sentences) == 1 and word_count(block) > 220:
            paragraphs.extend(_split_words_into_groups(block, 180))
            continue
        paragraphs.append(block)

    return paragraphs


def format_duration(seconds: int | float | None) -> str:
    """Format seconds as ``H:MM:SS`` or ``M:SS``."""

    if seconds is None:
        return ""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def extract_numeric_tokens(text: str) -> set[str]:
    """Extract normalized number tokens for the cleaning quality guard.

    Thousands separators and decimal commas are canonicalized, zero-padding is
    removed, and ISO dates are ignored so rewriting a date in words does not
    look like a dropped number.
    """

    without_dates = _ISO_DATE.sub(" ", text or "")
    tokens: set[str] = set()
    for raw in _NUMBER.findall(without_dates):
        normalized = _normalize_number_token(raw)
        if normalized is not None:
            tokens.add(normalized)
    return tokens


def _normalize_number_token(raw: str) -> str | None:
    token = raw.rstrip(".,")
    if not token:
        return None

    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            integer, fraction = token.rsplit(",", 1)
            integer = integer.replace(".", "")
        else:
            integer, fraction = token.rsplit(".", 1)
            integer = integer.replace(",", "")
        token = f"{integer}.{fraction}"
    elif "," in token:
        parts = token.split(",")
        if all(len(part) == 3 for part in parts[1:]):
            token = "".join(parts)
        else:
            integer, fraction = token.rsplit(",", 1)
            token = f"{integer.replace(',', '')}.{fraction}"
    elif "." in token:
        parts = token.split(".")
        if all(len(part) == 3 for part in parts[1:]):
            token = "".join(parts)

    try:
        value = Decimal(token)
    except InvalidOperation:
        return None
    return format(value.normalize(), "f")


def looks_truncated(text: str) -> bool:
    """Heuristically detect LLM output cut off without terminal punctuation."""

    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped[-1] in _TERMINAL_PUNCTUATION:
        return False
    return bool(re.search(r"\w$", stripped))


def condense_for_prompt(
    text: str,
    max_words: int = 12000,
    *,
    marker: str = "\n\n[...]\n\n",
) -> str:
    """Fit text into a word budget by keeping its beginning and end.

    Text within the budget is returned unchanged.  Longer text is reduced to
    roughly the first half and last half of the budget, joined by ``marker``.
    Splits always happen between words.
    """

    if max_words < 2:
        raise ValueError("max_words must be at least 2")
    words = (text or "").split()
    if len(words) <= max_words:
        return text
    first_count = max_words // 2
    last_count = max_words - first_count
    return " ".join(words[:first_count]) + marker + " ".join(words[-last_count:])


def _split_into_pieces(text: str, max_words: int) -> list[tuple[str, bool]]:
    """Return ``(text, is_paragraph)`` pieces, splitting oversized content."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [block.strip() for block in re.split(r"\n\s*\n", normalized) if block.strip()]
    pieces: list[tuple[str, bool]] = []

    for block in blocks:
        block = normalize_transcript(block)
        if not block:
            continue
        if word_count(block) <= max_words:
            pieces.append((block, True))
            continue

        sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(block) if part.strip()]
        for sentence in sentences:
            if word_count(sentence) <= max_words:
                pieces.append((sentence, False))
                continue
            words = sentence.split()
            for start in range(0, len(words), max_words):
                pieces.append((" ".join(words[start : start + max_words]), False))

    return pieces


def _pack_pieces(pieces: list[tuple[str, bool]], max_words: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    current_words = 0

    for piece, is_paragraph in pieces:
        piece_words = word_count(piece)
        if current and current_words + piece_words > max_words:
            chunks.append(current)
            current = ""
            current_words = 0
        if not current:
            current = piece
        elif is_paragraph:
            current = f"{current}\n\n{piece}"
        else:
            current = f"{current} {piece}"
        current_words += piece_words

    if current:
        chunks.append(current)
    return chunks


def _rebalance_tail(chunks: list[str], max_words: int) -> list[str]:
    if len(chunks) < 2:
        return chunks

    tiny_threshold = max(1, min(100, max_words // 10))
    if word_count(chunks[-1]) >= tiny_threshold:
        return chunks

    combined = f"{chunks[-2]} {chunks[-1]}"
    combined_words = combined.split()
    if len(combined_words) <= max_words:
        return [*chunks[:-2], " ".join(combined_words)]

    split_at = len(combined_words) // 2
    return [
        *chunks[:-2],
        " ".join(combined_words[:split_at]),
        " ".join(combined_words[split_at:]),
    ]


def _split_words_into_groups(text: str, group_size: int) -> list[str]:
    words = text.split()
    return [
        " ".join(words[start : start + group_size]) for start in range(0, len(words), group_size)
    ]
