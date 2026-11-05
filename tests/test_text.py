"""Text normalization, chunking, and paragraph tests."""

from __future__ import annotations

import pytest

from ybook.text import (
    chunk_text,
    condense_for_prompt,
    extract_numeric_tokens,
    format_duration,
    looks_truncated,
    normalize_cleaned_text,
    normalize_transcript,
    to_paragraphs,
    word_count,
)


def test_normalize_transcript_removes_zero_width_and_collapses_space() -> None:
    value = "  hello\u200b\r\nworld\u00a0  again\ufeff "
    assert normalize_transcript(value) == "hello world again"


def test_normalize_cleaned_text_preserves_paragraphs_and_rejoins_wrapped_lines() -> None:
    value = "First line\nwrapped here.\n\n\nSecond paragraph.  \n"
    assert normalize_cleaned_text(value) == "First line wrapped here.\n\nSecond paragraph."


def test_word_count() -> None:
    assert word_count("one two  three\nfour") == 4
    assert word_count("") == 0


def test_chunk_text_respects_limit_and_keeps_all_words() -> None:
    words = [f"word{index}" for index in range(1005)]
    text = " ".join(words)
    chunks = chunk_text(text, max_words=100)

    assert len(chunks) > 1
    assert all(word_count(chunk) <= 100 for chunk in chunks)
    assert sum(word_count(chunk) for chunk in chunks) == 1005
    assert word_count(chunks[-1]) > 5


def test_chunk_text_prefers_sentence_boundaries() -> None:
    text = " ".join(f"Sentence number {index}." for index in range(30))
    chunks = chunk_text(text, max_words=20)

    assert all(word_count(chunk) <= 20 for chunk in chunks)
    assert all(chunk.rstrip().endswith(".") for chunk in chunks)
    assert " ".join(chunks).split() == text.split()


def test_chunk_text_empty_and_invalid() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n ") == []
    with pytest.raises(ValueError):
        chunk_text("hello", max_words=0)


def test_to_paragraphs_splits_blank_lines() -> None:
    text = "First paragraph.\n\nSecond paragraph."
    assert to_paragraphs(text) == ["First paragraph.", "Second paragraph."]


def test_to_paragraphs_groups_many_sentences() -> None:
    text = " ".join(f"Sentence {index}." for index in range(12))
    paragraphs = to_paragraphs(text)
    assert len(paragraphs) == 3
    assert all(len(paragraph.split(".")) <= 6 for paragraph in paragraphs)


def test_to_paragraphs_splits_unpunctuated_wall_of_text() -> None:
    text = " ".join(f"word{index}" for index in range(500))
    paragraphs = to_paragraphs(text)
    assert len(paragraphs) >= 3
    assert all(word_count(paragraph) <= 180 for paragraph in paragraphs)


def test_format_duration() -> None:
    assert format_duration(None) == ""
    assert format_duration(5) == "0:05"
    assert format_duration(65) == "1:05"
    assert format_duration(3661) == "1:01:01"
    assert format_duration(-10) == "0:00"


def test_extract_numeric_tokens_normalizes_separators() -> None:
    tokens = extract_numeric_tokens("It cost 1,000 dollars in 2024 at 3.14 percent.")
    assert tokens == {"1000", "2024", "3.14"}


def test_extract_numeric_tokens_handles_european_formats_and_padding() -> None:
    tokens = extract_numeric_tokens("Values: 1.000, 3,14, 01 and 05.")
    assert tokens == {"1000", "3.14", "1", "5"}


def test_extract_numeric_tokens_ignores_iso_dates() -> None:
    assert extract_numeric_tokens("The date is 2024-01-05, total 42.") == {"42"}


def test_looks_truncated() -> None:
    assert not looks_truncated("A complete sentence.")
    assert not looks_truncated("A question?")
    assert looks_truncated("A sentence that was cut off mid wor")
    assert not looks_truncated("")


def test_condense_for_prompt_keeps_short_text_unchanged() -> None:
    text = "one two three four five"
    assert condense_for_prompt(text, max_words=10) == text


def test_condense_for_prompt_keeps_first_and_last_half_at_word_boundaries() -> None:
    words = [f"word{index}" for index in range(20)]
    text = " ".join(words)
    result = condense_for_prompt(text, max_words=10)
    first, last = result.split("\n\n[...]\n\n")
    assert first.split() == words[:5]
    assert last.split() == words[-5:]
    assert "word7" not in result
    assert "word10" not in result


def test_condense_for_prompt_odd_budget_and_validation() -> None:
    text = " ".join(f"w{index}" for index in range(12))
    result = condense_for_prompt(text, max_words=5)
    first, last = result.split("\n\n[...]\n\n")
    assert len(first.split()) == 2
    assert len(last.split()) == 3
    with pytest.raises(ValueError):
        condense_for_prompt(text, max_words=1)
