"""WebVTT parsing and rolling-caption merge tests."""

from __future__ import annotations

from ybook.vtt import parse_vtt, vtt_to_text


def test_manual_captions_with_voice_and_entities() -> None:
    content = """WEBVTT
Kind: captions
Language: en

NOTE this note is ignored

cue-1
00:00:01.000 --> 00:00:03.500 align:start position:0%
<v Alice>Hello &amp; welcome.</v>

00:00:04.000 --> 00:00:06.000
This is a test.
"""
    cues = parse_vtt(content)
    assert len(cues) == 2
    assert cues[0].start == 1.0
    assert cues[0].end == 3.5
    assert cues[0].text == "Alice: Hello & welcome."
    assert vtt_to_text(content) == "Alice: Hello & welcome. This is a test."


def test_auto_rolling_captions_merge_to_longest_cue() -> None:
    content = """WEBVTT

00:00:00.000 --> 00:00:02.000
hello

00:00:01.500 --> 00:00:03.000
hello world

00:00:02.500 --> 00:00:04.000
hello world this is a test
"""
    cues = parse_vtt(content)
    assert len(cues) == 1
    assert cues[0].start == 0.0
    assert cues[0].end == 4.0
    assert vtt_to_text(content) == "hello world this is a test"


def test_auto_captions_with_overlap_are_joined() -> None:
    content = """WEBVTT

00:00:00.000 --> 00:00:02.000
one two three

00:00:02.000 --> 00:00:04.000
two three four
"""
    assert vtt_to_text(content) == "one two three four"


def test_single_word_overlap_merges_when_cues_overlap_in_time() -> None:
    content = """WEBVTT

00:00:00.000 --> 00:00:02.000
hello world

00:00:01.900 --> 00:00:04.000
world this is
"""
    assert vtt_to_text(content) == "hello world this is"


def test_single_word_overlap_does_not_merge_without_timing_overlap() -> None:
    content = """WEBVTT

00:00:00.000 --> 00:00:01.000
hello world

00:00:03.000 --> 00:00:05.000
world this is
"""
    assert vtt_to_text(content) == "hello world world this is"


def test_inline_tags_timestamps_and_entities_are_removed() -> None:
    content = """WEBVTT

00:00:01.000 --> 00:00:02.000
<c.yellow>word</c> <00:00:01.250>next</00:00:01.250> <b>bold &amp; bright</b>
"""
    assert parse_vtt(content)[0].text == "word next bold & bright"


def test_sound_only_cues_are_dropped_but_unclear_is_kept() -> None:
    content = """WEBVTT

00:00:01.000 --> 00:00:02.000
[Music]

00:00:02.000 --> 00:00:03.000
[Applause]

00:00:03.000 --> 00:00:04.000
[Музыка]

00:00:04.000 --> 00:00:05.000
♪

00:00:05.000 --> 00:00:06.000
(upbeat music)

00:00:06.000 --> 00:00:07.000
[unclear]
"""
    cues = parse_vtt(content)
    assert len(cues) == 1
    assert cues[0].text == "[unclear]"


def test_multiline_cue_and_exact_duplicate() -> None:
    content = """WEBVTT

00:00:01.000 --> 00:00:02.000
first line
second line

00:00:02.000 --> 00:00:03.000
first line second line
"""
    cues = parse_vtt(content)
    assert len(cues) == 1
    assert cues[0].text == "first line second line"
    assert cues[0].end == 3.0


def test_repeated_speaker_label_is_not_repeated_in_text() -> None:
    content = """WEBVTT

00:00:01.000 --> 00:00:02.000
<v Alice>Hello</v>

00:00:02.000 --> 00:00:03.000
<v Alice>world</v>

00:00:03.000 --> 00:00:04.000
<v Bob>Hi there</v>
"""
    assert vtt_to_text(content) == "Alice: Hello world Bob: Hi there"


def test_malformed_timing_is_skipped() -> None:
    content = """WEBVTT

not-a-timestamp --> 00:00:02.000
bad cue

00:00:03.000 --> 00:00:04.000
good cue
"""
    cues = parse_vtt(content)
    assert len(cues) == 1
    assert cues[0].text == "good cue"


def test_empty_vtt_returns_empty() -> None:
    assert parse_vtt("") == []
    assert vtt_to_text("WEBVTT\n") == ""
