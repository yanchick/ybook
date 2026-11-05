"""WebVTT parsing for manual and auto-generated YouTube captions.

Auto captions are rolling: each cue often contains all or part of the previous
cue.  The parser removes tags, sound-only cues, exact duplicates, and merges
rolling repetitions into one readable text stream.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, replace

from .text import normalize_transcript

_TIMESTAMP = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?")
_VOICE_TAG = re.compile(r"<v(?:\.[^ >]+)*\s+([^>]+)>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]*>")
_BRACKETED = re.compile(r"[\[(]([^\])]*)[\])]")
_SOUND_WORDS = {
    "applause",
    "background music",
    "cheering",
    "chuckles",
    "laughter",
    "laughing",
    "laughs",
    "music",
    "noise",
    "silence",
    "sighs",
    "upbeat music",
    "аплодисменты",
    "вздох",
    "закадровый смех",
    "музыка",
    "смех",
    "смеются",
    "тишина",
    "шум",
}
_TIMING_OVERLAP_TOLERANCE = 0.1


@dataclass(frozen=True)
class VTTCue:
    """One cleaned subtitle cue."""

    start: float
    end: float | None
    text: str


def parse_vtt(content: str, *, merge: bool = True) -> list[VTTCue]:
    """Parse WebVTT text into cleaned cues.

    ``merge`` also removes rolling duplicates and joins overlapping cues.
    """

    if not content or not content.strip():
        return []

    normalized = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    blocks = re.split(r"\n\s*\n", normalized)
    cues: list[VTTCue] = []

    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        first = lines[0].upper()
        if first.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue

        timing_index = next((index for index, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        cue = _parse_cue(lines[timing_index], lines[timing_index + 1 :])
        if cue is not None:
            cues.append(cue)

    return _merge_rolling_cues(cues) if merge else cues


def vtt_to_text(content: str) -> str:
    """Parse VTT and return one normalized, space-joined text line."""

    return cues_to_text(parse_vtt(content))


def cues_to_text(cues: list[VTTCue]) -> str:
    """Join cleaned cue texts into a single normalized transcript."""

    parts: list[str] = []
    previous_speaker: str | None = None
    for cue in cues:
        if not cue.text:
            continue
        speaker, body = _split_speaker(cue.text)
        if speaker is not None and speaker == previous_speaker:
            body = body
        elif speaker is not None:
            body = f"{speaker}: {body}"
        previous_speaker = speaker
        if body:
            parts.append(body)
    return normalize_transcript(" ".join(parts))


def _parse_cue(timing_line: str, text_lines: list[str]) -> VTTCue | None:
    if "-->" not in timing_line:
        return None
    start_raw, end_raw = timing_line.split("-->", 1)
    start = _parse_timestamp(start_raw.strip().split()[0] if start_raw.strip() else "")
    end_token = end_raw.strip().split()[0] if end_raw.strip() else ""
    end = _parse_timestamp(end_token)
    if start is None or end is None:
        return None

    text = _clean_cue_text(" ".join(text_lines))
    if not text or _is_sound_only(text):
        return None
    return VTTCue(start=start, end=end, text=text)


def _parse_timestamp(value: str) -> float | None:
    match = _TIMESTAMP.fullmatch(value.strip())
    if match is None:
        return None
    hours_raw, minutes_raw, seconds_raw, milliseconds_raw = match.groups()
    hours = int(hours_raw) if hours_raw else 0
    minutes = int(minutes_raw)
    seconds = int(seconds_raw)
    milliseconds = int((milliseconds_raw or "0").ljust(3, "0"))
    if minutes >= 60 or seconds >= 60:
        return None
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def _clean_cue_text(raw: str) -> str:
    text = _VOICE_TAG.sub(lambda match: f"{match.group(1).strip()}: ", raw)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = _TAG.sub("", text)
    text = html.unescape(text)
    return normalize_transcript(text)


def _split_speaker(text: str) -> tuple[str | None, str]:
    match = re.match(r"^([A-Za-zА-Яа-яЁё][^:]{0,59}):\s+(.+)$", text)
    if match is None:
        return None, text
    return match.group(1).strip(), match.group(2).strip()


def _is_sound_only(text: str) -> bool:
    groups = [match.group(1).strip() for match in _BRACKETED.finditer(text)]
    residual = _BRACKETED.sub("", text)
    residual = residual.replace("♪", "").replace("♫", "")
    residual = re.sub(r"[\s\W_]+", "", residual, flags=re.UNICODE)
    if residual:
        return False
    if not groups:
        return True
    return any(_contains_sound_word(group) for group in groups)


def _contains_sound_word(value: str) -> bool:
    folded = value.casefold()
    return any(word in folded for word in _SOUND_WORDS)


def _merge_rolling_cues(cues: list[VTTCue]) -> list[VTTCue]:
    merged: list[VTTCue] = []
    for cue in cues:
        if not cue.text:
            continue
        if not merged:
            merged.append(cue)
            continue

        previous = merged[-1]
        if cue.text == previous.text:
            merged[-1] = replace(previous, end=_max_end(previous.end, cue.end))
            continue

        previous_speaker, previous_body = _split_speaker(previous.text)
        current_speaker, current_body = _split_speaker(cue.text)
        same_speaker = previous_speaker == current_speaker

        if same_speaker:
            if _extends(previous_body, current_body):
                merged[-1] = replace(
                    previous,
                    end=_max_end(previous.end, cue.end),
                    text=_with_speaker(current_speaker, current_body),
                )
                continue
            if _extends(current_body, previous_body):
                merged[-1] = replace(previous, end=_max_end(previous.end, cue.end))
                continue

            allow_single = _cues_overlap_in_time(previous.end, cue.start)
            overlap = _word_overlap(
                previous_body,
                current_body,
                allow_single=allow_single,
            )
            if overlap >= (1 if allow_single else 2):
                current_words = current_body.split()
                combined_body = " ".join([previous_body, *current_words[overlap:]])
                merged[-1] = replace(
                    previous,
                    end=_max_end(previous.end, cue.end),
                    text=_with_speaker(current_speaker, combined_body),
                )
                continue

        merged.append(cue)

    return merged


def _extends(shorter: str, longer: str) -> bool:
    if len(longer) <= len(shorter) or not shorter:
        return False
    if not longer.startswith(shorter):
        return False
    boundary = longer[len(shorter)] if len(longer) > len(shorter) else ""
    return not boundary.isalnum()


def _word_overlap(previous: str, current: str, *, allow_single: bool = False) -> int:
    previous_words = previous.split()
    current_words = current.split()
    maximum = min(len(previous_words), len(current_words))
    minimum = 1 if allow_single else 2
    for size in range(maximum, minimum - 1, -1):
        if previous_words[-size:] == current_words[:size]:
            return size
    return 0


def _cues_overlap_in_time(previous_end: float | None, current_start: float) -> bool:
    if previous_end is None:
        return False
    return current_start <= previous_end + _TIMING_OVERLAP_TOLERANCE


def _with_speaker(speaker: str | None, body: str) -> str:
    return f"{speaker}: {body}" if speaker else body


def _max_end(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)
