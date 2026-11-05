"""Editable prompt templates and prompt builders.

Prompt text is English on purpose: it gives the model unambiguous editing rules
regardless of the transcript language.  Output language is controlled by the
content and by the explicit annotation language.
"""

from __future__ import annotations

from dataclasses import dataclass

PROMPT_VERSION = "2026-11-02.1"

CLEANING_SYSTEM_PROMPT = """\
You are a careful copy editor for spoken YouTube transcripts. Your job is to \
produce a readable written version of what the speaker actually said.

Rules:
- Preserve the speaker's meaning, facts, examples, numbers, names, and voice.
- Add punctuation, capitalization, and paragraph breaks.
- Remove filler words, false starts, and repeated words only when they add no meaning.
- Fix only obvious speech-recognition errors. If a word is genuinely unclear, keep it and write [unclear] instead of guessing.
- Do not summarize, shorten, expand, add facts, censor, or add headings, lists, or commentary.
- Do not translate. Return the edited text in the same language as the input.
- Return only the edited transcript text. Do not wrap it in quotes or code fences and do not preface it with a sentence.
"""

CLEANING_USER_TEMPLATE = """\
Channel: {channel}
Playlist: {playlist}
Video title: {video_title}

Edit the transcript chunk below according to the system rules.

<transcript>
{chunk}
</transcript>
"""

CLEANING_RETRY_INSTRUCTION = """\
Your previous response was empty or contained commentary. Return only the edited \
transcript text, in the same language as the input, with no preamble, quotes, \
headings, or code fences.
"""

ANNOTATION_SYSTEM_PROMPT = """\
You write concise, factual book annotations for chapters made from YouTube \
transcripts. An annotation helps a reader decide whether to read the chapter. \
Use only information present in the supplied cleaned transcript. Do not invent \
facts, quote long passages, or add commentary about the transcript itself.
"""

ANNOTATION_USER_TEMPLATE = """\
Write a 2-4 sentence annotation for this chapter in {language}.
The annotation must be plain prose: no heading, no bullets, no quotation marks, \
no preamble. Base it only on the cleaned chapter text below.

Channel: {channel}
Playlist: {playlist}
Video title: {video_title}

<chapter>
{text}
</chapter>
"""

ANNOTATION_RETRY_INSTRUCTION = """\
Your previous annotation was too short. Write a new annotation of this chapter \
using 2 to 4 complete sentences. Return only the annotation, with no heading, \
bullets, quotation marks, or preamble.
"""


@dataclass(frozen=True)
class ChapterContext:
    """Context supplied to cleaning and annotation prompts."""

    channel: str
    playlist: str
    video_title: str


def build_cleaning_messages(context: ChapterContext, chunk: str) -> list[dict[str, str]]:
    """Build chat messages for one cleaning request."""

    user = CLEANING_USER_TEMPLATE.format(
        channel=context.channel or "Unknown channel",
        playlist=context.playlist or "Uploads",
        video_title=context.video_title or "Untitled video",
        chunk=chunk,
    )
    return [
        {"role": "system", "content": CLEANING_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_cleaning_retry_messages(context: ChapterContext, chunk: str) -> list[dict[str, str]]:
    """Build a stricter retry request after an empty/commentary response."""

    messages = build_cleaning_messages(context, chunk)
    messages.append({"role": "user", "content": CLEANING_RETRY_INSTRUCTION})
    return messages


def build_annotation_messages(
    context: ChapterContext,
    language: str,
    text: str,
) -> list[dict[str, str]]:
    """Build chat messages for a 2-4 sentence chapter annotation."""

    user = ANNOTATION_USER_TEMPLATE.format(
        language=language or "English",
        channel=context.channel or "Unknown channel",
        playlist=context.playlist or "Uploads",
        video_title=context.video_title or "Untitled video",
        text=text,
    )
    return [
        {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_annotation_retry_messages(
    context: ChapterContext,
    language: str,
    text: str,
) -> list[dict[str, str]]:
    """Build a stricter request when an annotation was shorter than two sentences."""

    messages = build_annotation_messages(context, language, text)
    messages.append({"role": "user", "content": ANNOTATION_RETRY_INSTRUCTION})
    return messages
