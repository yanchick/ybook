"""OpenAI-compatible chat client, response cache, and transcript cleaning.

The HTTP layer intentionally speaks the plain ``/chat/completions`` protocol
with ``httpx``.  No provider SDK is used.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import LLMConfig
from .errors import LLMCleanError, LLMConfigurationError, LLMRequestError, LLMResponseError
from .prompts import (
    PROMPT_VERSION,
    ChapterContext,
    build_annotation_messages,
    build_annotation_retry_messages,
    build_cleaning_messages,
    build_cleaning_retry_messages,
)
from .text import (
    chunk_text,
    condense_for_prompt,
    extract_numeric_tokens,
    looks_truncated,
    normalize_cleaned_text,
    word_count,
)
from .utils import atomic_write_json, atomic_write_text, redact_secret, sha256_text

LOGGER = logging.getLogger(__name__)

_PREAMBLE_PATTERNS = (
    re.compile(
        r"^\s*(?:sure|certainly|of course)[,!.:\s]+"
        r"here(?:'s| is)\s+(?:the|your|a)\s+[^\n:]{0,80}:\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*here(?:'s| is)\s+(?:the|your|a)\s+[^\n:]{0,80}:\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*конечно[,.!:\s]+вот\s+"
        r"(?:текст|исправленн\w*|очищенн\w*|отредактированн\w*|расшифровк\w*)"
        r"(?:\s+текст)?\s*(?::\s*|(?=\n|$))",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*вот\s+(?:текст|исправленный|очищенный)[^\n:]{0,80}:\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:as an ai|i am an ai|i'm an ai)[^\n]*(?:\n|$)",
        re.IGNORECASE,
    ),
)
_META_PATTERN = re.compile(
    r"(?:\bas an ai\b|\bi(?:'m| am) an ai\b|"
    r"^\s*(?:sure|certainly|of course)[,!.:\s]+"
    r"here(?:'s| is)\s+(?:the|your)\s+(?:edited|cleaned|transcript|text)\b|"
    r"^\s*here(?:'s| is)\s+(?:the|your)\s+(?:edited|cleaned|transcript|text)\b|"
    r"^\s*конечно[,.!:\s]+вот\s+"
    r"(?:текст|исправленн\w*|очищенн\w*|отредактированн\w*|расшифровк\w*)\b|"
    r"^\s*вот\s+текст\b)",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class LLMResponse:
    """A parsed chat-completions response with usage and computed cost."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class CachedCompletion:
    """Content and usage restored from the on-disk cache."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class CleaningResult:
    """Result of cleaning one transcript."""

    text: str
    status: Literal["cleaned", "needs_review"]
    warnings: tuple[str, ...]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cache_hits: int


@dataclass(frozen=True)
class AnnotationResult:
    """A clean annotation plus its token usage."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class _RetryableLLMError(Exception):
    """Internal marker for retryable HTTP statuses."""


class LLMClient:
    """Minimal OpenAI-compatible chat-completions client."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        api_key: str,
        client: httpx.Client | None = None,
    ):
        key = api_key.strip()
        if not key:
            raise LLMConfigurationError("LLM API key is empty")
        self.config = config
        self._api_key = key
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(config.timeout_seconds), follow_redirects=True
        )
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update({str(name): str(value) for name, value in config.extra_headers.items()})
        self._headers = headers
        self._sleep = time.sleep

    def _retrier(self) -> Retrying:
        """Create a per-request retry object; ``Retrying`` is not thread-safe."""

        return Retrying(
            retry=retry_if_exception_type((_RetryableLLMError, httpx.TransportError)),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
            stop=stop_after_attempt(4),
            sleep=self._sleep,
            reraise=True,
        )

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client if this instance created it."""

        if self._owns_client:
            self._client.close()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """POST a chat-completions request and parse content and usage."""

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            self.config.max_tokens_param: (
                self.config.max_tokens if max_tokens is None else max_tokens
            ),
        }

        try:
            response = self._retrier()(self._send, payload)
        except _RetryableLLMError as exc:
            raise LLMRequestError(f"LLM endpoint failed after retries: {exc}") from exc
        except httpx.TransportError as exc:
            raise LLMRequestError(f"LLM endpoint network error: {exc}") from exc

        return self._parse_response(response)

    def _send(self, payload: dict[str, Any]) -> httpx.Response:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        response = self._client.post(url, headers=self._headers, json=payload)
        if response.status_code == 429 or response.status_code >= 500:
            raise _RetryableLLMError(
                f"HTTP {response.status_code}: {_response_message(response, self._api_key)}"
            )
        return response

    def _parse_response(self, response: httpx.Response) -> LLMResponse:
        if not response.is_success:
            raise LLMRequestError(
                f"LLM endpoint returned HTTP {response.status_code}: "
                f"{_response_message(response, self._api_key)}"
            )

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseError("LLM endpoint returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise LLMResponseError("LLM endpoint returned an unexpected JSON shape")

        choices = payload.get("choices")
        if not choices or not isinstance(choices, list):
            raise LLMResponseError("LLM response has no choices")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise LLMResponseError("LLM response has an invalid choice")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise LLMResponseError("LLM response has no message object")

        content = _content_to_text(message.get("content"))
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        prompt_tokens = _as_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
        completion_tokens = _as_int(usage.get("completion_tokens") or usage.get("output_tokens"))
        cost = (
            prompt_tokens / 1_000_000 * self.config.input_cost_per_million
            + completion_tokens / 1_000_000 * self.config.output_cost_per_million
        )
        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )


class LLMCache:
    """Per-chunk cache keyed by prompt, provider, model, and request parameters."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def key(self, *, config: LLMConfig, context: ChapterContext, chunk: str) -> str:
        """Return a stable SHA-256 cache key."""

        serialized = json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "model": config.model,
                "base_url": config.base_url,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "max_tokens_param": config.max_tokens_param,
                "context": asdict(context),
                "chunk": chunk,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return sha256_text(serialized)

    def get(self, key: str) -> CachedCompletion | None:
        """Return a cached completion or ``None`` when absent/unreadable."""

        content_path = self.root / f"{key}.txt"
        if not content_path.is_file():
            return None
        try:
            content = content_path.read_text(encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("Cannot read LLM cache entry %s: %s", content_path, exc)
            return None
        if not content.strip():
            return None

        usage = _read_usage(self.root / f"{key}.json")
        return CachedCompletion(
            content=content,
            prompt_tokens=_as_int(usage.get("prompt_tokens")),
            completion_tokens=_as_int(usage.get("completion_tokens")),
            cost_usd=_as_float(usage.get("cost_usd")),
        )

    def put(self, key: str, content: str, response: LLMResponse) -> None:
        """Atomically store completion text and usage metadata."""

        if not content.strip():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.root / f"{key}.txt", content)
        atomic_write_json(
            self.root / f"{key}.json",
            {
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "cost_usd": response.cost_usd,
            },
        )


def clean_text(
    client: LLMClient,
    cache: LLMCache | None,
    *,
    context: ChapterContext,
    text: str,
    force: bool = False,
) -> CleaningResult:
    """Clean a transcript in chunks and apply the chapter-level quality guard."""

    if not text or not text.strip():
        raise LLMCleanError("Transcript is empty")

    chunks = chunk_text(text, max_words=client.config.max_words_per_chunk)
    if not chunks:
        raise LLMCleanError("Transcript contains no text to clean")

    cleaned_chunks: list[str] = []
    warnings: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    cost_usd = 0.0
    cache_hits = 0

    for index, chunk in enumerate(chunks, start=1):
        cache_key = (
            cache.key(config=client.config, context=context, chunk=chunk)
            if cache is not None
            else ""
        )
        cached = cache.get(cache_key) if cache is not None and not force else None
        if cached is not None:
            cleaned_chunks.append(cached.content)
            prompt_tokens += cached.prompt_tokens
            completion_tokens += cached.completion_tokens
            cost_usd += cached.cost_usd
            cache_hits += 1
            continue

        content, response, chunk_warning = _clean_chunk(client, context, chunk)
        if chunk_warning:
            warnings.append(f"chunk {index}: {chunk_warning}")
        cleaned_chunks.append(content)
        prompt_tokens += response.prompt_tokens
        completion_tokens += response.completion_tokens
        cost_usd += response.cost_usd
        if cache is not None:
            cache.put(cache_key, content, response)

    combined = normalize_cleaned_text("\n\n".join(cleaned_chunks))
    if not combined:
        raise LLMCleanError("LLM returned empty text for every chunk")

    warnings.extend(_quality_warnings(input_text=text, output_text=combined))
    status: Literal["cleaned", "needs_review"] = "needs_review" if warnings else "cleaned"
    return CleaningResult(
        text=combined,
        status=status,
        warnings=tuple(warnings),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        cache_hits=cache_hits,
    )


def generate_annotation(
    client: LLMClient,
    *,
    context: ChapterContext,
    language: str,
    text: str,
) -> AnnotationResult:
    """Ask the LLM for a 2-4 sentence annotation of a cleaned chapter."""

    if not text or not text.strip():
        raise LLMCleanError("Cannot annotate an empty chapter")
    annotation_input = condense_for_prompt(text, client.config.annotation_max_words)
    response = client.chat(build_annotation_messages(context, language, annotation_input))
    annotation = sanitize_annotation(response.content)
    prompt_tokens = response.prompt_tokens
    completion_tokens = response.completion_tokens
    cost_usd = response.cost_usd

    if _annotation_sentence_count(annotation) < 2:
        retry = client.chat(build_annotation_retry_messages(context, language, annotation_input))
        prompt_tokens += retry.prompt_tokens
        completion_tokens += retry.completion_tokens
        cost_usd += retry.cost_usd
        retried = sanitize_annotation(retry.content)
        if retried:
            annotation = retried

    if not annotation:
        raise LLMResponseError("LLM returned an empty annotation")
    annotation = _limit_annotation_sentences(annotation, maximum=4)
    return AnnotationResult(
        text=annotation,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
    )


def sanitize_cleaned_text(text: str) -> str:
    """Remove code fences, heading markers, and known LLM preambles."""

    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = "\n".join(re.sub(r"^\s{0,3}#{1,6}\s*", "", line) for line in cleaned.split("\n"))

    for _ in range(4):
        before = cleaned
        for pattern in _PREAMBLE_PATTERNS:
            cleaned = pattern.sub("", cleaned, count=1).lstrip()
        if cleaned == before:
            break
    return normalize_cleaned_text(cleaned)


def sanitize_annotation(text: str) -> str:
    """Remove headings, bullets, quotes, and preambles from an annotation."""

    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = re.sub(r"```[^\n]*\n?", "", cleaned).replace("```", "")
    cleaned = re.sub(
        r"^\s*(?:(?:sure|certainly|of course)[,!.:\s]+)?"
        r"here(?:'s| is)\s+(?:the|your|a|an)\s+[^\n:]{0,80}?"
        r"annotation[^\n:]{0,20}:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*(?:(?:конечно|итак)[,.!:\s]+)?"
        r"вот\s+(?:(?:краткая|готовая|ваша)\s+)?"
        r"аннотация[^\n:]{0,40}:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    lines: list[str] = []
    for line in cleaned.split("\n"):
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line)
        line = re.sub(r"^(?:annotation|аннотация)\s*:\s*", "", line, flags=re.IGNORECASE)
        line = line.strip()
        if line:
            lines.append(line)
    cleaned = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if len(cleaned) >= 2 and cleaned[0] in "\"'«“" and cleaned[-1] in "\"'»”":
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _annotation_sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?…])\s+", text.strip())
        if sentence.strip()
    ]


def _annotation_sentence_count(text: str) -> int:
    return len(_annotation_sentences(text))


def _limit_annotation_sentences(text: str, *, maximum: int) -> str:
    sentences = _annotation_sentences(text)
    if len(sentences) <= maximum:
        return text.strip()
    return " ".join(sentences[:maximum])


def _clean_chunk(
    client: LLMClient,
    context: ChapterContext,
    chunk: str,
) -> tuple[str, LLMResponse, str | None]:
    first = client.chat(build_cleaning_messages(context, chunk))
    content = sanitize_cleaned_text(first.content)
    prompt_tokens = first.prompt_tokens
    completion_tokens = first.completion_tokens
    cost_usd = first.cost_usd

    if content and not _contains_meta(content):
        return content, first, None

    retry = client.chat(build_cleaning_retry_messages(context, chunk))
    prompt_tokens += retry.prompt_tokens
    completion_tokens += retry.completion_tokens
    cost_usd += retry.cost_usd
    content = sanitize_cleaned_text(retry.content)
    if not content:
        raise LLMCleanError("LLM returned empty output after retry")

    warning = "LLM output still contains meta commentary" if _contains_meta(content) else None
    combined = LLMResponse(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
    )
    return content, combined, warning


def _quality_warnings(*, input_text: str, output_text: str) -> list[str]:
    warnings: list[str] = []
    input_words = word_count(input_text)
    output_words = word_count(output_text)
    if output_words == 0:
        warnings.append("Cleaned text is empty")
        return warnings

    if input_words:
        ratio = output_words / input_words
        if ratio < 0.75 or ratio > 1.30:
            warnings.append(f"Word-count ratio {ratio:.2f} outside the allowed 0.75-1.30 range")

    missing_numbers = sorted(
        extract_numeric_tokens(input_text) - extract_numeric_tokens(output_text)
    )
    if missing_numbers:
        preview = ", ".join(missing_numbers[:10])
        warnings.append(f"Missing numeric tokens: {preview}")

    if _contains_meta(output_text):
        warnings.append("Output contains meta commentary")

    if looks_truncated(output_text):
        warnings.append("Output appears truncated (no terminal punctuation)")

    return warnings


def _contains_meta(text: str) -> bool:
    return bool(_META_PATTERN.search(text[:1000]))


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    raise LLMResponseError("LLM message content has an unsupported type")


def _response_message(response: httpx.Response, api_key: str) -> str:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        body = response.text
    else:
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                body = str(error.get("message") or error)
            else:
                body = str(error if error is not None else payload)
        else:
            body = str(payload)
    return redact_secret(body.strip()[:1200], api_key)


def _read_usage(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
