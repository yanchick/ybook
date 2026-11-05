"""OpenAI-compatible HTTP client, cost parsing, retries, and cache tests."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ybook.config import LLMConfig
from ybook.errors import LLMCleanError, LLMRequestError, LLMResponseError
from ybook.llm import (
    LLMCache,
    LLMClient,
    clean_text,
    generate_annotation,
    sanitize_annotation,
    sanitize_cleaned_text,
)
from ybook.prompts import ChapterContext

CONTEXT = ChapterContext(
    channel="Test Channel",
    playlist="Test Playlist",
    video_title="Test Video",
)


def _config(**overrides) -> LLMConfig:
    values = {
        "base_url": "https://llm.test/v1",
        "model": "test-model",
        "temperature": 0.3,
        "max_tokens": 123,
        "max_words_per_chunk": 2500,
        "input_cost_per_million": 2.0,
        "output_cost_per_million": 4.0,
    }
    values.update(overrides)
    return LLMConfig(**values)


def _completion(
    content="Hello, world. This is a test.",
    prompt_tokens=1000,
    completion_tokens=500,
) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def test_chat_payload_headers_usage_and_cost() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_completion())

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    config = _config(extra_headers={"X-Title": "ybook", "X-Num": "7"})
    client = LLMClient(config, api_key="secret-key", client=http_client)
    try:
        response = client.chat([{"role": "user", "content": "hello"}])
    finally:
        http_client.close()

    assert response.content == "Hello, world. This is a test."
    assert response.prompt_tokens == 1000
    assert response.completion_tokens == 500
    assert response.cost_usd == pytest.approx(0.004)
    assert len(requests) == 1
    request = requests[0]
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer secret-key"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["X-Title"] == "ybook"
    assert request.headers["X-Num"] == "7"
    payload = json.loads(request.content)
    assert payload["model"] == "test-model"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["temperature"] == 0.3
    assert payload["max_tokens"] == 123
    assert "max_completion_tokens" not in payload


def test_content_parts_and_max_completion_tokens_parameter() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "Part one."},
                                {"type": "text", "text": "Part two."},
                            ]
                        }
                    }
                ],
                "usage": {"input_tokens": 12, "output_tokens": 8},
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = LLMClient(
        _config(max_tokens_param="max_completion_tokens"),
        api_key="key",
        client=http_client,
    )
    try:
        response = client.chat([{"role": "user", "content": "hi"}])
    finally:
        http_client.close()

    assert response.content == "Part one.\nPart two."
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 8
    payload = json.loads(requests[0].content)
    assert payload["max_completion_tokens"] == 123
    assert "max_tokens" not in payload


def test_429_is_retried_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                json={"error": {"message": "rate limited"}},
            )
        return httpx.Response(200, json=_completion())

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = LLMClient(_config(), api_key="key", client=http_client)
    client._sleep = lambda _: None
    try:
        response = client.chat([{"role": "user", "content": "hi"}])
    finally:
        http_client.close()

    assert calls == 2
    assert response.content.startswith("Hello")


def test_permanent_error_includes_redacted_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": "Invalid API key secret-key for model test-model",
                }
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = LLMClient(_config(), api_key="secret-key", client=http_client)
    try:
        with pytest.raises(LLMRequestError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}])
    finally:
        http_client.close()

    message = str(exc_info.value)
    assert "401" in message
    assert "secret-key" not in message
    assert "[redacted]" in message
    assert "test-model" in message


def test_malformed_response_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = LLMClient(_config(), api_key="key", client=http_client)
    try:
        with pytest.raises(LLMResponseError):
            client.chat([{"role": "user", "content": "hi"}])
    finally:
        http_client.close()


def test_clean_text_uses_cache_and_force_bypasses_it(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_completion())

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = LLMClient(_config(), api_key="key", client=http_client)
    cache = LLMCache(tmp_path / "cache")
    text = "Hello world. This is a test."
    try:
        first = clean_text(client, cache, context=CONTEXT, text=text)
        second = clean_text(client, cache, context=CONTEXT, text=text)
        third = clean_text(client, cache, context=CONTEXT, text=text, force=True)
    finally:
        http_client.close()

    assert calls == 2
    assert first.cache_hits == 0
    assert second.cache_hits == 1
    assert third.cache_hits == 0
    assert first.status == "cleaned"
    assert second.text == first.text
    assert second.prompt_tokens == 1000
    assert len(list((tmp_path / "cache").glob("*.txt"))) == 1
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_cache_key_includes_provider_and_request_parameters(tmp_path: Path) -> None:
    cache = LLMCache(tmp_path / "cache")
    config = _config()
    baseline = cache.key(config=config, context=CONTEXT, chunk="chunk text")

    assert baseline == cache.key(config=config, context=CONTEXT, chunk="chunk text")
    for update in (
        {"base_url": "https://other.example/v1"},
        {"temperature": 0.9},
        {"max_tokens": 999},
        {"max_tokens_param": "max_completion_tokens"},
        {"model": "other-model"},
    ):
        changed = cache.key(
            config=config.model_copy(update=update),
            context=CONTEXT,
            chunk="chunk text",
        )
        assert changed != baseline, update


def test_quality_guard_flags_missing_numbers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_completion(
                content="There were apples in the box.",
                prompt_tokens=10,
                completion_tokens=5,
            ),
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = LLMClient(_config(), api_key="key", client=http_client)
    try:
        result = clean_text(
            client,
            None,
            context=CONTEXT,
            text="There were 42 apples in the box.",
        )
    finally:
        http_client.close()

    assert result.status == "needs_review"
    assert any("42" in warning for warning in result.warnings)


def test_quality_guard_accepts_date_rewritten_in_words() -> None:
    source = "The meeting was on 2024-01-05 and cost 42 dollars."
    cleaned = "The meeting was on January 5, 2024 and cost 42 dollars."

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_completion(
                content=cleaned,
                prompt_tokens=10,
                completion_tokens=5,
            ),
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = LLMClient(_config(), api_key="key", client=http_client)
    try:
        result = clean_text(client, None, context=CONTEXT, text=source)
    finally:
        http_client.close()

    assert result.status == "cleaned"
    assert not any("numeric" in warning.lower() for warning in result.warnings)


@pytest.mark.parametrize(
    "text",
    [
        "Sure, that is correct.",
        "Конечно, это важно.",
    ],
)
def test_legitimate_sure_and_of_course_phrases_are_not_meta(text: str) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json=_completion(
                content=text,
                prompt_tokens=10,
                completion_tokens=5,
            ),
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = LLMClient(_config(), api_key="key", client=http_client)
    try:
        result = clean_text(client, None, context=CONTEXT, text=text)
    finally:
        http_client.close()

    assert sanitize_cleaned_text(text) == text
    assert result.text == text
    assert result.status == "cleaned"
    assert calls == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Sure, here is the cleaned transcript:\nHello world.", "Hello world."),
        ("Конечно, вот текст:\nПривет мир.", "Привет мир."),
    ],
)
def test_explicit_meta_preambles_are_still_removed(raw: str, expected: str) -> None:
    assert sanitize_cleaned_text(raw) == expected


def test_empty_or_meta_response_is_retried_once() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                json=_completion(
                    content="As an AI, here is the text:\n",
                    prompt_tokens=10,
                    completion_tokens=5,
                ),
            )
        return httpx.Response(
            200,
            json=_completion(
                content="Hello, world.",
                prompt_tokens=11,
                completion_tokens=6,
            ),
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = LLMClient(_config(), api_key="key", client=http_client)
    try:
        result = clean_text(client, None, context=CONTEXT, text="Hello world.")
    finally:
        http_client.close()

    assert calls == 2
    assert result.status == "cleaned"
    assert result.text == "Hello, world."
    assert result.prompt_tokens == 21


def test_empty_after_retry_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(content=""))

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = LLMClient(_config(), api_key="key", client=http_client)
    try:
        with pytest.raises(LLMCleanError):
            clean_text(client, None, context=CONTEXT, text="Hello world.")
    finally:
        http_client.close()


def test_generate_annotation_returns_usage_and_sanitized_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_completion(
                content="This chapter explains the topic. It gives two useful examples.",
                prompt_tokens=200,
                completion_tokens=30,
            ),
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = LLMClient(_config(), api_key="key", client=http_client)
    try:
        annotation = generate_annotation(
            client,
            context=CONTEXT,
            language="en",
            text="Cleaned chapter text.",
        )
    finally:
        http_client.close()

    assert annotation.text == ("This chapter explains the topic. It gives two useful examples.")
    assert annotation.prompt_tokens == 200
    assert annotation.completion_tokens == 30
    assert annotation.cost_usd == pytest.approx(200 / 1_000_000 * 2 + 30 / 1_000_000 * 4)


def test_generate_annotation_condenses_long_chapters() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=_completion(
                content="A concise annotation. It has two sentences.",
                prompt_tokens=100,
                completion_tokens=12,
            ),
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = LLMClient(
        _config(annotation_max_words=1000),
        api_key="key",
        client=http_client,
    )
    long_text = " ".join(f"word{index}" for index in range(1500))
    try:
        generate_annotation(
            client,
            context=CONTEXT,
            language="en",
            text=long_text,
        )
    finally:
        http_client.close()

    payload = json.loads(requests[0].content)
    user_content = payload["messages"][1]["content"]
    assert "\n\n[...]\n\n" in user_content
    assert "word0" in user_content
    assert "word1499" in user_content
    assert "word750" not in user_content


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "Sure, here is a 2–4 sentence annotation:\nFirst sentence. Second sentence.",
            "First sentence. Second sentence.",
        ),
        (
            "Here is the annotation:\nFirst sentence. Second sentence.",
            "First sentence. Second sentence.",
        ),
        (
            "Вот аннотация:\nПервое предложение. Второе предложение.",
            "Первое предложение. Второе предложение.",
        ),
    ],
)
def test_annotation_preambles_are_stripped(raw: str, expected: str) -> None:
    assert sanitize_annotation(raw) == expected


def test_annotation_longer_than_four_sentences_is_trimmed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_completion(
                content="One. Two. Three. Four. Five.",
                prompt_tokens=10,
                completion_tokens=5,
            ),
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = LLMClient(_config(), api_key="key", client=http_client)
    try:
        annotation = generate_annotation(
            client,
            context=CONTEXT,
            language="en",
            text="Cleaned chapter text.",
        )
    finally:
        http_client.close()

    assert annotation.text == "One. Two. Three. Four."


def test_short_annotation_is_retried_once() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = "Too short." if calls == 1 else "First sentence. Second sentence."
        return httpx.Response(
            200,
            json=_completion(
                content=content,
                prompt_tokens=10,
                completion_tokens=5,
            ),
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = LLMClient(_config(), api_key="key", client=http_client)
    try:
        annotation = generate_annotation(
            client,
            context=CONTEXT,
            language="en",
            text="Cleaned chapter text.",
        )
    finally:
        http_client.close()

    assert calls == 2
    assert annotation.text == "First sentence. Second sentence."
    assert annotation.prompt_tokens == 20
    assert annotation.completion_tokens == 10


def test_still_short_annotation_is_accepted_after_retry() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = "Short." if calls == 1 else "Still short."
        return httpx.Response(
            200,
            json=_completion(
                content=content,
                prompt_tokens=10,
                completion_tokens=5,
            ),
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = LLMClient(_config(), api_key="key", client=http_client)
    try:
        annotation = generate_annotation(
            client,
            context=CONTEXT,
            language="en",
            text="Cleaned chapter text.",
        )
    finally:
        http_client.close()

    assert calls == 2
    assert annotation.text == "Still short."
