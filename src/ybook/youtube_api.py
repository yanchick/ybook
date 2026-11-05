"""YouTube Data API v3 client with channel resolution and pagination.

Only ``httpx`` is used so tests can inject an ``httpx.MockTransport``.  API
keys never appear in errors or logs.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import unquote, urlparse

import httpx
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .errors import ChannelInputError, ChannelNotFoundError, QuotaExceededError, YouTubeError
from .utils import redact_secret

LOGGER = logging.getLogger(__name__)

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
_CHANNEL_ID_PATTERN = re.compile(r"^UC[A-Za-z0-9_-]{2,}$")
_ISO_DURATION = re.compile(
    r"^P(?:(\d+)W)?(?:(\d+)D)?" r"(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:[.,]\d+)?)S)?)?$"
)

ReferenceKind = Literal["handle", "id", "custom", "username"]


@dataclass(frozen=True)
class ChannelReference:
    """Parsed channel input before it is resolved by the API."""

    kind: ReferenceKind
    value: str


@dataclass(frozen=True)
class Channel:
    """Resolved channel metadata."""

    id: str
    title: str
    description: str
    url: str
    uploads_playlist_id: str | None


@dataclass(frozen=True)
class Playlist:
    """Channel playlist metadata."""

    id: str
    title: str
    description: str
    item_count: int


@dataclass(frozen=True)
class PlaylistItem:
    """One video membership in a playlist."""

    video_id: str
    position: int


@dataclass(frozen=True)
class Video:
    """Video metadata needed by the book pipeline."""

    id: str
    title: str
    description: str
    published_at: str
    duration_seconds: int
    default_audio_language: str | None
    live_status: str


class _RetryableYouTubeError(Exception):
    """Internal marker for 429/5xx responses that tenacity should retry."""


class YouTubeClient:
    """Small, typed wrapper around the YouTube Data API list endpoints."""

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        base_url: str = YOUTUBE_API_BASE,
    ):
        key = api_key.strip()
        if not key:
            raise YouTubeError("YouTube API key is empty")
        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=httpx.Timeout(30.0), follow_redirects=True)
        self._retrying = Retrying(
            retry=retry_if_exception_type((_RetryableYouTubeError, httpx.TransportError)),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
            stop=stop_after_attempt(4),
            reraise=True,
        )

    def __enter__(self) -> YouTubeClient:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client if this instance created it."""

        if self._owns_client:
            self._client.close()

    def resolve_channel(self, value: str | ChannelReference) -> Channel:
        """Resolve a handle, channel ID, custom name, or username to a channel."""

        reference = parse_channel_input(value) if isinstance(value, str) else value

        if reference.kind == "id":
            channel = self._first_channel({"id": reference.value})
        elif reference.kind == "handle":
            channel = self._first_channel({"forHandle": reference.value})
        elif reference.kind == "username":
            channel = self._first_channel({"forUsername": reference.value})
        else:
            channel = self._resolve_custom_name(reference.value)

        if channel is None:
            raise ChannelNotFoundError(
                f"Channel not found for {reference.kind} '{reference.value}'. "
                "Check the URL and make sure the channel is public."
            )
        return channel

    def list_playlists(self, channel_id: str) -> list[Playlist]:
        """List all public playlists of a channel, following page tokens."""

        playlists: list[Playlist] = []
        for page in self._paginate(
            "playlists",
            {
                "channelId": channel_id,
                "part": "snippet,contentDetails",
                "maxResults": 50,
            },
        ):
            for item in page.get("items", []):
                playlist_id = item.get("id")
                if not playlist_id:
                    continue
                snippet = item.get("snippet") or {}
                content_details = item.get("contentDetails") or {}
                playlists.append(
                    Playlist(
                        id=str(playlist_id),
                        title=str(snippet.get("title") or "Untitled playlist"),
                        description=str(snippet.get("description") or ""),
                        item_count=_as_int(content_details.get("itemCount"), default=0),
                    )
                )
        return playlists

    def list_playlist_items(self, playlist_id: str) -> list[PlaylistItem]:
        """List playable items in a playlist, skipping deleted/private entries."""

        items: list[PlaylistItem] = []
        ordinal = 0
        for page in self._paginate(
            "playlistItems",
            {
                "playlistId": playlist_id,
                "part": "snippet,contentDetails",
                "maxResults": 50,
            },
        ):
            page_items = page.get("items", [])
            for index, item in enumerate(page_items):
                snippet = item.get("snippet") or {}
                content_details = item.get("contentDetails") or {}
                title = str(snippet.get("title") or "")
                if title.casefold() in {"deleted video", "private video"}:
                    continue
                video_id = content_details.get("videoId")
                if not video_id:
                    video_id = (snippet.get("resourceId") or {}).get("videoId")
                if not video_id:
                    continue
                position = snippet.get("position")
                if not isinstance(position, int):
                    position = ordinal + index
                items.append(PlaylistItem(video_id=str(video_id), position=position))
            ordinal += len(page_items)
        return items

    def fetch_videos(self, video_ids: list[str]) -> list[Video]:
        """Fetch video metadata in API batches of up to 50 IDs."""

        unique_ids = list(dict.fromkeys(video_id for video_id in video_ids if video_id))
        videos: list[Video] = []
        for start in range(0, len(unique_ids), 50):
            batch = unique_ids[start : start + 50]
            data = self._get(
                "videos",
                {
                    "id": ",".join(batch),
                    "part": "snippet,contentDetails",
                    "maxResults": 50,
                },
            )
            for item in data.get("items", []):
                video = _video_from_item(item)
                if video is not None:
                    videos.append(video)
        return videos

    def _resolve_custom_name(self, name: str) -> Channel | None:
        channel = self._first_channel({"forHandle": name})
        if channel is not None:
            return channel
        channel = self._first_channel({"forUsername": name})
        if channel is not None:
            return channel

        LOGGER.warning(
            "Custom channel name '%s' not found via forHandle/forUsername; "
            "falling back to search.list (costs 100 quota units)",
            name,
        )
        search = self._get(
            "search",
            {
                "part": "snippet",
                "type": "channel",
                "q": name,
                "maxResults": 1,
            },
        )
        search_items = search.get("items", [])
        if not search_items:
            return None
        channel_id = (search_items[0].get("id") or {}).get("channelId")
        if not channel_id:
            return None
        return self._first_channel({"id": str(channel_id)})

    def _first_channel(self, identifier: dict[str, str]) -> Channel | None:
        data = self._get(
            "channels",
            {
                "part": "snippet,contentDetails",
                **identifier,
            },
        )
        items = data.get("items", [])
        if not items:
            return None
        return _channel_from_item(items[0])

    def _paginate(self, endpoint: str, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        page_token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            page_params = dict(params)
            if page_token:
                page_params["pageToken"] = page_token
            data = self._get(endpoint, page_params)
            yield data
            next_token = data.get("nextPageToken")
            if not next_token:
                return
            token = str(next_token)
            if token in seen_tokens:
                raise YouTubeError("YouTube API returned a repeated page token")
            seen_tokens.add(token)
            page_token = token

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        request_params = dict(params)
        request_params["key"] = self._api_key
        try:
            response = self._retrying(self._send, endpoint, request_params)
        except _RetryableYouTubeError as exc:
            raise YouTubeError(f"YouTube API request failed after retries: {exc}") from exc
        except httpx.TransportError as exc:
            message = redact_secret(str(exc), self._api_key)
            raise YouTubeError(f"YouTube API network error: {message}") from exc
        return self._parse_response(response)

    def _send(self, endpoint: str, params: dict[str, Any]) -> httpx.Response:
        response = self._client.get(f"{self._base_url}/{endpoint}", params=params)
        rate_limit_reasons = _error_reasons(response) if response.status_code == 403 else set()
        if response.status_code == 403 and rate_limit_reasons & {
            "rateLimitExceeded",
            "userRateLimitExceeded",
        }:
            raise _RetryableYouTubeError(
                f"HTTP 403 rate limit from {endpoint}: "
                f"{_response_message(response, self._api_key)}"
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise _RetryableYouTubeError(
                f"HTTP {response.status_code} from {endpoint}: "
                f"{_response_message(response, self._api_key)}"
            )
        return response

    def _parse_response(self, response: httpx.Response) -> dict[str, Any]:
        if response.is_success:
            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                raise YouTubeError("YouTube API returned invalid JSON") from exc
            if not isinstance(payload, dict):
                raise YouTubeError("YouTube API returned an unexpected JSON shape")
            return payload

        message = _response_message(response, self._api_key)
        reasons = _error_reasons(response)
        if response.status_code == 403 and (
            "quotaExceeded" in reasons or "dailyLimitExceeded" in reasons
        ):
            raise QuotaExceededError(
                "YouTube Data API quota exceeded. Wait for the daily reset, use a "
                "different key, or request a quota increase in Google Cloud."
            )
        if response.status_code in {400, 401, 403} and "keyInvalid" in reasons:
            raise YouTubeError(f"YouTube API key is invalid or restricted: {message}")
        raise YouTubeError(f"YouTube API returned HTTP {response.status_code}: {message}")


def parse_channel_input(value: str) -> ChannelReference:
    """Parse supported YouTube channel URLs, ``@handle`` values, and ``UC...`` IDs.

    Playlist, watch, and individual video URLs are rejected with actionable
    messages because a channel is required for enumeration.
    """

    text = value.strip()
    if not text:
        raise ChannelInputError("Channel URL or handle is empty")

    if text.startswith("@"):
        handle = text[1:].strip()
        if not handle or "/" in handle or "?" in handle or " " in handle:
            raise ChannelInputError(
                "Bare handles must look like @handle (channel URLs may include /videos)."
            )
        return ChannelReference(kind="handle", value=handle)

    if _CHANNEL_ID_PATTERN.fullmatch(text):
        return ChannelReference(kind="id", value=text)

    lowered = text.lower()
    if lowered.startswith(("youtu.be/", "www.youtu.be/")):
        raise ChannelInputError(
            "This looks like a video URL. Provide a channel URL, @handle, or UC... ID."
        )
    if "://" not in text and lowered.startswith(
        ("youtube.com/", "www.youtube.com/", "m.youtube.com/")
    ):
        text = f"https://{text}"

    parsed = urlparse(text)
    if not parsed.scheme:
        raise ChannelInputError(
            "Unsupported channel reference. Use a channel URL, @handle, or UC... ID."
        )
    if parsed.scheme not in {"http", "https"}:
        raise ChannelInputError("Only http:// and https:// YouTube URLs are supported")

    host = (parsed.hostname or "").lower()
    if host in {"youtu.be", "www.youtu.be"}:
        raise ChannelInputError(
            "This looks like a video URL. Provide a channel URL, @handle, or UC... ID."
        )
    if host != "youtube.com" and not host.endswith(".youtube.com"):
        raise ChannelInputError(f"Not a YouTube URL: {host or value}")

    segments = [unquote(segment) for segment in parsed.path.split("/") if segment]
    if not segments:
        raise ChannelInputError(
            "This YouTube URL does not identify a channel. Add /@handle, /channel/UC..., "
            "/c/..., or /user/... to the URL."
        )

    first = segments[0]
    if first in {"playlist", "watch", "embed", "live"}:
        raise ChannelInputError(
            f"YouTube {first} URLs are not supported. Provide a channel URL, "
            "@handle, or UC... ID."
        )
    if first == "shorts" and len(segments) > 1:
        raise ChannelInputError(
            "Short video URLs are not supported. Provide a channel URL, @handle, or UC... ID."
        )
    if first.startswith("@"):
        handle = first[1:].strip()
        if not handle:
            raise ChannelInputError("Channel handle is empty")
        return ChannelReference(kind="handle", value=handle)
    if first == "channel" and len(segments) > 1:
        channel_id = segments[1]
        if not _CHANNEL_ID_PATTERN.fullmatch(channel_id):
            raise ChannelInputError(f"Invalid YouTube channel ID: {channel_id}")
        return ChannelReference(kind="id", value=channel_id)
    if first == "c" and len(segments) > 1:
        return ChannelReference(kind="custom", value=segments[1])
    if first == "user" and len(segments) > 1:
        return ChannelReference(kind="username", value=segments[1])

    raise ChannelInputError(
        "Unsupported YouTube channel URL. Use /@handle, /channel/UC..., /c/..., or /user/...."
    )


def parse_iso8601_duration(value: str) -> int:
    """Convert an ISO-8601 duration such as ``PT1H2M3S`` to seconds."""

    if not value:
        raise ValueError("Duration is empty")
    match = _ISO_DURATION.fullmatch(value.strip())
    if match is None or not any(group is not None for group in match.groups()):
        raise ValueError(f"Invalid ISO-8601 duration: {value}")

    weeks, days, hours, minutes, seconds = match.groups()
    total = (
        int(weeks or 0) * 604800
        + int(days or 0) * 86400
        + int(hours or 0) * 3600
        + int(minutes or 0) * 60
        + float((seconds or "0").replace(",", "."))
    )
    return int(round(total))


def _channel_from_item(item: dict[str, Any]) -> Channel:
    snippet = item.get("snippet") or {}
    content_details = item.get("contentDetails") or {}
    channel_id = str(item.get("id") or "")
    uploads = (content_details.get("relatedPlaylists") or {}).get("uploads")
    return Channel(
        id=channel_id,
        title=str(snippet.get("title") or channel_id or "YouTube channel"),
        description=str(snippet.get("description") or ""),
        url=f"https://www.youtube.com/channel/{channel_id}" if channel_id else "",
        uploads_playlist_id=str(uploads) if uploads else None,
    )


def _video_from_item(item: dict[str, Any]) -> Video | None:
    video_id = item.get("id")
    if not video_id:
        return None
    snippet = item.get("snippet") or {}
    content_details = item.get("contentDetails") or {}
    duration_raw = str(content_details.get("duration") or "")
    try:
        duration_seconds = parse_iso8601_duration(duration_raw)
    except ValueError:
        LOGGER.warning("Could not parse duration %r for video %s", duration_raw, video_id)
        duration_seconds = 0

    return Video(
        id=str(video_id),
        title=str(snippet.get("title") or "Untitled video"),
        description=str(snippet.get("description") or ""),
        published_at=str(snippet.get("publishedAt") or ""),
        duration_seconds=duration_seconds,
        default_audio_language=_optional_string(snippet.get("defaultAudioLanguage")),
        live_status=str(snippet.get("liveBroadcastContent") or "none"),
    )


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
            elif error is not None:
                body = str(error)
            else:
                body = str(payload)
        else:
            body = str(payload)
    return redact_secret(body.strip()[:800], api_key)


def _error_reasons(response: httpx.Response) -> set[str]:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return set()
    if not isinstance(payload, dict):
        return set()
    error = payload.get("error")
    if not isinstance(error, dict):
        return set()
    errors = error.get("errors") or []
    reasons: set[str] = set()
    for item in errors:
        reason = item.get("reason") if isinstance(item, dict) else None
        if reason:
            reasons.add(str(reason))
    return reasons


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
