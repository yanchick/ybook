"""YouTube URL parsing, duration parsing, pagination, and uploads tests."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ybook.config import Config, PathsConfig, TranscriptsConfig
from ybook.db import Database
from ybook.errors import ChannelInputError, ConfigError, QuotaExceededError, YouTubeError
from ybook.pipeline import fetch_channel_data
from ybook.youtube_api import (
    Channel,
    ChannelReference,
    YouTubeClient,
    parse_channel_input,
    parse_iso8601_duration,
)

CHANNEL_ID = "UC" + "a" * 22


def _channel_item(
    channel_id: str = CHANNEL_ID,
    title: str = "Test Channel",
    uploads: str = "UU" + "a" * 22,
) -> dict:
    return {
        "id": channel_id,
        "snippet": {"title": title, "description": "Channel description"},
        "contentDetails": {"relatedPlaylists": {"uploads": uploads}},
    }


@pytest.mark.parametrize(
    ("value", "kind", "expected"),
    [
        ("https://www.youtube.com/@example", "handle", "example"),
        ("https://www.youtube.com/@example/videos", "handle", "example"),
        ("https://m.youtube.com/@example/shorts", "handle", "example"),
        (f"https://www.youtube.com/channel/{CHANNEL_ID}", "id", CHANNEL_ID),
        (f"https://www.youtube.com/channel/{CHANNEL_ID}/videos", "id", CHANNEL_ID),
        ("https://www.youtube.com/c/SomeName", "custom", "SomeName"),
        ("https://www.youtube.com/user/SomeName", "username", "SomeName"),
        ("@example", "handle", "example"),
        (CHANNEL_ID, "id", CHANNEL_ID),
    ],
)
def test_parse_channel_input(value: str, kind: str, expected: str) -> None:
    reference = parse_channel_input(value)
    assert reference == ChannelReference(kind=kind, value=expected)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        "",
        "https://www.youtube.com/playlist?list=PL123",
        "https://www.youtube.com/watch?v=abc",
        "https://youtu.be/abc",
        "https://www.youtube.com/shorts/abc",
        "https://example.com/@example",
        "https://www.youtube.com/",
        "example",
    ],
)
def test_parse_channel_input_rejects_non_channel_urls(value: str) -> None:
    with pytest.raises(ChannelInputError):
        parse_channel_input(value)


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("PT1H2M3S", 3723),
        ("PT45S", 45),
        ("PT2M", 120),
        ("PT1H", 3600),
        ("P1DT2H", 93600),
        ("P1W", 604800),
        ("P0D", 0),
        ("PT0S", 0),
        ("PT1.5S", 2),
    ],
)
def test_parse_iso8601_duration(value: str, seconds: int) -> None:
    assert parse_iso8601_duration(value) == seconds


@pytest.mark.parametrize("value", ["", "1:02", "PT", "hello", "P"])
def test_parse_iso8601_duration_rejects_invalid(value: str) -> None:
    with pytest.raises(ValueError):
        parse_iso8601_duration(value)


def _client(handler) -> tuple[YouTubeClient, httpx.Client]:
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return (
        YouTubeClient(
            "test-key",
            client=http_client,
            base_url="https://yt.test/youtube/v3",
        ),
        http_client,
    )


def test_resolve_channel_by_handle() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path.endswith("/channels")
        assert request.url.params["forHandle"] == "example"
        assert request.url.params["key"] == "test-key"
        return httpx.Response(200, json={"items": [_channel_item()]})

    client, http_client = _client(handler)
    try:
        channel = client.resolve_channel("https://www.youtube.com/@example")
        assert channel.id == CHANNEL_ID
        assert channel.title == "Test Channel"
        assert channel.uploads_playlist_id == "UU" + "a" * 22
        assert channel.url == f"https://www.youtube.com/channel/{CHANNEL_ID}"
        assert len(requests) == 1
    finally:
        http_client.close()


def test_list_playlists_paginates() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path.endswith("/playlists")
        assert request.url.params["channelId"] == CHANNEL_ID
        assert request.url.params["part"] == "snippet,contentDetails"
        assert request.url.params["maxResults"] == "50"
        if request.url.params.get("pageToken") is None:
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": "PL1",
                            "snippet": {"title": "One", "description": "First"},
                            "contentDetails": {"itemCount": 2},
                        }
                    ],
                    "nextPageToken": "page-2",
                },
            )
        assert request.url.params["pageToken"] == "page-2"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "PL2",
                        "snippet": {"title": "Two", "description": ""},
                        "contentDetails": {"itemCount": 5},
                    }
                ]
            },
        )

    client, http_client = _client(handler)
    try:
        playlists = client.list_playlists(CHANNEL_ID)
        assert [playlist.id for playlist in playlists] == ["PL1", "PL2"]
        assert playlists[1].item_count == 5
        assert len(requests) == 2
    finally:
        http_client.close()


def test_list_playlist_items_skips_deleted_and_private_and_paginates() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path.endswith("/playlistItems")
        assert request.url.params["playlistId"] == "PL1"
        if request.url.params.get("pageToken") is None:
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "snippet": {"title": "Deleted video", "position": 0},
                            "contentDetails": {"videoId": "deleted"},
                        },
                        {
                            "snippet": {"title": "Private video", "position": 1},
                            "contentDetails": {"videoId": "private"},
                        },
                        {
                            "snippet": {
                                "title": "First",
                                "position": 2,
                                "resourceId": {"videoId": "v1"},
                            },
                            "contentDetails": {"videoId": "v1"},
                        },
                    ],
                    "nextPageToken": "n",
                },
            )
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "snippet": {
                            "title": "Second",
                            "position": 3,
                            "resourceId": {"videoId": "v2"},
                        },
                        "contentDetails": {"videoId": "v2"},
                    }
                ]
            },
        )

    client, http_client = _client(handler)
    try:
        items = client.list_playlist_items("PL1")
        assert [(item.video_id, item.position) for item in items] == [
            ("v1", 2),
            ("v2", 3),
        ]
        assert len(requests) == 2
    finally:
        http_client.close()


def test_fetch_videos_batches_fifty_ids() -> None:
    video_ids = [f"v{index}" for index in range(60)]
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/videos")
        batch = request.url.params["id"].split(",")
        calls.append(batch)
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": video_id,
                        "snippet": {
                            "title": f"Title {video_id}",
                            "description": "desc",
                            "publishedAt": "2024-01-02T00:00:00Z",
                            "liveBroadcastContent": "none",
                            "defaultAudioLanguage": "en",
                        },
                        "contentDetails": {"duration": "PT1H2M3S"},
                    }
                    for video_id in batch
                ]
            },
        )

    client, http_client = _client(handler)
    try:
        videos = client.fetch_videos(video_ids)
        assert len(videos) == 60
        assert [len(batch) for batch in calls] == [50, 10]
        assert videos[0].duration_seconds == 3723
        assert videos[0].default_audio_language == "en"
    finally:
        http_client.close()


def test_quota_exceeded_is_reported_clearly() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            403,
            json={
                "error": {
                    "errors": [{"reason": "quotaExceeded"}],
                    "message": "The request cannot be completed because you have exceeded your quota.",
                }
            },
        )

    client, http_client = _client(handler)
    try:
        with pytest.raises(QuotaExceededError, match="quota"):
            client.list_playlists(CHANNEL_ID)
    finally:
        http_client.close()

    assert calls == 1


def test_youtube_retries_429_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                json={"error": {"message": "rate limited"}},
            )
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "PL1",
                        "snippet": {"title": "One", "description": ""},
                        "contentDetails": {"itemCount": 1},
                    }
                ]
            },
        )

    client, http_client = _client(handler)
    client._retrying.sleep = lambda _: None  # type: ignore[method-assign]
    try:
        playlists = client.list_playlists(CHANNEL_ID)
    finally:
        http_client.close()

    assert calls == 2
    assert [playlist.id for playlist in playlists] == ["PL1"]


def test_youtube_retries_403_rate_limit_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                403,
                json={
                    "error": {
                        "errors": [{"reason": "userRateLimitExceeded"}],
                        "message": "User rate limit exceeded",
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "PL1",
                        "snippet": {"title": "One", "description": ""},
                        "contentDetails": {"itemCount": 1},
                    }
                ]
            },
        )

    client, http_client = _client(handler)
    client._retrying.sleep = lambda _: None  # type: ignore[method-assign]
    try:
        playlists = client.list_playlists(CHANNEL_ID)
    finally:
        http_client.close()

    assert calls == 2
    assert [playlist.id for playlist in playlists] == ["PL1"]


def test_transport_error_redacts_api_key() -> None:
    key = "super-secret-youtube-key"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            f"cannot connect while using key={key}",
            request=request,
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = YouTubeClient(
        key,
        client=http_client,
        base_url="https://yt.test/youtube/v3",
    )
    client._retrying.sleep = lambda _: None  # type: ignore[method-assign]
    try:
        with pytest.raises(YouTubeError) as exc_info:
            client.list_playlists(CHANNEL_ID)
    finally:
        http_client.close()

    message = str(exc_info.value)
    assert "network error" in message
    assert key not in message
    assert "[redacted]" in message


def test_custom_channel_falls_back_to_search() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/channels"):
            calls.append("channels:" + ("handle" if "forHandle" in request.url.params else "id"))
            if "forHandle" in request.url.params or "forUsername" in request.url.params:
                return httpx.Response(200, json={"items": []})
            return httpx.Response(200, json={"items": [_channel_item()]})
        if path.endswith("/search"):
            calls.append("search")
            return httpx.Response(
                200,
                json={"items": [{"id": {"channelId": CHANNEL_ID}}]},
            )
        raise AssertionError(path)

    client, http_client = _client(handler)
    try:
        channel = client.resolve_channel("https://www.youtube.com/c/SomeName")
        assert channel.id == CHANNEL_ID
        assert "search" in calls
        assert "channels:id" in calls
    finally:
        http_client.close()


def test_fetch_refuses_to_mix_channels(tmp_path: Path) -> None:
    config = Config(
        channel="@new-channel",
        paths=PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "out"),
    )
    db = Database(config.db_path)
    db.upsert_channel(channel_id="UCold", title="Old")
    db.upsert_video(video_id="old1", channel_id="UCold", title="Old video")

    class FakeClient:
        def resolve_channel(self, value):
            return Channel(
                id="UCnew",
                title="New",
                description="",
                url="https://www.youtube.com/channel/UCnew",
                uploads_playlist_id=None,
            )

    try:
        with pytest.raises(ConfigError, match="one channel"):
            fetch_channel_data(
                config,
                db,
                client=FakeClient(),  # type: ignore[arg-type]
                write_snapshots=False,
            )
        assert db.list_channel_ids() == ["UCold"]
        assert db.get_channel_by_id("UCold") is not None
        assert any(video["id"] == "old1" for video in db.list_videos())
    finally:
        db.close()


def test_fetch_channel_stores_uploads_and_snapshots(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    config = Config(
        channel="https://www.youtube.com/@test",
        paths=PathsConfig(data_dir=data_dir, output_dir=tmp_path / "out"),
        transcripts=TranscriptsConfig(sleep_requests=0.0),
    )
    uploads_id = "UU" + "a" * 22
    playlist_id = "PLregular"
    requested_video_ids: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/channels"):
            return httpx.Response(200, json={"items": [_channel_item()]})
        if path.endswith("/playlists"):
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": playlist_id,
                            "snippet": {
                                "title": "Regular playlist",
                                "description": "A playlist",
                            },
                            "contentDetails": {"itemCount": 1},
                        }
                    ]
                },
            )
        if path.endswith("/playlistItems"):
            playlist = request.url.params["playlistId"]
            if playlist == uploads_id:
                items = [
                    _playlist_item("v1", 0),
                    _playlist_item("v2", 1),
                    {
                        "snippet": {"title": "Private video", "position": 2},
                        "contentDetails": {"videoId": "private"},
                    },
                ]
            elif playlist == playlist_id:
                items = [_playlist_item("v2", 0)]
            else:
                raise AssertionError(playlist)
            return httpx.Response(200, json={"items": items})
        if path.endswith("/videos"):
            ids = request.url.params["id"].split(",")
            requested_video_ids.append(ids)
            return httpx.Response(
                200,
                json={
                    "items": [_video_item(video_id, index) for index, video_id in enumerate(ids)]
                },
            )
        raise AssertionError(path)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    youtube = YouTubeClient(
        "test-key",
        client=http_client,
        base_url="https://yt.test/youtube/v3",
    )
    db = Database(config.db_path)
    db.upsert_playlist(
        playlist_id="PLstale",
        channel_id=CHANNEL_ID,
        title="Stale playlist",
        position=99,
    )
    db.replace_playlist_items("PLstale", [("stalevid", 0)])
    db.upsert_video(video_id="stalevid", channel_id=CHANNEL_ID, title="Stale video")
    db.upsert_transcript(video_id="stalevid", status="ok", text_path="/t/stale.txt")
    db.upsert_chapter(video_id="stalevid", status="cleaned", cleaned_path="/c/stale.txt")
    try:
        summary = fetch_channel_data(
            config,
            db,
            client=youtube,
            write_snapshots=True,
        )
        assert summary.channel_title == "Test Channel"
        assert summary.playlists == 1
        assert summary.uploads_videos == 2
        assert summary.videos == 2

        playlists = {playlist["id"]: playlist for playlist in db.list_playlists()}
        assert set(playlists) == {uploads_id, playlist_id}
        assert playlists[uploads_id]["kind"] == "uploads"
        assert playlists[uploads_id]["position"] == -1
        assert playlists[playlist_id]["kind"] == "regular"
        assert {item["video_id"] for item in db.list_playlist_items(uploads_id)} == {
            "v1",
            "v2",
        }
        assert [item["video_id"] for item in db.list_playlist_items(playlist_id)] == ["v2"]
        assert db.count_videos() == 2
        assert db.get_transcript("stalevid") is None
        assert db.get_chapter("stalevid") is None
        assert requested_video_ids == [["v1", "v2"]]

        api_dir = config.api_dir
        assert (api_dir / "channel.json").is_file()
        assert (api_dir / "playlists.json").is_file()
        assert (api_dir / "videos.json").is_file()
        snapshot_text = "".join(
            path.read_text(encoding="utf-8")
            for path in (
                api_dir / "channel.json",
                api_dir / "playlists.json",
                api_dir / "videos.json",
            )
        )
        assert "test-key" not in snapshot_text
        snapshots = json.loads((api_dir / "playlists.json").read_text(encoding="utf-8"))
        assert any(item["kind"] == "uploads" for item in snapshots)
    finally:
        db.close()
        http_client.close()


def _playlist_item(video_id: str, position: int) -> dict:
    return {
        "snippet": {
            "title": f"Video {video_id}",
            "position": position,
            "resourceId": {"videoId": video_id},
        },
        "contentDetails": {"videoId": video_id},
    }


def _video_item(video_id: str, index: int) -> dict:
    return {
        "id": video_id,
        "snippet": {
            "title": f"Video {video_id}",
            "description": f"Description {video_id}",
            "publishedAt": f"2024-01-0{index + 1}T00:00:00Z",
            "liveBroadcastContent": "none",
            "defaultAudioLanguage": "en",
        },
        "contentDetails": {"duration": "PT1M"},
    }
