"""yt-dlp fallback tests without network or a real yt-dlp process."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import requests

from ybook.config import TranscriptsConfig
from ybook.transcripts import TranscriptFetcher

MANUAL_VTT = """WEBVTT

00:00:01.000 --> 00:00:02.000
Manual subtitle text.
"""

AUTO_VTT = """WEBVTT

00:00:01.000 --> 00:00:02.000
Automatic subtitle text.
"""


def _config(**overrides) -> TranscriptsConfig:
    values = {
        "provider": "yt-dlp",
        "sleep_requests": 0.0,
        "timeout_seconds": 10,
    }
    values.update(overrides)
    return TranscriptsConfig(**values)


def test_ytdlp_manual_subtitles_are_read_before_temp_dir_is_deleted(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        template = Path(command[command.index("-o") + 1])
        (template.parent / "vid1.en.vtt").write_text(MANUAL_VTT, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("ybook.transcripts.subprocess.run", fake_run)
    fetcher = TranscriptFetcher(tmp_path / "transcripts", _config())

    result = fetcher.fetch("vid1", languages=["en"])

    assert result.status == "ok"
    assert result.source == "yt-dlp-manual"
    assert result.is_generated is False
    assert result.language == "en"
    assert result.raw_path is not None
    assert Path(result.raw_path).read_text(encoding="utf-8") == MANUAL_VTT
    assert result.text_path is not None
    assert Path(result.text_path).read_text(encoding="utf-8").strip() == ("Manual subtitle text.")
    assert len(calls) == 1
    assert "--skip-download" in calls[0]
    assert "--no-playlist" in calls[0]
    assert "--write-subs" in calls[0]
    assert "--write-auto-subs" not in calls[0]
    assert calls[0][calls[0].index("--sub-langs") + 1] == "en.*"


def test_ytdlp_falls_back_to_auto_subtitles(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "--write-auto-subs" in command:
            template = Path(command[command.index("-o") + 1])
            (template.parent / "vid1.en-orig.vtt").write_text(AUTO_VTT, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("ybook.transcripts.subprocess.run", fake_run)
    fetcher = TranscriptFetcher(tmp_path / "transcripts", _config())

    result = fetcher.fetch("vid1", languages=["en"])

    assert result.status == "ok"
    assert result.source == "yt-dlp-auto"
    assert result.is_generated is True
    assert result.language == "en-orig"
    assert len(calls) == 2
    assert "--write-subs" in calls[0]
    assert "--write-auto-subs" in calls[1]
    assert calls[1][calls[1].index("--sub-langs") + 1] == "en.*-orig"


def test_ytdlp_translated_auto_track_is_rejected(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "--write-auto-subs" in command:
            template = Path(command[command.index("-o") + 1])
            (template.parent / "vid1.en.vtt").write_text(AUTO_VTT, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("ybook.transcripts.subprocess.run", fake_run)
    fetcher = TranscriptFetcher(tmp_path / "transcripts", _config())

    result = fetcher.fetch("vid1", languages=["en"])

    assert result.status == "skipped"
    assert len(calls) == 2
    assert not (tmp_path / "transcripts" / "vid1.vtt").exists()
    assert not (tmp_path / "transcripts" / "vid1.txt").exists()


def test_ytdlp_language_priority_and_cookies(tmp_path: Path, monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        template = Path(command[command.index("-o") + 1])
        directory = template.parent
        (directory / "vid1.en.vtt").write_text(MANUAL_VTT, encoding="utf-8")
        (directory / "vid1.ru.vtt").write_text(
            MANUAL_VTT.replace("Manual", "Russian"), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("ybook.transcripts.subprocess.run", fake_run)
    fetcher = TranscriptFetcher(
        tmp_path / "transcripts",
        _config(cookies_from_browser="firefox"),
    )

    result = fetcher.fetch("vid1", languages=["ru", "en"])

    assert result.language == "ru"
    assert "--cookies-from-browser" in commands[0]
    assert commands[0][commands[0].index("--cookies-from-browser") + 1] == "firefox"
    assert commands[0][commands[0].index("--sub-langs") + 1] == "ru.*,en.*"


def test_ytdlp_no_subtitles_marks_skipped(tmp_path: Path, monkeypatch) -> None:
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("ybook.transcripts.subprocess.run", fake_run)
    fetcher = TranscriptFetcher(tmp_path / "transcripts", _config())

    result = fetcher.fetch("vid1", languages=["en"])

    assert result.status == "skipped"
    assert result.error and "no subtitles" in result.error


def test_ytdlp_bot_check_is_failed(tmp_path: Path, monkeypatch) -> None:
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="ERROR: Sign in to confirm you're not a bot",
        )

    monkeypatch.setattr("ybook.transcripts.subprocess.run", fake_run)
    fetcher = TranscriptFetcher(tmp_path / "transcripts", _config())

    result = fetcher.fetch("vid1", languages=["en"])

    assert result.status == "failed"
    assert result.error and "bot" in result.error.lower()


def test_raw_subtitle_file_is_never_overwritten(tmp_path: Path, monkeypatch) -> None:
    current = {"content": MANUAL_VTT}

    def fake_run(command, **kwargs):
        template = Path(command[command.index("-o") + 1])
        (template.parent / "vid1.en.vtt").write_text(current["content"], encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("ybook.transcripts.subprocess.run", fake_run)
    fetcher = TranscriptFetcher(tmp_path / "transcripts", _config())

    first = fetcher.fetch("vid1", languages=["en"])
    assert first.raw_path is not None
    original_path = Path(first.raw_path)
    assert original_path.read_text(encoding="utf-8") == MANUAL_VTT

    current["content"] = MANUAL_VTT.replace("Manual", "Changed")
    second = fetcher.fetch("vid1", languages=["en"])

    assert second.raw_path is not None
    assert Path(second.raw_path) != original_path
    assert original_path.read_text(encoding="utf-8") == MANUAL_VTT
    assert Path(second.raw_path).read_text(encoding="utf-8") == current["content"]


def test_youtube_transcript_api_saves_metadata_and_text(tmp_path: Path, monkeypatch) -> None:
    class FakeFetched:
        language = "English"
        language_code = "en"
        is_generated = False

        def to_raw_data(self):
            return [
                {"text": "Hello", "start": 0.0, "duration": 1.0},
                {"text": "world", "start": 1.0, "duration": 1.0},
            ]

    class FakeTranscript:
        language = "English"
        language_code = "en"
        is_generated = False

        def fetch(self):
            return FakeFetched()

    class FakeApi:
        def __init__(self, **kwargs):
            assert "http_client" in kwargs

        def list(self, video_id):
            assert video_id == "vid1"
            return [FakeTranscript()]

    monkeypatch.setattr("ybook.transcripts.YouTubeTranscriptApi", FakeApi)
    fetcher = TranscriptFetcher(
        tmp_path / "transcripts",
        _config(provider="youtube-transcript-api"),
    )

    result = fetcher.fetch("vid1", languages=["en"])

    assert result.status == "ok"
    assert result.source == "youtube-transcript-api"
    assert result.language == "en"
    assert result.is_generated is False
    assert result.raw_path is not None
    raw = json.loads(Path(result.raw_path).read_text(encoding="utf-8"))
    assert raw["language"] == "English"
    assert raw["language_code"] == "en"
    assert raw["is_generated"] is False
    assert raw["snippets"][0]["text"] == "Hello"
    assert result.text_path is not None
    assert Path(result.text_path).read_text(encoding="utf-8").strip() == "Hello world"


def test_youtube_transcript_api_accepts_language_variant(tmp_path: Path, monkeypatch) -> None:
    class FakeFetched:
        language = "English (United States)"
        language_code = "en-US"
        is_generated = True

        def to_raw_data(self):
            return [{"text": "Variant text", "start": 0.0, "duration": 1.0}]

    class FakeTranscript:
        language = "English (United States)"
        language_code = "en-US"
        is_generated = True

        def fetch(self):
            return FakeFetched()

    class FakeApi:
        def __init__(self, **kwargs):
            pass

        def list(self, video_id):
            return [FakeTranscript()]

    monkeypatch.setattr("ybook.transcripts.YouTubeTranscriptApi", FakeApi)
    fetcher = TranscriptFetcher(
        tmp_path / "transcripts",
        _config(provider="youtube-transcript-api"),
    )

    result = fetcher.fetch("vid1", languages=["en"])

    assert result.status == "ok"
    assert result.language == "en-US"
    assert result.is_generated is True


def test_youtube_transcript_api_prefers_manual_over_generated(tmp_path: Path, monkeypatch) -> None:
    class FakeFetched:
        language = "English (United States)"
        language_code = "en-US"
        is_generated = False

        def to_raw_data(self):
            return [{"text": "Manual variant", "start": 0.0, "duration": 1.0}]

    class FakeTranscript:
        def __init__(self, language_code: str, is_generated: bool):
            self.language_code = language_code
            self.language = language_code
            self.is_generated = is_generated

        def fetch(self):
            return FakeFetched()

    class FakeApi:
        def __init__(self, **kwargs):
            pass

        def list(self, video_id):
            return [
                FakeTranscript("en", True),
                FakeTranscript("en-US", False),
            ]

    monkeypatch.setattr("ybook.transcripts.YouTubeTranscriptApi", FakeApi)
    fetcher = TranscriptFetcher(
        tmp_path / "transcripts",
        _config(provider="youtube-transcript-api"),
    )

    result = fetcher.fetch("vid1", languages=["en"])

    assert result.status == "ok"
    assert result.language == "en-US"
    assert result.is_generated is False


def test_timeout_session_supplies_default_timeout(monkeypatch) -> None:
    from ybook.transcripts import _TimeoutSession

    captured: dict[str, object] = {}

    def fake_request(self, method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["timeout"] = kwargs.get("timeout")
        return requests.Response()

    monkeypatch.setattr(requests.Session, "request", fake_request)
    session = _TimeoutSession(12.5)
    session.request("GET", "https://example.test")
    assert captured == {
        "method": "GET",
        "url": "https://example.test",
        "timeout": 12.5,
    }

    session.request("GET", "https://example.test", timeout=3.0)
    assert captured["timeout"] == 3.0
