"""Transcript acquisition via youtube-transcript-api and yt-dlp.

Only subtitles are downloaded.  Audio is never fetched and no speech-to-text
model is involved.  A video without a transcript in the requested languages is
recorded as skipped, not transcribed.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    AgeRestricted,
    InvalidVideoId,
    IpBlocked,
    NoTranscriptFound,
    PoTokenRequired,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
    YouTubeRequestFailed,
    YouTubeTranscriptApiException,
)

from .config import TranscriptsConfig
from .text import normalize_transcript
from .utils import atomic_write_text, language_matches, utc_now, utc_now_iso
from .vtt import vtt_to_text

LOGGER = logging.getLogger(__name__)

TranscriptStatus = Literal["ok", "skipped", "failed"]


@dataclass(frozen=True)
class TranscriptResult:
    """Outcome of one transcript fetch attempt across providers."""

    video_id: str
    status: TranscriptStatus
    language: str | None = None
    is_generated: bool | None = None
    source: str | None = None
    raw_path: str | None = None
    text_path: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class _ProviderOutcome:
    status: TranscriptStatus
    message: str = ""
    language: str | None = None
    is_generated: bool | None = None
    source: str | None = None
    raw_path: str | None = None
    text_path: str | None = None


@dataclass(frozen=True)
class _SubtitleFile:
    name: str
    content: str


class _TimeoutSession(requests.Session):
    """Requests session that supplies a default timeout to every request."""

    def __init__(self, timeout: float):
        super().__init__()
        self._default_timeout = timeout

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self._default_timeout
        return super().request(method, url, **kwargs)


class TranscriptFetcher:
    """Download subtitles for one video and persist raw and normalized files."""

    def __init__(self, transcripts_dir: str | Path, config: TranscriptsConfig):
        self.transcripts_dir = Path(transcripts_dir)
        self.config = config

    def fetch(
        self,
        video_id: str,
        *,
        languages: list[str] | None = None,
    ) -> TranscriptResult:
        """Fetch a transcript, trying the configured providers in order.

        Existing raw files are never overwritten; a new raw file is written with
        a unique suffix when the fetched contents differ.
        """

        if not video_id:
            return TranscriptResult(
                video_id=video_id,
                status="failed",
                error="Video ID is empty",
            )
        requested = _normalize_languages(languages or self.config.languages)
        if not requested:
            return TranscriptResult(
                video_id=video_id,
                status="failed",
                error="No transcript languages configured",
            )

        if self.config.sleep_requests > 0:
            time.sleep(self.config.sleep_requests)

        providers = _providers_for(self.config.provider)
        outcomes: list[tuple[str, _ProviderOutcome]] = []
        for provider in providers:
            if provider == "youtube-transcript-api":
                outcome = self._fetch_with_api(video_id, requested)
            else:
                outcome = self._fetch_with_ytdlp(video_id, requested)
            if outcome.status == "ok":
                return TranscriptResult(
                    video_id=video_id,
                    status="ok",
                    language=outcome.language,
                    is_generated=outcome.is_generated,
                    source=outcome.source,
                    raw_path=outcome.raw_path,
                    text_path=outcome.text_path,
                )
            outcomes.append((provider, outcome))
            LOGGER.debug(
                "Transcript provider %s for %s: %s (%s)",
                provider,
                video_id,
                outcome.status,
                outcome.message,
            )

        status: TranscriptStatus = (
            "failed" if any(outcome.status == "failed" for _, outcome in outcomes) else "skipped"
        )
        message = "; ".join(f"{provider}: {outcome.message}" for provider, outcome in outcomes)
        return TranscriptResult(video_id=video_id, status=status, error=message)

    def _fetch_with_api(self, video_id: str, languages: list[str]) -> _ProviderOutcome:
        try:
            api = YouTubeTranscriptApi(http_client=_TimeoutSession(self.config.timeout_seconds))
            transcript_list = api.list(video_id)
            candidates = [
                transcript
                for transcript in transcript_list
                if language_matches(getattr(transcript, "language_code", None), languages)
            ]
            if not candidates:
                return _ProviderOutcome(
                    status="skipped",
                    message=("No transcript in requested languages " f"{', '.join(languages)}"),
                )
            selected = min(
                candidates,
                key=lambda transcript: (
                    bool(getattr(transcript, "is_generated", False)),
                    _language_priority(getattr(transcript, "language_code", None), languages),
                ),
            )
            fetched = selected.fetch()
        except (TranscriptsDisabled, NoTranscriptFound) as exc:
            return _ProviderOutcome(
                status="skipped",
                message=f"Transcript unavailable in {', '.join(languages)} ({type(exc).__name__})",
            )
        except (VideoUnavailable, VideoUnplayable, AgeRestricted) as exc:
            return _ProviderOutcome(
                status="skipped",
                message=f"Video cannot provide a transcript ({type(exc).__name__})",
            )
        except (IpBlocked, RequestBlocked, PoTokenRequired, YouTubeRequestFailed) as exc:
            return _ProviderOutcome(
                status="failed",
                message=(
                    f"YouTube blocked or failed the transcript request ({type(exc).__name__}); "
                    "check your network/cookies and retry"
                ),
            )
        except InvalidVideoId as exc:
            return _ProviderOutcome(status="failed", message=f"Invalid video ID ({exc})")
        except YouTubeTranscriptApiException as exc:
            return _ProviderOutcome(
                status="failed",
                message=f"youtube-transcript-api failed ({type(exc).__name__}: {exc})",
            )
        except Exception as exc:  # defensive: a provider must not kill the run
            LOGGER.warning("Unexpected transcript API error for %s: %s", video_id, exc)
            return _ProviderOutcome(
                status="failed",
                message=f"Unexpected youtube-transcript-api error ({type(exc).__name__}: {exc})",
            )

        raw_snippets = list(fetched.to_raw_data())
        text = normalize_transcript(
            " ".join(str(snippet.get("text") or "") for snippet in raw_snippets)
        )
        if not text:
            return _ProviderOutcome(status="skipped", message="Transcript is empty")

        language_code = getattr(fetched, "language_code", None) or getattr(
            selected, "language_code", None
        )
        language_name = (
            getattr(fetched, "language", None)
            or getattr(selected, "language", None)
            or language_code
        )
        is_generated = bool(
            getattr(fetched, "is_generated", getattr(selected, "is_generated", False))
        )
        raw_path = self._store_raw_unique(
            self.transcripts_dir / f"{video_id}.json",
            json.dumps(
                {
                    "video_id": video_id,
                    "source": "youtube-transcript-api",
                    "language": language_name,
                    "language_code": language_code,
                    "is_generated": is_generated,
                    "fetched_at": utc_now_iso(),
                    "snippets": raw_snippets,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        text_path = self._store_text(video_id, text)
        return _ProviderOutcome(
            status="ok",
            language=str(language_code or ""),
            is_generated=is_generated,
            source="youtube-transcript-api",
            raw_path=str(raw_path),
            text_path=str(text_path),
        )

    def _fetch_with_ytdlp(self, video_id: str, languages: list[str]) -> _ProviderOutcome:
        manual_files, manual_error = self._run_ytdlp(video_id, languages, auto=False)
        selected = _select_vtt(manual_files, video_id, languages)
        if selected is not None:
            subtitle, language = selected
            return self._store_vtt_result(
                video_id,
                subtitle,
                language=language,
                is_generated=False,
                source="yt-dlp-manual",
            )

        auto_files, auto_error = self._run_ytdlp(video_id, languages, auto=True)
        selected = _select_vtt(auto_files, video_id, languages, require_original=True)
        if selected is not None:
            subtitle, language = selected
            return self._store_vtt_result(
                video_id,
                subtitle,
                language=language,
                is_generated=True,
                source="yt-dlp-auto",
            )

        errors = [error for error in (manual_error, auto_error) if error]
        if errors:
            return _ProviderOutcome(
                status="failed",
                message="; ".join(errors),
            )
        return _ProviderOutcome(
            status="skipped",
            message=f"yt-dlp found no subtitles in {', '.join(languages)}",
        )

    def _run_ytdlp(
        self,
        video_id: str,
        languages: list[str],
        *,
        auto: bool,
    ) -> tuple[list[_SubtitleFile], str | None]:
        with tempfile.TemporaryDirectory(prefix="ybook-subs-") as temporary:
            output_dir = Path(temporary)
            command = [
                sys.executable,
                "-m",
                "yt_dlp",
                "--skip-download",
                "--no-playlist",
                "--no-progress",
                "--write-auto-subs" if auto else "--write-subs",
                "--sub-langs",
                ",".join(
                    f"{language}.*-orig" if auto else f"{language}.*" for language in languages
                ),
                "--sub-format",
                "vtt",
                "--retries",
                "3",
                "--socket-timeout",
                "30",
                "--sleep-requests",
                f"{self.config.sleep_requests:g}",
                "-o",
                str(output_dir / "%(id)s.%(ext)s"),
            ]
            if self.config.cookies_from_browser:
                command.extend(["--cookies-from-browser", self.config.cookies_from_browser])
            command.append(f"https://www.youtube.com/watch?v={video_id}")

            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.config.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return [], f"yt-dlp timed out after {self.config.timeout_seconds}s"
            except OSError as exc:
                return [], f"yt-dlp could not start: {exc}"

            files: list[_SubtitleFile] = []
            for path in sorted(output_dir.glob("*.vtt")):
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                files.append(_SubtitleFile(name=path.name, content=content))
            if files:
                return files, None
            if completed.returncode != 0:
                stderr = (completed.stderr or "").strip()
                if _is_bot_check(stderr):
                    return [], (
                        "yt-dlp hit YouTube's bot/rate-limit check; update yt-dlp, "
                        "configure cookies_from_browser, or retry later"
                    )
                tail = stderr.splitlines()[-1] if stderr else "no stderr"
                return [], f"yt-dlp exited with code {completed.returncode}: {tail[:400]}"
            return [], None

    def _store_vtt_result(
        self,
        video_id: str,
        subtitle: _SubtitleFile,
        *,
        language: str,
        is_generated: bool,
        source: str,
    ) -> _ProviderOutcome:
        raw_text = subtitle.content
        text = vtt_to_text(raw_text)
        if not text:
            return _ProviderOutcome(status="skipped", message="Parsed subtitle text is empty")

        raw_path = self._store_raw_unique(self.transcripts_dir / f"{video_id}.vtt", raw_text)
        text_path = self._store_text(video_id, text)
        return _ProviderOutcome(
            status="ok",
            language=language,
            is_generated=is_generated,
            source=source,
            raw_path=str(raw_path),
            text_path=str(text_path),
        )

    def _store_raw_unique(self, path: Path, content: str) -> Path:
        """Never overwrite an existing raw transcript file."""

        if not path.exists():
            return atomic_write_text(path, content)
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError:
            existing = None
        if existing == content:
            return path
        stamp = utc_now().strftime("%Y%m%dT%H%M%S%f")
        unique = path.with_name(f"{path.stem}.{stamp}{path.suffix}")
        return atomic_write_text(unique, content)

    def _store_text(self, video_id: str, text: str) -> Path:
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        return atomic_write_text(self.transcripts_dir / f"{video_id}.txt", text + "\n")


def _providers_for(provider: str) -> list[str]:
    if provider == "auto":
        return ["youtube-transcript-api", "yt-dlp"]
    if provider in {"youtube-transcript-api", "yt-dlp"}:
        return [provider]
    raise ValueError(f"Unknown transcript provider: {provider}")


def _normalize_languages(languages: list[str]) -> list[str]:
    normalized = [language.strip() for language in languages if language and language.strip()]
    return list(dict.fromkeys(normalized))


def _language_priority(language_code: str | None, languages: list[str]) -> int:
    folded = (language_code or "").strip().casefold()
    if not folded:
        return len(languages)
    for index, requested in enumerate(languages):
        code = requested.strip().casefold()
        if code and (folded == code or folded.startswith(f"{code}-")):
            return index
    return len(languages)


def _select_vtt(
    files: list[_SubtitleFile],
    video_id: str,
    languages: list[str],
    *,
    require_original: bool = False,
) -> tuple[_SubtitleFile, str] | None:
    if not files:
        return None

    tagged: list[tuple[_SubtitleFile, str]] = []
    for subtitle in files:
        name = subtitle.name
        stem = name[:-4] if name.lower().endswith(".vtt") else Path(name).stem
        prefix = f"{video_id}."
        if stem.startswith(prefix):
            language = stem[len(prefix) :]
        else:
            parts = stem.split(".")
            language = parts[-1] if len(parts) > 1 else ""
        if require_original and not language.casefold().endswith("-orig"):
            continue
        tagged.append((subtitle, language))

    for requested in languages:
        folded = requested.casefold()
        exact = [(subtitle, tag) for subtitle, tag in tagged if tag.casefold() == folded]
        if exact:
            return exact[0][0], exact[0][1]
        prefixed = [
            (subtitle, tag)
            for subtitle, tag in tagged
            if tag and tag.casefold().startswith(f"{folded}-")
        ]
        if prefixed:
            return prefixed[0][0], prefixed[0][1]

    if not require_original:
        untagged = [(subtitle, tag) for subtitle, tag in tagged if not tag]
        if len(untagged) == 1 and len(files) == 1:
            return untagged[0][0], languages[0]
    return None


def _is_bot_check(stderr: str) -> bool:
    lowered = stderr.casefold()
    return any(
        marker in lowered
        for marker in (
            "sign in to confirm",
            "not a bot",
            "confirm you’re not a bot",
            "confirm you're not a bot",
            "http error 429",
            "too many requests",
        )
    )
