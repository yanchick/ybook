"""High-level orchestration for fetch, transcripts, clean, and build."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .db import Database
from .epub import book_from_database, build_epub
from .errors import ConfigError, LLMError, TranscriptError
from .llm import (
    AnnotationResult,
    LLMCache,
    LLMClient,
    clean_text,
    generate_annotation,
)
from .prompts import ChapterContext
from .transcripts import TranscriptFetcher, TranscriptResult
from .utils import (
    atomic_write_json,
    atomic_write_text,
    language_matches,
    utc_now_iso,
)
from .youtube_api import Channel, YouTubeClient, parse_channel_input

LOGGER = logging.getLogger(__name__)

ANNOTATION_FAILED_WARNING = "annotation failed"


@dataclass(frozen=True)
class FetchSummary:
    """Result of a channel fetch."""

    channel_title: str
    playlists: int
    uploads_videos: int
    videos: int


@dataclass(frozen=True)
class TranscriptSummary:
    """Result of transcript acquisition."""

    selected: int
    ok: int
    skipped: int
    failed: int
    already_done: int


@dataclass(frozen=True)
class CleanSummary:
    """Result of cleaning and annotation."""

    selected: int
    cleaned: int
    needs_review: int
    failed: int
    annotations: int


@dataclass(frozen=True)
class BuildSummary:
    """Result of EPUB assembly."""

    path: Path
    parts: int
    chapters: int


@dataclass(frozen=True)
class _ChapterWorkResult:
    video_id: str
    status: str
    cleaned_path: str | None
    annotation: str
    warnings: tuple[str, ...]
    tokens_in: int
    tokens_out: int
    cost_usd: float
    error: str | None = None
    model: str | None = None


def _failed_chapter_result(row: dict[str, Any], error: str) -> _ChapterWorkResult:
    """Build a failed result that preserves the chapter's existing state."""

    return _ChapterWorkResult(
        video_id=str(row["video_id"]),
        status="failed",
        cleaned_path=None,
        annotation=str(row.get("existing_annotation") or ""),
        warnings=_warnings_from_db(row.get("existing_warnings")),
        tokens_in=_as_int(row.get("existing_tokens_in")),
        tokens_out=_as_int(row.get("existing_tokens_out")),
        cost_usd=_as_float(row.get("existing_cost_usd")),
        model=str(row.get("existing_model") or "") or None,
        error=error,
    )


def fetch_channel_data(
    config: Config,
    db: Database,
    *,
    channel: str | None = None,
    client: YouTubeClient | None = None,
    write_snapshots: bool = True,
) -> FetchSummary:
    """Fetch a channel, its playlists, uploads, and video metadata into SQLite."""

    channel_value = (channel or config.channel).strip()
    if not channel_value:
        raise ConfigError("Channel is not configured. Pass --channel or set channel in ybook.toml.")
    config.ensure_directories()
    parse_channel_input(channel_value)

    owns_client = client is None
    if client is None:
        client = YouTubeClient(config.youtube_api_key())

    try:
        resolved = client.resolve_channel(channel_value)
        other_channels = [
            channel_id for channel_id in db.list_channel_ids() if channel_id != resolved.id
        ]
        if other_channels:
            raise ConfigError(
                "This database already contains another channel "
                f"({', '.join(other_channels)}). ybook stores one channel per data "
                "directory; use a fresh data/ directory or remove data/state.sqlite."
            )
        LOGGER.info("Resolved channel: %s (%s)", resolved.title, resolved.id)
        db.upsert_channel(
            channel_id=resolved.id,
            title=resolved.title,
            description=resolved.description,
            url=resolved.url,
            uploads_playlist_id=resolved.uploads_playlist_id,
        )

        regular_playlists = client.list_playlists(resolved.id)
        playlist_rows: list[dict[str, Any]] = []
        playlist_members: dict[str, list[str]] = {}

        for position, playlist in enumerate(regular_playlists):
            db.upsert_playlist(
                playlist_id=playlist.id,
                channel_id=resolved.id,
                title=playlist.title,
                description=playlist.description,
                item_count=playlist.item_count,
                position=position,
                kind="regular",
            )
            items = client.list_playlist_items(playlist.id)
            db.replace_playlist_items(
                playlist.id, [(item.video_id, item.position) for item in items]
            )
            playlist_members[playlist.id] = [item.video_id for item in items]
            playlist_rows.append(
                {
                    **asdict(playlist),
                    "kind": "regular",
                    "position": position,
                }
            )
            LOGGER.info(
                "Playlist %s: %s items (%d/%d)",
                playlist.title,
                len(items),
                position + 1,
                len(regular_playlists),
            )

        upload_ids: list[str] = []
        if resolved.uploads_playlist_id:
            upload_items = client.list_playlist_items(resolved.uploads_playlist_id)
            upload_ids = [item.video_id for item in upload_items]
            db.upsert_playlist(
                playlist_id=resolved.uploads_playlist_id,
                channel_id=resolved.id,
                title="Uploads",
                description="All public uploads.",
                item_count=len(upload_items),
                position=-1,
                kind="uploads",
            )
            db.replace_playlist_items(
                resolved.uploads_playlist_id,
                [(item.video_id, item.position) for item in upload_items],
            )
            playlist_rows.insert(
                0,
                {
                    "id": resolved.uploads_playlist_id,
                    "title": "Uploads",
                    "description": "All public uploads.",
                    "item_count": len(upload_items),
                    "kind": "uploads",
                    "position": -1,
                },
            )
            LOGGER.info("Uploads playlist: %d items", len(upload_items))
        else:
            LOGGER.warning("Channel has no uploads playlist in contentDetails")

        all_video_ids = [
            *upload_ids,
            *[video for ids in playlist_members.values() for video in ids],
        ]
        videos = client.fetch_videos(all_video_ids)
        for video in videos:
            db.upsert_video(
                video_id=video.id,
                channel_id=resolved.id,
                title=video.title,
                description=video.description,
                published_at=video.published_at,
                duration_seconds=video.duration_seconds,
                default_audio_language=video.default_audio_language,
                live_status=video.live_status,
            )

        current_playlist_ids = {playlist.id for playlist in regular_playlists}
        if resolved.uploads_playlist_id:
            current_playlist_ids.add(resolved.uploads_playlist_id)
        removed = db.reconcile_channel(
            resolved.id,
            playlist_ids=current_playlist_ids,
            video_ids=set(all_video_ids),
        )
        if removed["playlists"] or removed["videos"]:
            LOGGER.info(
                "Removed stale channel data: %d playlists, %d videos",
                removed["playlists"],
                removed["videos"],
            )

        if write_snapshots:
            _write_snapshots(config, resolved, playlist_rows, videos)

        LOGGER.info(
            "Fetched %d playlists, %d upload entries, %d videos",
            len(regular_playlists),
            len(upload_ids),
            len(videos),
        )
        return FetchSummary(
            channel_title=resolved.title,
            playlists=len(regular_playlists),
            uploads_videos=len(upload_ids),
            videos=len(videos),
        )
    finally:
        if owns_client:
            client.close()


def run_transcripts(
    config: Config,
    db: Database,
    *,
    language: str | None = None,
    limit: int | None = None,
    force: bool = False,
    workers: int | None = None,
    provider: str | None = None,
) -> TranscriptSummary:
    """Fetch transcripts for videos, skipping already successful work."""

    config.ensure_directories()
    videos = db.list_videos()
    if not videos:
        raise TranscriptError("No videos in the database. Run 'ybook fetch' first.")
    if provider is not None and provider not in {
        "auto",
        "youtube-transcript-api",
        "yt-dlp",
    }:
        raise ConfigError(f"Unknown transcript provider: {provider}")
    if workers is not None and workers < 1:
        raise ConfigError("workers must be at least 1")

    languages = (
        [language.strip()] if language and language.strip() else list(config.transcripts.languages)
    )
    if not languages:
        raise ConfigError("No transcript language configured")

    updates: dict[str, Any] = {}
    if provider:
        updates["provider"] = provider
    if workers is not None:
        updates["workers"] = workers
    transcript_config = config.transcripts.model_copy(update=updates)
    worker_count = transcript_config.workers

    candidate_state: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    already_done = 0
    for video in videos:
        existing = db.get_transcript(str(video["id"]))
        if not force and _transcript_is_complete(existing, languages):
            already_done += 1
            continue
        candidate_state.append((video, existing))

    if limit is not None:
        candidate_state = candidate_state[: max(0, limit)]

    if not candidate_state:
        return TranscriptSummary(
            selected=0,
            ok=0,
            skipped=0,
            failed=0,
            already_done=already_done,
        )

    candidates: list[dict[str, Any]] = []
    for video, existing in candidate_state:
        if (
            existing is not None
            and existing.get("status") == "ok"
            and not language_matches(existing.get("language"), languages)
        ):
            _invalidate_cleaned_chapter(config, db, str(video["id"]))
        candidates.append(video)

    fetcher = TranscriptFetcher(config.transcripts_dir, transcript_config)

    def fetch_one(video: dict[str, Any]) -> TranscriptResult:
        video_id = str(video["id"])
        try:
            return fetcher.fetch(video_id, languages=languages)
        except Exception as exc:  # a single video must never abort the run
            LOGGER.exception("Unexpected transcript failure for %s", video_id)
            return TranscriptResult(
                video_id=video_id,
                status="failed",
                error=f"Unexpected error: {exc}",
            )

    counts = {"ok": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(fetch_one, video): video for video in candidates}
        for future in as_completed(futures):
            video = futures[future]
            result = future.result()
            counts[result.status] += 1
            db.upsert_transcript(
                video_id=result.video_id,
                status=result.status,
                language=result.language,
                is_generated=result.is_generated,
                source=result.source,
                raw_path=result.raw_path,
                text_path=result.text_path,
                error=result.error,
            )
            if result.status == "ok":
                _invalidate_cleaned_chapter(config, db, result.video_id)
                LOGGER.info(
                    "Transcript ok: %s (%s, %s)",
                    video["title"],
                    result.language,
                    result.source,
                )
            elif result.status == "skipped":
                LOGGER.warning("Transcript skipped: %s (%s)", video["title"], result.error)
            else:
                LOGGER.error("Transcript failed: %s (%s)", video["title"], result.error)

    return TranscriptSummary(
        selected=len(candidates),
        ok=counts["ok"],
        skipped=counts["skipped"],
        failed=counts["failed"],
        already_done=already_done,
    )


def run_clean(
    config: Config,
    db: Database,
    *,
    limit: int | None = None,
    force: bool = False,
    model: str | None = None,
    workers: int | None = None,
    annotations: bool | None = None,
    annotation_language: str | None = None,
    llm_enabled: bool | None = None,
) -> CleanSummary:
    """Clean transcripts and generate annotations concurrently.

    Network work happens in worker threads; SQLite writes happen only in this
    main thread. When LLM use is disabled, normalized transcripts are copied
    unchanged and no LLM client or API key is needed.
    """

    config.ensure_directories()
    use_llm = config.llm.enabled if llm_enabled is None else llm_enabled
    annotations_enabled = (
        config.book.annotations if annotations is None else annotations
    ) and use_llm
    channel = db.get_channel()
    current_channel_id = str((channel or {}).get("id") or "")
    rows = db.list_clean_candidates()
    if current_channel_id:
        channel_rows = [
            row for row in rows if str(row.get("channel_id") or "") == current_channel_id
        ]
        if len(channel_rows) != len(rows):
            LOGGER.warning("Ignoring cleaning candidates from non-current channels")
        rows = channel_rows

    candidates: list[dict[str, Any]] = []
    for row in rows:
        if force or not _chapter_is_complete(row):
            candidates.append(row)
            continue
        if annotations_enabled and not _annotation_present(row):
            candidates.append(row)
    if limit is not None:
        candidates = candidates[: max(0, limit)]

    if not candidates:
        return CleanSummary(
            selected=0,
            cleaned=0,
            needs_review=0,
            failed=0,
            annotations=0,
        )

    if not use_llm:
        if model or workers is not None:
            LOGGER.warning("LLM is disabled; ignoring model and worker overrides")
        return _run_clean_without_llm(config, db, candidates, force=force)

    llm_config = config.llm
    if model and model.strip():
        llm_config = llm_config.model_copy(update={"model": model.strip()})
    worker_count = workers if workers is not None else llm_config.workers
    if worker_count < 1:
        raise ConfigError("workers must be at least 1")

    api_key = config.llm_api_key()
    cache = LLMCache(config.llm_cache_dir)
    channel_title = str((channel or {}).get("title") or "YouTube")
    playlist_titles = db.primary_playlist_titles()
    annotation_language_value = (
        annotation_language.strip()
        if annotation_language and annotation_language.strip()
        else config.book.language
    )

    def process_one(row: dict[str, Any]) -> _ChapterWorkResult:
        video_id = str(row["video_id"])
        existing_annotation = str(row.get("existing_annotation") or "")
        existing_warnings = _warnings_from_db(row.get("existing_warnings"))
        existing_tokens_in = _as_int(row.get("existing_tokens_in"))
        existing_tokens_out = _as_int(row.get("existing_tokens_out"))
        existing_cost = _as_float(row.get("existing_cost_usd"))
        existing_cleaned_path = str(row.get("existing_cleaned_path") or "")
        existing_status = str(row.get("chapter_status") or "")
        context = ChapterContext(
            channel=channel_title,
            playlist=playlist_titles.get(video_id, "Uploads"),
            video_title=str(row.get("title") or "Untitled video"),
        )

        def failed(
            error: str,
            *,
            cleaned_path: str | None = None,
            annotation: str = existing_annotation,
            warnings: tuple[str, ...] = existing_warnings,
            tokens_in: int = existing_tokens_in,
            tokens_out: int = existing_tokens_out,
            cost_usd: float = existing_cost,
        ) -> _ChapterWorkResult:
            return _ChapterWorkResult(
                video_id=video_id,
                status="failed",
                cleaned_path=cleaned_path,
                annotation=annotation,
                warnings=warnings,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                error=error,
            )

        annotation_only = (
            annotations_enabled
            and not force
            and _chapter_is_complete(row)
            and not existing_annotation.strip()
        )

        if annotation_only:
            cleaned_path = Path(existing_cleaned_path)
            try:
                chapter_text = cleaned_path.read_text(encoding="utf-8")
            except OSError as exc:
                return failed(f"Cannot read cleaned chapter {cleaned_path}: {exc}")

            annotation, annotation_error = _annotation_or_error(
                client,
                context,
                annotation_language_value,
                chapter_text,
            )
            if annotation is not None:
                warnings = tuple(
                    warning for warning in existing_warnings if warning != ANNOTATION_FAILED_WARNING
                )
                status = existing_status
                if status == "needs_review" and not warnings:
                    status = "cleaned"
                _write_annotation(config, video_id, annotation.text)
                return _ChapterWorkResult(
                    video_id=video_id,
                    status=status,
                    cleaned_path=str(cleaned_path),
                    annotation=annotation.text,
                    warnings=warnings,
                    tokens_in=existing_tokens_in + annotation.prompt_tokens,
                    tokens_out=existing_tokens_out + annotation.completion_tokens,
                    cost_usd=existing_cost + annotation.cost_usd,
                )

            LOGGER.warning("Annotation backfill failed for %s: %s", video_id, annotation_error)
            _remove_annotation(config, video_id)
            return _ChapterWorkResult(
                video_id=video_id,
                status="needs_review",
                cleaned_path=str(cleaned_path),
                annotation="",
                warnings=_add_warning(existing_warnings, ANNOTATION_FAILED_WARNING),
                tokens_in=existing_tokens_in,
                tokens_out=existing_tokens_out,
                cost_usd=existing_cost,
            )

        text_path = Path(str(row.get("text_path") or ""))
        try:
            transcript_text = text_path.read_text(encoding="utf-8")
        except OSError as exc:
            return failed(f"Cannot read transcript file {text_path}: {exc}")

        try:
            result = clean_text(
                client,
                cache,
                context=context,
                text=transcript_text,
                force=force,
            )
        except LLMError as exc:
            return failed(str(exc))

        cleaned_path = config.cleaned_dir / f"{video_id}.txt"
        try:
            atomic_write_text(cleaned_path, result.text + "\n")
        except OSError as exc:
            return failed(
                f"Cannot write cleaned chapter: {exc}",
                warnings=tuple(result.warnings),
                tokens_in=result.prompt_tokens,
                tokens_out=result.completion_tokens,
                cost_usd=result.cost_usd,
            )

        annotation_text = existing_annotation
        warnings = tuple(result.warnings)
        status = result.status
        tokens_in = result.prompt_tokens
        tokens_out = result.completion_tokens
        cost_usd = result.cost_usd

        if annotations_enabled:
            annotation, annotation_error = _annotation_or_error(
                client,
                context,
                annotation_language_value,
                result.text,
            )
            if annotation is not None:
                annotation_text = annotation.text
                tokens_in += annotation.prompt_tokens
                tokens_out += annotation.completion_tokens
                cost_usd += annotation.cost_usd
                _write_annotation(config, video_id, annotation_text)
            else:
                LOGGER.warning("Annotation failed for %s: %s", video_id, annotation_error)
                _remove_annotation(config, video_id)
                annotation_text = ""
                warnings = _add_warning(warnings, ANNOTATION_FAILED_WARNING)
                status = "needs_review"
        elif force:
            annotation_text = ""
            _remove_annotation(config, video_id)

        return _ChapterWorkResult(
            video_id=video_id,
            status=status,
            cleaned_path=str(cleaned_path),
            annotation=annotation_text,
            warnings=warnings,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
        )

    counts = {"cleaned": 0, "needs_review": 0, "failed": 0}
    annotation_count = 0

    with LLMClient(llm_config, api_key=api_key) as client:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(process_one, row): row for row in candidates}
            for future in as_completed(futures):
                row = futures[future]
                video_id = str(row["video_id"])
                try:
                    result = future.result()
                except Exception as exc:
                    LOGGER.exception("Unexpected cleaning failure for %s", video_id)
                    result = _ChapterWorkResult(
                        video_id=video_id,
                        status="failed",
                        cleaned_path=None,
                        annotation=str(row.get("existing_annotation") or ""),
                        warnings=_warnings_from_db(row.get("existing_warnings")),
                        tokens_in=_as_int(row.get("existing_tokens_in")),
                        tokens_out=_as_int(row.get("existing_tokens_out")),
                        cost_usd=_as_float(row.get("existing_cost_usd")),
                        error=f"Unexpected error: {exc}",
                    )

                counts[result.status] += 1
                if result.annotation:
                    annotation_count += 1
                db.upsert_chapter(
                    video_id=result.video_id,
                    status=result.status,
                    cleaned_path=result.cleaned_path,
                    annotation=result.annotation,
                    error=result.error,
                    warnings=json.dumps(list(result.warnings), ensure_ascii=False),
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    cost_usd=result.cost_usd,
                    model=llm_config.model,
                    updated_at=utc_now_iso(),
                )
                if result.status == "cleaned":
                    LOGGER.info("Chapter cleaned: %s", row["title"])
                elif result.status == "needs_review":
                    LOGGER.warning(
                        "Chapter needs review: %s (%s)",
                        row["title"],
                        "; ".join(result.warnings),
                    )
                else:
                    LOGGER.error("Chapter failed: %s (%s)", row["title"], result.error)

    return CleanSummary(
        selected=len(candidates),
        cleaned=counts["cleaned"],
        needs_review=counts["needs_review"],
        failed=counts["failed"],
        annotations=annotation_count,
    )


def _copy_transcript_without_llm(
    config: Config,
    row: dict[str, Any],
    *,
    force: bool,
) -> _ChapterWorkResult:
    """Copy one normalized transcript, reporting specific failures."""

    video_id = str(row["video_id"])
    text_path = Path(str(row.get("text_path") or ""))
    try:
        transcript_text = text_path.read_text(encoding="utf-8")
    except OSError as exc:
        return _failed_chapter_result(row, f"Cannot read transcript file {text_path}: {exc}")

    stripped = transcript_text.strip()
    if not stripped:
        return _failed_chapter_result(row, f"Transcript file is empty: {text_path}")

    cleaned_path = config.cleaned_dir / f"{video_id}.txt"
    try:
        atomic_write_text(cleaned_path, stripped + "\n")
    except OSError as exc:
        return _failed_chapter_result(row, f"Cannot write cleaned chapter: {exc}")

    annotation_text = str(row.get("existing_annotation") or "")
    if force:
        annotation_text = ""
        _remove_annotation(config, video_id)
    return _ChapterWorkResult(
        video_id=video_id,
        status="cleaned",
        cleaned_path=str(cleaned_path),
        annotation=annotation_text,
        warnings=(),
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        model=None,
    )


def _run_clean_without_llm(
    config: Config,
    db: Database,
    candidates: list[dict[str, Any]],
    *,
    force: bool,
) -> CleanSummary:
    """Copy normalized transcripts into cleaned chapters without LLM calls."""

    LOGGER.info("LLM disabled: copying transcripts unchanged, without annotations")

    counts = {"cleaned": 0, "failed": 0}
    annotation_count = 0

    for row in candidates:
        try:
            result = _copy_transcript_without_llm(config, row, force=force)
        except Exception as exc:
            LOGGER.exception("Unexpected cleaning failure for %s", row.get("video_id"))
            result = _failed_chapter_result(row, f"Unexpected error: {exc}")

        counts[result.status] += 1
        if result.annotation.strip():
            annotation_count += 1
        db.upsert_chapter(
            video_id=result.video_id,
            status=result.status,
            cleaned_path=result.cleaned_path,
            annotation=result.annotation,
            error=result.error,
            warnings=json.dumps(list(result.warnings), ensure_ascii=False),
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=result.cost_usd,
            model=None if result.status == "cleaned" else result.model,
            updated_at=utc_now_iso(),
        )
        if result.status == "cleaned":
            LOGGER.info("Chapter copied without LLM: %s", row["title"])
        else:
            LOGGER.error("Chapter failed: %s (%s)", row["title"], result.error)

    return CleanSummary(
        selected=len(candidates),
        cleaned=counts["cleaned"],
        needs_review=0,
        failed=counts["failed"],
        annotations=annotation_count,
    )


def build_book(
    config: Config,
    db: Database,
    *,
    out: str | Path | None = None,
    title: str | None = None,
    author: str | None = None,
    language: str | None = None,
    chronological: bool = False,
) -> BuildSummary:
    """Assemble and write the EPUB from SQLite without network access."""

    target = Path(out) if out is not None else config.paths.output_dir / "book.epub"
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    book = book_from_database(
        db,
        config,
        title=title,
        author=author,
        language=language,
        chronological=chronological,
    )
    path = build_epub(book, target)
    parts = sum(1 for part in book.parts if part.chapters)
    chapters = sum(len(part.chapters) for part in book.parts if part.chapters)
    return BuildSummary(path=path, parts=parts, chapters=chapters)


def _transcript_is_complete(existing: dict[str, Any] | None, languages: list[str]) -> bool:
    if existing is None or existing.get("status") != "ok":
        return False
    text_path = existing.get("text_path")
    if not text_path or not Path(str(text_path)).is_file():
        return False
    return language_matches(str(existing.get("language") or ""), languages)


def _invalidate_cleaned_chapter(config: Config, db: Database, video_id: str) -> None:
    """Remove stale cleaned output after a transcript was refreshed."""

    deleted_row = db.delete_chapter(video_id)
    removed_files = False
    for path in (
        config.cleaned_dir / f"{video_id}.txt",
        config.cleaned_dir / f"{video_id}.annotation.txt",
    ):
        try:
            existed = path.exists()
            path.unlink(missing_ok=True)
            removed_files = removed_files or existed
        except OSError as exc:
            LOGGER.warning("Cannot remove stale cleaned file %s: %s", path, exc)
    if deleted_row or removed_files:
        LOGGER.info("Invalidated cleaned chapter after transcript refresh: %s", video_id)


def _chapter_is_complete(row: dict[str, Any]) -> bool:
    if row.get("chapter_status") not in {"cleaned", "needs_review"}:
        return False
    cleaned_path = row.get("existing_cleaned_path")
    return bool(cleaned_path) and Path(str(cleaned_path)).is_file()


def _annotation_present(row: dict[str, Any]) -> bool:
    return bool(str(row.get("existing_annotation") or "").strip())


def _annotation_or_error(
    client: LLMClient,
    context: ChapterContext,
    language: str,
    text: str,
) -> tuple[AnnotationResult | None, str | None]:
    try:
        return (
            generate_annotation(
                client,
                context=context,
                language=language,
                text=text,
            ),
            None,
        )
    except Exception as exc:  # annotations are optional by design
        return None, str(exc)


def _warnings_from_db(value: object) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, list | tuple):
        return tuple(str(item).strip() for item in value if str(item).strip())
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return (str(value).strip(),) if str(value).strip() else ()
    if isinstance(parsed, list):
        return tuple(str(item).strip() for item in parsed if str(item).strip())
    return (str(parsed).strip(),) if str(parsed).strip() else ()


def _add_warning(warnings: tuple[str, ...], warning: str) -> tuple[str, ...]:
    return warnings if warning in warnings else (*warnings, warning)


def _as_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _write_annotation(config: Config, video_id: str, annotation: str) -> None:
    if not annotation.strip():
        _remove_annotation(config, video_id)
        return
    try:
        atomic_write_text(
            config.cleaned_dir / f"{video_id}.annotation.txt", annotation.strip() + "\n"
        )
    except OSError as exc:
        LOGGER.warning("Cannot write annotation file for %s: %s", video_id, exc)


def _remove_annotation(config: Config, video_id: str) -> None:
    try:
        (config.cleaned_dir / f"{video_id}.annotation.txt").unlink(missing_ok=True)
    except OSError as exc:
        LOGGER.warning("Cannot remove stale annotation file for %s: %s", video_id, exc)


def _write_snapshots(
    config: Config,
    channel: Channel,
    playlists: list[dict[str, Any]],
    videos: list[Any],
) -> None:
    """Write sanitized YouTube API snapshots; never include the API key."""

    try:
        atomic_write_json(config.api_dir / "channel.json", asdict(channel))
        atomic_write_json(config.api_dir / "playlists.json", playlists)
        atomic_write_json(
            config.api_dir / "videos.json",
            [asdict(video) for video in videos],
        )
    except OSError as exc:
        LOGGER.warning("Could not write API debug snapshots: %s", exc)
