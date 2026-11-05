"""Offline database-to-EPUB pipeline tests, including playlist grouping."""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

import pytest

from ybook.config import BookConfig, Config, LLMConfig, PathsConfig, TranscriptsConfig
from ybook.db import Database
from ybook.epub import book_from_database
from ybook.errors import BookBuildError, LLMError
from ybook.llm import AnnotationResult, CleaningResult
from ybook.pipeline import (
    _transcript_is_complete,
    build_book,
    run_clean,
    run_transcripts,
)
from ybook.transcripts import TranscriptResult


def _config(tmp_path: Path, **book_overrides) -> Config:
    book_values = {
        "title": "Database Book",
        "author": "Database Author",
        "include_live": False,
        "include_shorts": True,
        "group_by_playlist": True,
    }
    book_values.update(book_overrides)
    return Config(
        paths=PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "out"),
        book=BookConfig(**book_values),
    )


def _prepare_db(tmp_path: Path, config: Config) -> Database:
    db = Database(config.db_path)
    db.upsert_channel(
        channel_id="UCbook",
        title="Channel",
        description="Channel description",
        url="https://www.youtube.com/channel/UCbook",
    )
    db.upsert_playlist(
        playlist_id="small",
        channel_id="UCbook",
        title="Small series",
        item_count=2,
        position=0,
    )
    db.upsert_playlist(
        playlist_id="big",
        channel_id="UCbook",
        title="Everything",
        item_count=100,
        position=1,
    )
    db.upsert_playlist(
        playlist_id="empty",
        channel_id="UCbook",
        title="Empty",
        item_count=0,
        position=2,
    )
    db.upsert_playlist(
        playlist_id="uploads",
        channel_id="UCbook",
        title="Uploads",
        item_count=5,
        position=-1,
        kind="uploads",
    )
    db.replace_playlist_items("small", [("v1", 0)])
    db.replace_playlist_items("big", [("v1", 0), ("v2", 1)])
    db.replace_playlist_items("uploads", [("v1", 0), ("v2", 1), ("v3", 2), ("v4", 3), ("v5", 4)])

    videos = [
        ("v1", 120, "2024-01-01T00:00:00Z", "none"),
        ("v2", 600, "2024-01-02T00:00:00Z", "none"),
        ("v3", 600, "2024-01-03T00:00:00Z", "none"),
        ("v4", 600, "2024-01-04T00:00:00Z", "live"),
        ("v5", 600, "2024-01-05T00:00:00Z", "none"),
    ]
    cleaned_dir = config.cleaned_dir
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    for video_id, duration, published, live in videos:
        db.upsert_video(
            video_id=video_id,
            channel_id="UCbook",
            title=f"Video {video_id}",
            published_at=published,
            duration_seconds=duration,
            live_status=live,
        )
        cleaned_path = cleaned_dir / f"{video_id}.txt"
        if video_id != "v5":
            cleaned_path.write_text(f"Text for {video_id}.", encoding="utf-8")
        db.upsert_chapter(
            video_id=video_id,
            status="cleaned",
            cleaned_path=str(cleaned_path),
            annotation=f"Annotation for {video_id}.",
        )
    return db


def test_book_groups_by_most_specific_playlist(tmp_path: Path) -> None:
    config = _config(tmp_path)
    db = _prepare_db(tmp_path, config)
    try:
        book = book_from_database(db, config)
        assert [part.title for part in book.parts] == [
            "Small series",
            "Everything",
            "Other videos",
        ]
        assert [chapter.video_id for chapter in book.parts[0].chapters] == ["v1"]
        assert [chapter.video_id for chapter in book.parts[1].chapters] == ["v2"]
        assert [chapter.video_id for chapter in book.parts[2].chapters] == ["v3"]
        assert all("v4" != chapter.video_id for part in book.parts for chapter in part.chapters)
        assert all("v5" != chapter.video_id for part in book.parts for chapter in part.chapters)
    finally:
        db.close()


def test_short_filter_and_chronological_mode(tmp_path: Path) -> None:
    config = _config(tmp_path, include_shorts=False)
    db = _prepare_db(tmp_path, config)
    try:
        grouped = book_from_database(db, config)
        assert [part.title for part in grouped.parts] == ["Everything", "Other videos"]
        assert [chapter.video_id for part in grouped.parts for chapter in part.chapters] == [
            "v2",
            "v3",
        ]

        chronological = book_from_database(db, config, chronological=True)
        assert [part.title for part in chronological.parts] == ["All chapters"]
        assert [chapter.video_id for part in chronological.parts for chapter in part.chapters] == [
            "v2",
            "v3",
        ]
    finally:
        db.close()


def test_build_book_writes_epub_offline(tmp_path: Path) -> None:
    config = _config(tmp_path)
    db = _prepare_db(tmp_path, config)
    try:
        summary = build_book(config, db, out=tmp_path / "result.epub")
    finally:
        db.close()

    assert summary.path == (tmp_path / "result.epub").resolve()
    assert summary.parts == 3
    assert summary.chapters == 3
    assert summary.path.is_file()
    with zipfile.ZipFile(summary.path) as archive:
        names = archive.namelist()
        assert "OEBPS/content.opf" in names
        nav = archive.read("OEBPS/nav.xhtml").decode("utf-8")
        assert "Small series" in nav
        assert "Everything" in nav
        assert "Other videos" in nav
        assert "Annotation for v1." in archive.read("OEBPS/text/chapter-001.xhtml").decode("utf-8")


def test_transcript_completeness_depends_on_language(tmp_path: Path) -> None:
    text_path = tmp_path / "transcript.txt"
    text_path.write_text("text", encoding="utf-8")

    def record(language: str | None, status: str = "ok") -> dict:
        return {"status": status, "text_path": str(text_path), "language": language}

    assert _transcript_is_complete(record("en"), ["en"]) is True
    assert _transcript_is_complete(record("EN-us"), ["en"]) is True
    assert _transcript_is_complete(record("en-orig"), ["en"]) is True
    assert _transcript_is_complete(record("en"), ["ru"]) is False
    assert _transcript_is_complete(record(""), ["en"]) is False
    assert _transcript_is_complete(record(None), ["en"]) is False
    assert _transcript_is_complete(record("en", status="skipped"), ["en"]) is False
    assert _transcript_is_complete(record("en"), ["ru", "en"]) is True


def test_run_transcripts_refetches_when_requested_language_changes(
    tmp_path: Path, monkeypatch
) -> None:
    config = Config(
        paths=PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "out"),
        transcripts=TranscriptsConfig(workers=1, sleep_requests=0.0),
    )
    config.ensure_directories()
    db = Database(config.db_path)
    text_path = config.transcripts_dir / "vid1.txt"
    text_path.write_text("English text", encoding="utf-8")
    cleaned_path = config.cleaned_dir / "vid1.txt"
    cleaned_path.write_text("English cleaned text.", encoding="utf-8")
    annotation_path = config.cleaned_dir / "vid1.annotation.txt"
    annotation_path.write_text("English annotation.", encoding="utf-8")
    db.upsert_video(
        video_id="vid1",
        channel_id="UC1",
        title="Video 1",
        published_at="2024-01-01T00:00:00Z",
    )
    db.upsert_transcript(
        video_id="vid1",
        status="ok",
        language="en",
        source="test",
        text_path=str(text_path),
    )
    db.upsert_chapter(
        video_id="vid1",
        status="cleaned",
        cleaned_path=str(cleaned_path),
        annotation="English annotation.",
    )

    calls: list[list[str]] = []

    class FakeFetcher:
        def __init__(self, *args, **kwargs):
            pass

        def fetch(self, video_id: str, *, languages: list[str]) -> TranscriptResult:
            calls.append(languages)
            text_path.write_text(f"Text in {languages[0]}", encoding="utf-8")
            return TranscriptResult(
                video_id=video_id,
                status="ok",
                language=languages[0],
                source="fake",
                text_path=str(text_path),
            )

    monkeypatch.setattr("ybook.pipeline.TranscriptFetcher", FakeFetcher)
    try:
        english = run_transcripts(config, db, language="en")
        assert english.selected == 0
        assert english.already_done == 1
        assert calls == []
        assert db.get_chapter("vid1") is not None
        assert cleaned_path.is_file()
        assert annotation_path.is_file()

        russian = run_transcripts(config, db, language="ru")
        assert russian.selected == 1
        assert russian.ok == 1
        assert calls == [["ru"]]
        assert db.get_transcript("vid1")["language"] == "ru"
        assert db.get_chapter("vid1") is None
        assert not cleaned_path.exists()
        assert not annotation_path.exists()

        repeated = run_transcripts(config, db, language="ru")
        assert repeated.selected == 0
        assert repeated.already_done == 1
        assert calls == [["ru"]]
    finally:
        db.close()


def test_force_no_annotations_clears_stale_annotation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = Config(
        paths=PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "out"),
    )
    config.ensure_directories()
    db = Database(config.db_path)
    transcript_path = config.transcripts_dir / "vid1.txt"
    transcript_path.write_text("Raw transcript.", encoding="utf-8")
    cleaned_path = config.cleaned_dir / "vid1.txt"
    cleaned_path.write_text("Old cleaned text.", encoding="utf-8")
    annotation_path = config.cleaned_dir / "vid1.annotation.txt"
    annotation_path.write_text("Old annotation.", encoding="utf-8")

    db.upsert_video(
        video_id="vid1",
        channel_id="UC1",
        title="Video 1",
        published_at="2024-01-01T00:00:00Z",
    )
    db.upsert_transcript(
        video_id="vid1",
        status="ok",
        language="en",
        source="test",
        text_path=str(transcript_path),
    )
    db.upsert_chapter(
        video_id="vid1",
        status="cleaned",
        cleaned_path=str(cleaned_path),
        annotation="Old annotation.",
    )

    def fake_clean_text(*args, **kwargs) -> CleaningResult:
        return CleaningResult(
            text="Fresh cleaned text.",
            status="cleaned",
            warnings=(),
            prompt_tokens=1,
            completion_tokens=1,
            cost_usd=0.0,
            cache_hits=0,
        )

    def fail_annotation(*args, **kwargs):
        raise AssertionError("annotation must not be requested with --no-annotations")

    monkeypatch.setattr("ybook.pipeline.clean_text", fake_clean_text)
    monkeypatch.setattr("ybook.pipeline.generate_annotation", fail_annotation)
    try:
        summary = run_clean(config, db, force=True, annotations=False)

        assert summary.selected == 1
        assert summary.cleaned == 1
        assert summary.annotations == 0
        chapter = db.get_chapter("vid1")
        assert chapter["annotation"] == ""
        assert annotation_path.exists() is False
        assert cleaned_path.read_text(encoding="utf-8").strip() == "Fresh cleaned text."

        db.upsert_chapter(
            video_id="vid1",
            status="cleaned",
            cleaned_path=str(cleaned_path),
            annotation="Kept annotation.",
        )
        annotation_path.write_text("Kept annotation.", encoding="utf-8")

        untouched = run_clean(config, db, force=False, annotations=False)
        assert untouched.selected == 0
        assert db.get_chapter("vid1")["annotation"] == "Kept annotation."
        assert annotation_path.read_text(encoding="utf-8") == "Kept annotation."
    finally:
        db.close()


def _prepare_annotated_chapter(
    tmp_path: Path,
    monkeypatch,
    *,
    annotation: str = "Old annotation.",
    status: str = "cleaned",
    warnings: str = "[]",
    tokens_in: int = 7,
    tokens_out: int = 3,
    cost_usd: float = 0.01,
) -> tuple[Config, Database, Path, Path]:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = Config(
        paths=PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "out"),
    )
    config.ensure_directories()
    db = Database(config.db_path)
    transcript_path = config.transcripts_dir / "vid1.txt"
    transcript_path.write_text("Raw transcript.", encoding="utf-8")
    cleaned_path = config.cleaned_dir / "vid1.txt"
    cleaned_path.write_text("Existing cleaned chapter text.", encoding="utf-8")
    annotation_path = config.cleaned_dir / "vid1.annotation.txt"
    if annotation:
        annotation_path.write_text(annotation, encoding="utf-8")
    db.upsert_video(
        video_id="vid1",
        channel_id="UC1",
        title="Video 1",
        published_at="2024-01-01T00:00:00Z",
    )
    db.upsert_transcript(
        video_id="vid1",
        status="ok",
        language="en",
        source="test",
        text_path=str(transcript_path),
    )
    db.upsert_chapter(
        video_id="vid1",
        status=status,
        cleaned_path=str(cleaned_path),
        annotation=annotation,
        warnings=warnings,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        model="old-model",
    )
    return config, db, cleaned_path, annotation_path


def test_missing_annotation_backfills_without_recleaning(tmp_path: Path, monkeypatch) -> None:
    config, db, cleaned_path, annotation_path = _prepare_annotated_chapter(
        tmp_path,
        monkeypatch,
        annotation="",
        status="needs_review",
        warnings=json.dumps(["word-count ratio"]),
        tokens_in=7,
        tokens_out=3,
        cost_usd=0.01,
    )

    def fail_clean(*args, **kwargs):
        raise AssertionError("clean_text must not run for annotation-only backfill")

    def fake_annotation(*args, **kwargs) -> AnnotationResult:
        return AnnotationResult(
            text="Backfilled annotation. Second sentence.",
            prompt_tokens=10,
            completion_tokens=4,
            cost_usd=0.002,
        )

    monkeypatch.setattr("ybook.pipeline.clean_text", fail_clean)
    monkeypatch.setattr("ybook.pipeline.generate_annotation", fake_annotation)
    try:
        summary = run_clean(config, db, annotations=True)
        chapter = db.get_chapter("vid1")

        assert summary.selected == 1
        assert summary.cleaned == 0
        assert summary.needs_review == 1
        assert summary.annotations == 1
        assert chapter["status"] == "needs_review"
        assert chapter["annotation"] == "Backfilled annotation. Second sentence."
        assert chapter["tokens_in"] == 17
        assert chapter["tokens_out"] == 7
        assert chapter["cost_usd"] == pytest.approx(0.012)
        assert json.loads(chapter["warnings"]) == ["word-count ratio"]
        assert cleaned_path.read_text(encoding="utf-8") == "Existing cleaned chapter text."
        assert annotation_path.read_text(encoding="utf-8").strip() == (
            "Backfilled annotation. Second sentence."
        )
    finally:
        db.close()


def test_annotation_failure_is_retryable_and_chapter_stays_in_book(
    tmp_path: Path, monkeypatch
) -> None:
    config, db, _, annotation_path = _prepare_annotated_chapter(
        tmp_path,
        monkeypatch,
        annotation="",
        status="cleaned",
    )

    def fail_clean(*args, **kwargs):
        raise AssertionError("clean_text must not run for annotation-only backfill")

    def failing_annotation(*args, **kwargs):
        raise LLMError("annotation service unavailable")

    monkeypatch.setattr("ybook.pipeline.clean_text", fail_clean)
    monkeypatch.setattr("ybook.pipeline.generate_annotation", failing_annotation)
    try:
        first = run_clean(config, db, annotations=True)
        chapter = db.get_chapter("vid1")

        assert first.selected == 1
        assert first.needs_review == 1
        assert chapter["status"] == "needs_review"
        assert chapter["annotation"] == ""
        assert "annotation failed" in chapter["warnings"]
        assert annotation_path.exists() is False
        assert any(row["video_id"] == "vid1" for row in db.list_ready_chapters())

        def successful_annotation(*args, **kwargs) -> AnnotationResult:
            return AnnotationResult(
                text="Recovered annotation. It has two sentences.",
                prompt_tokens=5,
                completion_tokens=3,
                cost_usd=0.001,
            )

        monkeypatch.setattr("ybook.pipeline.generate_annotation", successful_annotation)
        second = run_clean(config, db, annotations=True)
        recovered = db.get_chapter("vid1")

        assert second.selected == 1
        assert recovered["annotation"] == ("Recovered annotation. It has two sentences.")
        assert recovered["status"] == "cleaned"
        assert json.loads(recovered["warnings"]) == []
    finally:
        db.close()


def test_annotation_backfill_keeps_other_warnings_and_needs_review(
    tmp_path: Path, monkeypatch
) -> None:
    config, db, _, annotation_path = _prepare_annotated_chapter(
        tmp_path,
        monkeypatch,
        annotation="",
        status="needs_review",
        warnings=json.dumps(["annotation failed", "word-count ratio"]),
    )

    def fail_clean(*args, **kwargs):
        raise AssertionError("clean_text must not run for annotation-only backfill")

    def successful_annotation(*args, **kwargs) -> AnnotationResult:
        return AnnotationResult(
            text="Backfilled annotation. It has two sentences.",
            prompt_tokens=5,
            completion_tokens=3,
            cost_usd=0.001,
        )

    monkeypatch.setattr("ybook.pipeline.clean_text", fail_clean)
    monkeypatch.setattr("ybook.pipeline.generate_annotation", successful_annotation)
    try:
        summary = run_clean(config, db, annotations=True)
        chapter = db.get_chapter("vid1")

        assert summary.selected == 1
        assert chapter["annotation"] == "Backfilled annotation. It has two sentences."
        assert chapter["status"] == "needs_review"
        assert json.loads(chapter["warnings"]) == ["word-count ratio"]
        assert annotation_path.read_text(encoding="utf-8").strip() == (
            "Backfilled annotation. It has two sentences."
        )
    finally:
        db.close()


def test_cleaning_failure_preserves_previous_annotation(tmp_path: Path, monkeypatch) -> None:
    config, db, _, annotation_path = _prepare_annotated_chapter(
        tmp_path,
        monkeypatch,
        annotation="Old annotation.",
        status="cleaned",
    )

    def failing_clean(*args, **kwargs):
        raise LLMError("cleaning service unavailable")

    def fail_annotation(*args, **kwargs):
        raise AssertionError("annotations disabled; generation must not run")

    monkeypatch.setattr("ybook.pipeline.clean_text", failing_clean)
    monkeypatch.setattr("ybook.pipeline.generate_annotation", fail_annotation)
    try:
        summary = run_clean(config, db, force=True, annotations=False)
        chapter = db.get_chapter("vid1")

        assert summary.failed == 1
        assert chapter["status"] == "failed"
        assert chapter["annotation"] == "Old annotation."
        assert annotation_path.read_text(encoding="utf-8") == "Old annotation."
    finally:
        db.close()


def test_wrong_language_invalidation_respects_limit(tmp_path: Path, monkeypatch) -> None:
    config = Config(
        paths=PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "out"),
        transcripts=TranscriptsConfig(workers=1, sleep_requests=0.0),
    )
    config.ensure_directories()
    db = Database(config.db_path)
    files: dict[str, tuple[Path, Path]] = {}
    for index, video_id in enumerate(("vid1", "vid2"), start=1):
        text_path = config.transcripts_dir / f"{video_id}.txt"
        text_path.write_text(f"English text {index}", encoding="utf-8")
        cleaned_path = config.cleaned_dir / f"{video_id}.txt"
        cleaned_path.write_text(f"English cleaned {index}.", encoding="utf-8")
        annotation_path = config.cleaned_dir / f"{video_id}.annotation.txt"
        annotation_path.write_text(f"English annotation {index}.", encoding="utf-8")
        files[video_id] = (cleaned_path, annotation_path)
        db.upsert_video(
            video_id=video_id,
            channel_id="UC1",
            title=f"Video {index}",
            published_at=f"2024-01-0{index}T00:00:00Z",
        )
        db.upsert_transcript(
            video_id=video_id,
            status="ok",
            language="en",
            source="test",
            text_path=str(text_path),
        )
        db.upsert_chapter(
            video_id=video_id,
            status="cleaned",
            cleaned_path=str(cleaned_path),
            annotation=f"English annotation {index}.",
        )

    calls: list[str] = []

    class FakeFetcher:
        def __init__(self, *args, **kwargs):
            pass

        def fetch(self, video_id: str, *, languages: list[str]) -> TranscriptResult:
            calls.append(video_id)
            return TranscriptResult(
                video_id=video_id,
                status="skipped",
                error="No Russian transcript",
            )

    monkeypatch.setattr("ybook.pipeline.TranscriptFetcher", FakeFetcher)
    try:
        summary = run_transcripts(config, db, language="ru", limit=1)

        assert summary.selected == 1
        assert summary.skipped == 1
        assert calls == ["vid1"]
        assert db.get_chapter("vid1") is None
        assert not files["vid1"][0].exists()
        assert not files["vid1"][1].exists()
        assert db.get_chapter("vid2") is not None
        assert files["vid2"][0].is_file()
        assert files["vid2"][1].is_file()
        assert db.get_transcript("vid2")["language"] == "en"
    finally:
        db.close()


def test_wrong_language_chapter_is_removed_even_if_refetch_skips(
    tmp_path: Path, monkeypatch
) -> None:
    config = Config(
        paths=PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "out"),
        transcripts=TranscriptsConfig(workers=1, sleep_requests=0.0),
        book=BookConfig(title="Test Book", author="Test Author"),
    )
    config.ensure_directories()
    db = Database(config.db_path)
    text_path = config.transcripts_dir / "vid1.txt"
    text_path.write_text("English text", encoding="utf-8")
    cleaned_path = config.cleaned_dir / "vid1.txt"
    cleaned_path.write_text("English cleaned text.", encoding="utf-8")
    annotation_path = config.cleaned_dir / "vid1.annotation.txt"
    annotation_path.write_text("English annotation.", encoding="utf-8")
    db.upsert_channel(channel_id="UC1", title="Channel")
    db.upsert_video(
        video_id="vid1",
        channel_id="UC1",
        title="Video 1",
        published_at="2024-01-01T00:00:00Z",
    )
    db.upsert_transcript(
        video_id="vid1",
        status="ok",
        language="en",
        source="test",
        text_path=str(text_path),
    )
    db.upsert_chapter(
        video_id="vid1",
        status="cleaned",
        cleaned_path=str(cleaned_path),
        annotation="English annotation.",
    )

    class FakeFetcher:
        def __init__(self, *args, **kwargs):
            pass

        def fetch(self, video_id: str, *, languages: list[str]) -> TranscriptResult:
            return TranscriptResult(
                video_id=video_id,
                status="skipped",
                error="No Russian transcript",
            )

    monkeypatch.setattr("ybook.pipeline.TranscriptFetcher", FakeFetcher)
    try:
        summary = run_transcripts(config, db, language="ru")
        assert summary.selected == 1
        assert summary.skipped == 1
        assert db.get_chapter("vid1") is None
        assert not cleaned_path.exists()
        assert not annotation_path.exists()
        assert db.list_ready_chapters() == []
        with pytest.raises(BookBuildError, match="No cleaned chapters"):
            book_from_database(db, config)
    finally:
        db.close()


def _prepare_no_llm_chapter(
    tmp_path: Path,
    *,
    transcript_text: str | None = "Raw transcript text.\n",
    cleaned_text: str | None = None,
    annotation: str = "",
    status: str = "cleaned",
    warnings: str = "[]",
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
) -> tuple[Config, Database, Path, Path, Path]:
    config = Config(
        paths=PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "out"),
    )
    config.ensure_directories()
    db = Database(config.db_path)
    transcript_path = config.transcripts_dir / "vid1.txt"
    if transcript_text is not None:
        transcript_path.write_text(transcript_text, encoding="utf-8")
    cleaned_path = config.cleaned_dir / "vid1.txt"
    if cleaned_text is not None:
        cleaned_path.write_text(cleaned_text, encoding="utf-8")
    annotation_path = config.cleaned_dir / "vid1.annotation.txt"
    if annotation:
        annotation_path.write_text(annotation, encoding="utf-8")

    db.upsert_video(
        video_id="vid1",
        channel_id="UC1",
        title="Video 1",
        published_at="2024-01-01T00:00:00Z",
    )
    db.upsert_transcript(
        video_id="vid1",
        status="ok",
        language="en",
        source="test",
        text_path=str(transcript_path),
    )
    db.upsert_chapter(
        video_id="vid1",
        status=status,
        cleaned_path=str(cleaned_path) if cleaned_text is not None else None,
        annotation=annotation,
        warnings=warnings,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        model="old-model",
    )
    return config, db, transcript_path, cleaned_path, annotation_path


def _forbid_llm_calls(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("LLM must not be used when LLM is disabled")

    class FailingLLMClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("LLMClient must not be constructed when LLM is disabled")

    monkeypatch.setattr("ybook.pipeline.clean_text", fail)
    monkeypatch.setattr("ybook.pipeline.generate_annotation", fail)
    monkeypatch.setattr("ybook.pipeline.LLMClient", FailingLLMClient)


def test_run_clean_without_llm_copies_transcripts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("YBOOK_LLM_API_KEY", raising=False)
    config, db, transcript_path, cleaned_path, annotation_path = _prepare_no_llm_chapter(tmp_path)
    _forbid_llm_calls(monkeypatch)
    try:
        summary = run_clean(config, db, llm_enabled=False)
        chapter = db.get_chapter("vid1")

        assert summary.selected == 1
        assert summary.cleaned == 1
        assert summary.needs_review == 0
        assert summary.failed == 0
        assert summary.annotations == 0
        assert chapter["status"] == "cleaned"
        assert chapter["annotation"] == ""
        assert chapter["tokens_in"] == 0
        assert chapter["tokens_out"] == 0
        assert chapter["cost_usd"] == 0.0
        assert chapter["model"] is None
        assert cleaned_path.read_bytes() == transcript_path.read_bytes()
        assert annotation_path.exists() is False

        repeated = run_clean(config, db, llm_enabled=False)
        assert repeated.selected == 0
        assert repeated.cleaned == 0
    finally:
        db.close()


def test_run_clean_without_llm_ignores_annotations_flag(tmp_path: Path, monkeypatch) -> None:
    config, db, _, _, annotation_path = _prepare_no_llm_chapter(tmp_path)
    _forbid_llm_calls(monkeypatch)
    try:
        summary = run_clean(config, db, llm_enabled=False, annotations=True)

        assert summary.selected == 1
        assert summary.cleaned == 1
        assert summary.annotations == 0
        assert db.get_chapter("vid1")["annotation"] == ""
        assert annotation_path.exists() is False
    finally:
        db.close()


def test_run_clean_uses_config_to_disable_llm(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("YBOOK_LLM_API_KEY", raising=False)
    config, db, _, cleaned_path, _ = _prepare_no_llm_chapter(tmp_path)
    config = config.model_copy(update={"llm": LLMConfig(enabled=False)})
    _forbid_llm_calls(monkeypatch)
    try:
        summary = run_clean(config, db)

        assert summary.selected == 1
        assert summary.cleaned == 1
        assert cleaned_path.read_text(encoding="utf-8").strip() == "Raw transcript text."
        assert db.get_chapter("vid1")["model"] is None
    finally:
        db.close()


def test_force_without_llm_recopies_transcript_and_clears_annotation(
    tmp_path: Path, monkeypatch
) -> None:
    config, db, _, cleaned_path, annotation_path = _prepare_no_llm_chapter(
        tmp_path,
        transcript_text="Raw transcript text.",
        cleaned_text="Old cleaned text.",
        annotation="Old annotation.",
        tokens_in=7,
        tokens_out=3,
        cost_usd=0.01,
    )
    _forbid_llm_calls(monkeypatch)
    try:
        summary = run_clean(config, db, force=True, llm_enabled=False)
        chapter = db.get_chapter("vid1")

        assert summary.selected == 1
        assert summary.cleaned == 1
        assert cleaned_path.read_text(encoding="utf-8").strip() == "Raw transcript text."
        assert annotation_path.exists() is False
        assert chapter["annotation"] == ""
        assert chapter["tokens_in"] == 0
        assert chapter["tokens_out"] == 0
        assert chapter["cost_usd"] == 0.0
        assert chapter["model"] is None
    finally:
        db.close()


def test_run_clean_without_llm_failure_preserves_annotation(tmp_path: Path, monkeypatch) -> None:
    config, db, transcript_path, cleaned_path, annotation_path = _prepare_no_llm_chapter(
        tmp_path,
        transcript_text=None,
        annotation="Old annotation.",
        warnings=json.dumps(["word-count ratio"]),
        tokens_in=7,
        tokens_out=3,
        cost_usd=0.01,
    )
    _forbid_llm_calls(monkeypatch)
    assert transcript_path.exists() is False
    try:
        summary = run_clean(config, db, llm_enabled=False)
        chapter = db.get_chapter("vid1")

        assert summary.selected == 1
        assert summary.cleaned == 0
        assert summary.failed == 1
        assert summary.annotations == 1
        assert chapter["status"] == "failed"
        assert chapter["annotation"] == "Old annotation."
        assert json.loads(chapter["warnings"]) == ["word-count ratio"]
        assert chapter["tokens_in"] == 7
        assert chapter["tokens_out"] == 3
        assert chapter["cost_usd"] == pytest.approx(0.01)
        assert chapter["model"] == "old-model"
        assert annotation_path.read_text(encoding="utf-8") == "Old annotation."
        assert cleaned_path.exists() is False
    finally:
        db.close()


def test_run_clean_without_llm_empty_transcript_fails(tmp_path: Path, monkeypatch) -> None:
    config, db, _, cleaned_path, annotation_path = _prepare_no_llm_chapter(
        tmp_path,
        transcript_text="  \n\t",
        annotation="Old annotation.",
        warnings=json.dumps(["word-count ratio"]),
        tokens_in=7,
        tokens_out=3,
        cost_usd=0.01,
    )
    _forbid_llm_calls(monkeypatch)
    try:
        summary = run_clean(config, db, llm_enabled=False)
        chapter = db.get_chapter("vid1")

        assert summary.selected == 1
        assert summary.failed == 1
        assert summary.cleaned == 0
        assert chapter["status"] == "failed"
        assert "empty" in chapter["error"].lower()
        assert chapter["annotation"] == "Old annotation."
        assert json.loads(chapter["warnings"]) == ["word-count ratio"]
        assert chapter["tokens_in"] == 7
        assert chapter["tokens_out"] == 3
        assert chapter["cost_usd"] == pytest.approx(0.01)
        assert chapter["model"] == "old-model"
        assert annotation_path.read_text(encoding="utf-8") == "Old annotation."
        assert cleaned_path.exists() is False
    finally:
        db.close()


def test_run_clean_without_llm_invalid_utf8_fails_one_chapter(tmp_path: Path, monkeypatch) -> None:
    config, db, transcript_path, cleaned_path, _ = _prepare_no_llm_chapter(
        tmp_path,
        transcript_text=None,
        annotation="Old annotation.",
    )
    transcript_path.write_bytes(b"\xff\xfe")
    _forbid_llm_calls(monkeypatch)
    try:
        summary = run_clean(config, db, llm_enabled=False)
        chapter = db.get_chapter("vid1")

        assert summary.selected == 1
        assert summary.failed == 1
        assert summary.cleaned == 0
        assert chapter["status"] == "failed"
        assert "unexpected error" in chapter["error"].lower()
        assert chapter["model"] == "old-model"
        assert cleaned_path.exists() is False
    finally:
        db.close()


def test_run_clean_without_llm_skips_complete_annotated_chapter(
    tmp_path: Path, monkeypatch
) -> None:
    config, db, _, cleaned_path, annotation_path = _prepare_no_llm_chapter(
        tmp_path,
        transcript_text="Raw transcript text.\n",
        cleaned_text="Existing cleaned chapter.\n",
        annotation="Existing annotation.",
    )
    _forbid_llm_calls(monkeypatch)
    try:
        summary = run_clean(config, db, llm_enabled=False)
        chapter = db.get_chapter("vid1")

        assert summary.selected == 0
        assert summary.cleaned == 0
        assert cleaned_path.read_text(encoding="utf-8") == "Existing cleaned chapter.\n"
        assert annotation_path.read_text(encoding="utf-8") == "Existing annotation."
        assert chapter["annotation"] == "Existing annotation."
    finally:
        db.close()


def test_run_clean_llm_override_uses_llm_when_config_disables_it(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config, db, _, cleaned_path, _ = _prepare_no_llm_chapter(tmp_path)
    config = config.model_copy(update={"llm": LLMConfig(enabled=False, model="override-model")})

    class FakeLLMClient:
        def __init__(self, config, *, api_key):
            self.config = config
            self.api_key = api_key

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    calls: list[str] = []

    def fake_clean_text(client, cache, *, context, text, force) -> CleaningResult:
        calls.append(text)
        return CleaningResult(
            text="Fresh LLM text.",
            status="cleaned",
            warnings=(),
            prompt_tokens=2,
            completion_tokens=3,
            cost_usd=0.01,
            cache_hits=0,
        )

    def fail_annotation(*args, **kwargs):
        raise AssertionError("annotations must not be requested")

    monkeypatch.setattr("ybook.pipeline.LLMClient", FakeLLMClient)
    monkeypatch.setattr("ybook.pipeline.clean_text", fake_clean_text)
    monkeypatch.setattr("ybook.pipeline.generate_annotation", fail_annotation)
    try:
        summary = run_clean(config, db, llm_enabled=True, annotations=False)
        chapter = db.get_chapter("vid1")

        assert summary.selected == 1
        assert summary.cleaned == 1
        assert len(calls) == 1
        assert calls[0].strip() == "Raw transcript text."
        assert chapter["status"] == "cleaned"
        assert chapter["model"] == "override-model"
        assert chapter["tokens_in"] == 2
        assert chapter["cost_usd"] == pytest.approx(0.01)
        assert cleaned_path.read_text(encoding="utf-8") == "Fresh LLM text.\n"
    finally:
        db.close()


def test_force_without_llm_missing_transcript_keeps_cleaned_file_unreferenced(
    tmp_path: Path, monkeypatch
) -> None:
    config, db, _, cleaned_path, annotation_path = _prepare_no_llm_chapter(
        tmp_path,
        transcript_text=None,
        cleaned_text="Previous cleaned chapter.\n",
        annotation="Old annotation.",
        warnings=json.dumps(["word-count ratio"]),
        tokens_in=7,
        tokens_out=3,
        cost_usd=0.01,
    )
    _forbid_llm_calls(monkeypatch)
    try:
        summary = run_clean(config, db, force=True, llm_enabled=False)
        chapter = db.get_chapter("vid1")

        assert summary.selected == 1
        assert summary.failed == 1
        assert chapter["status"] == "failed"
        assert chapter["cleaned_path"] in (None, "")
        assert chapter["annotation"] == "Old annotation."
        assert chapter["model"] == "old-model"
        assert annotation_path.read_text(encoding="utf-8") == "Old annotation."
        assert cleaned_path.read_text(encoding="utf-8") == "Previous cleaned chapter.\n"
        assert db.list_ready_chapters() == []
    finally:
        db.close()


def test_build_filters_ready_chapters_to_most_recent_channel(tmp_path: Path, caplog) -> None:
    config = Config(
        paths=PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "out"),
        book=BookConfig(title="Test Book", author="Test Author"),
    )
    config.ensure_directories()
    db = Database(config.db_path)
    db.upsert_channel(
        channel_id="UCold",
        title="Old channel",
        fetched_at="2024-01-01T00:00:00+00:00",
    )
    db.upsert_channel(
        channel_id="UCnew",
        title="New channel",
        fetched_at="2024-02-01T00:00:00+00:00",
    )
    for video_id, channel_id, title in (
        ("old1", "UCold", "Old video"),
        ("new1", "UCnew", "New video"),
    ):
        cleaned_path = config.cleaned_dir / f"{video_id}.txt"
        cleaned_path.write_text(f"Text for {video_id}.", encoding="utf-8")
        db.upsert_video(
            video_id=video_id,
            channel_id=channel_id,
            title=title,
            published_at="2024-01-01T00:00:00Z",
        )
        db.upsert_chapter(
            video_id=video_id,
            status="cleaned",
            cleaned_path=str(cleaned_path),
            annotation=f"Annotation for {video_id}.",
        )

    caplog.set_level(logging.WARNING, logger="ybook.epub")
    try:
        book = book_from_database(db, config)
        assert book.channel_id == "UCnew"
        assert [chapter.video_id for part in book.parts for chapter in part.chapters] == ["new1"]
        assert "non-current channel" in caplog.text
    finally:
        db.close()
