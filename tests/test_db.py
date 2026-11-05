"""SQLite schema, upserts, statuses, and build-query tests."""

from __future__ import annotations

from pathlib import Path

from ybook.db import Database


def _make_db(tmp_path: Path) -> Database:
    return Database(tmp_path / "state.sqlite")


def test_upserts_are_idempotent(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        db.upsert_channel(
            channel_id="UC1",
            title="Old",
            description="old",
            url="https://youtube.com/channel/UC1",
            uploads_playlist_id="UU1",
        )
        db.upsert_channel(
            channel_id="UC1",
            title="New",
            description="new",
            url="https://youtube.com/channel/UC1",
            uploads_playlist_id="UU1",
        )
        assert db.get_channel()["title"] == "New"

        db.upsert_playlist(
            playlist_id="PL1",
            channel_id="UC1",
            title="One",
            item_count=1,
            position=0,
        )
        db.upsert_playlist(
            playlist_id="PL1",
            channel_id="UC1",
            title="One updated",
            item_count=2,
            position=0,
        )
        assert db.count_playlists() == 1
        assert db.list_playlists()[0]["title"] == "One updated"

        for _ in range(2):
            db.upsert_video(
                video_id="v1",
                channel_id="UC1",
                title="Video",
                published_at="2024-01-01T00:00:00Z",
                duration_seconds=60,
            )
        assert db.count_videos() == 1
        assert db.list_videos()[0]["duration_seconds"] == 60
    finally:
        db.close()


def test_replace_playlist_items_removes_stale_members(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        db.replace_playlist_items("PL1", [("v1", 0), ("v2", 1)])
        assert [item["video_id"] for item in db.list_playlist_items("PL1")] == [
            "v1",
            "v2",
        ]
        db.replace_playlist_items("PL1", [("v2", 0), ("v3", 1)])
        assert [item["video_id"] for item in db.list_playlist_items("PL1")] == [
            "v2",
            "v3",
        ]
    finally:
        db.close()


def test_transcript_and_chapter_status_counts(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        for index, status in enumerate(("ok", "skipped", "failed", "ok"), start=1):
            db.upsert_transcript(
                video_id=f"v{index}",
                status=status,
                language="en",
                is_generated=bool(index % 2),
                source="test",
                raw_path=f"/raw/v{index}",
                text_path=f"/text/v{index}.txt" if status == "ok" else None,
                error=None if status == "ok" else "reason",
            )
        assert db.transcript_counts() == {
            "ok": 2,
            "skipped": 1,
            "failed": 1,
            "total": 4,
        }

        db.upsert_chapter(
            video_id="v1",
            status="cleaned",
            cleaned_path="/cleaned/v1.txt",
            annotation="An annotation.",
            warnings="[]",
            tokens_in=10,
            tokens_out=20,
            cost_usd=0.001,
            model="test",
        )
        db.upsert_chapter(
            video_id="v4",
            status="needs_review",
            cleaned_path="/cleaned/v4.txt",
            annotation="",
            warnings='["ratio"]',
        )
        db.upsert_chapter(
            video_id="v2",
            status="failed",
            cleaned_path=None,
            annotation="ignored",
            error="boom",
        )
        assert db.chapter_counts() == {
            "cleaned": 1,
            "needs_review": 1,
            "failed": 1,
            "total": 3,
        }
        assert db.annotation_count() == 1
    finally:
        db.close()


def test_chapter_and_transcript_upserts_preserve_last_values(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        db.upsert_transcript(video_id="v1", status="failed", error="one")
        db.upsert_transcript(video_id="v1", status="ok", text_path="/t/v1.txt")
        row = db.get_transcript("v1")
        assert row["status"] == "ok"
        assert row["error"] is None
        assert row["text_path"] == "/t/v1.txt"

        db.upsert_chapter(video_id="v1", status="cleaned", cleaned_path="/c/v1.txt")
        db.upsert_chapter(
            video_id="v1",
            status="needs_review",
            cleaned_path="/c/v1.txt",
            warnings='["w"]',
        )
        assert db.get_chapter("v1")["status"] == "needs_review"
    finally:
        db.close()


def test_clean_candidates_only_include_successful_transcripts(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        db.upsert_channel(channel_id="UC1", title="Channel")
        for index, status in enumerate(("ok", "skipped", "failed"), start=1):
            db.upsert_video(
                video_id=f"v{index}",
                channel_id="UC1",
                title=f"Video {index}",
                published_at=f"2024-01-0{index}T00:00:00Z",
            )
            db.upsert_transcript(
                video_id=f"v{index}",
                status=status,
                text_path=f"/t/v{index}.txt" if status == "ok" else None,
            )
        candidates = db.list_clean_candidates()
        assert [row["video_id"] for row in candidates] == ["v1"]

        db.upsert_chapter(
            video_id="v1",
            status="cleaned",
            cleaned_path="/c/v1.txt",
            annotation="A",
        )
        db.upsert_chapter(
            video_id="v2",
            status="failed",
            cleaned_path=None,
        )
        ready = db.list_ready_chapters()
        assert [row["video_id"] for row in ready] == ["v1"]
        assert ready[0]["annotation"] == "A"
    finally:
        db.close()


def test_most_specific_playlist_membership_and_effective_count(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        db.upsert_playlist(
            playlist_id="small",
            channel_id="UC1",
            title="Small series",
            item_count=1,
            position=0,
        )
        db.upsert_playlist(
            playlist_id="big",
            channel_id="UC1",
            title="Everything",
            item_count=100,
            position=1,
        )
        db.upsert_playlist(
            playlist_id="unknown",
            channel_id="UC1",
            title="Unknown count",
            item_count=0,
            position=2,
        )
        db.replace_playlist_items("small", [("v1", 0)])
        db.replace_playlist_items("big", [("v1", 0), ("v2", 1)])
        db.replace_playlist_items("unknown", [("v2", 0)])
        titles = db.primary_playlist_titles()
        assert titles["v1"] == "Small series"
        assert titles["v2"] == "Unknown count"

        memberships = db.playlist_memberships()
        assert any(
            row["playlist_id"] == "unknown" and row["effective_item_count"] == 1
            for row in memberships
        )
    finally:
        db.close()


def test_state_summary(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        db.upsert_channel(channel_id="UC1", title="Channel")
        db.upsert_playlist(playlist_id="PL1", channel_id="UC1", title="P", kind="regular")
        db.upsert_playlist(
            playlist_id="UU1", channel_id="UC1", title="U", kind="uploads", position=-1
        )
        db.upsert_video(video_id="v1", channel_id="UC1", title="V")
        db.upsert_transcript(video_id="v1", status="ok", text_path="/t/v1.txt")
        db.upsert_chapter(
            video_id="v1",
            status="cleaned",
            cleaned_path="/c/v1.txt",
            annotation="A",
        )
        summary = db.state_summary()
        assert summary["channels"] == 1
        assert summary["playlists"] == 2
        assert summary["regular_playlists"] == 1
        assert summary["uploads_playlists"] == 1
        assert summary["videos"] == 1
        assert summary["transcripts_ok"] == 1
        assert summary["chapters_cleaned"] == 1
        assert summary["annotations"] == 1
    finally:
        db.close()


def test_reconcile_channel_removes_only_stale_rows_for_that_channel(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        db.upsert_channel(channel_id="UCkeep", title="Keep")
        db.upsert_channel(channel_id="UCother", title="Other")
        db.upsert_playlist(playlist_id="PLkeep", channel_id="UCkeep", title="Keep", position=0)
        db.upsert_playlist(playlist_id="PLstale", channel_id="UCkeep", title="Stale", position=1)
        db.upsert_playlist(playlist_id="PLother", channel_id="UCother", title="Other", position=0)
        db.upsert_video(video_id="vkeep", channel_id="UCkeep", title="Keep")
        db.upsert_video(video_id="vstale", channel_id="UCkeep", title="Stale")
        db.upsert_video(video_id="vother", channel_id="UCother", title="Other")
        db.replace_playlist_items("PLstale", [("vstale", 0)])
        db.replace_playlist_items("PLother", [("vother", 0)])
        db.upsert_transcript(video_id="vkeep", status="ok", text_path="/t/keep.txt")
        db.upsert_transcript(video_id="vstale", status="ok", text_path="/t/stale.txt")
        db.upsert_transcript(video_id="vother", status="ok", text_path="/t/other.txt")
        db.upsert_chapter(video_id="vkeep", status="cleaned", cleaned_path="/c/keep.txt")
        db.upsert_chapter(video_id="vstale", status="cleaned", cleaned_path="/c/stale.txt")
        db.upsert_chapter(video_id="vother", status="cleaned", cleaned_path="/c/other.txt")

        removed = db.reconcile_channel(
            "UCkeep",
            playlist_ids={"PLkeep"},
            video_ids={"vkeep"},
        )

        assert removed == {"playlists": 1, "videos": 1}
        assert {playlist["id"] for playlist in db.list_playlists()} == {
            "PLkeep",
            "PLother",
        }
        assert {video["id"] for video in db.list_videos()} == {"vkeep", "vother"}
        assert db.get_transcript("vstale") is None
        assert db.get_chapter("vstale") is None
        assert db.list_playlist_items("PLstale") == []
        assert db.get_transcript("vother") is not None
        assert db.get_chapter("vother") is not None
    finally:
        db.close()
