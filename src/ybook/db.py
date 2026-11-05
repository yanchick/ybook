"""SQLite state store: idempotent upserts, status queries, and build inputs."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .utils import utc_now_iso

TRANSCRIPT_STATUSES = frozenset({"ok", "skipped", "failed"})
CHAPTER_STATUSES = frozenset({"cleaned", "needs_review", "failed"})
PLAYLIST_KINDS = frozenset({"regular", "uploads"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS channels (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    uploads_playlist_id TEXT,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS playlists (
    id TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    item_count INTEGER NOT NULL DEFAULT 0,
    position INTEGER NOT NULL DEFAULT 0,
    kind TEXT NOT NULL DEFAULT 'regular',
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    published_at TEXT NOT NULL DEFAULT '',
    duration_seconds INTEGER NOT NULL DEFAULT 0,
    default_audio_language TEXT,
    live_status TEXT NOT NULL DEFAULT 'none',
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS playlist_items (
    playlist_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    position INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (playlist_id, video_id)
);

CREATE TABLE IF NOT EXISTS transcripts (
    video_id TEXT PRIMARY KEY,
    language TEXT,
    is_generated INTEGER,
    source TEXT,
    raw_path TEXT,
    text_path TEXT,
    status TEXT NOT NULL,
    error TEXT,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chapters (
    video_id TEXT PRIMARY KEY,
    cleaned_path TEXT,
    annotation TEXT,
    status TEXT NOT NULL,
    error TEXT,
    warnings TEXT,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    model TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel_id);
CREATE INDEX IF NOT EXISTS idx_playlist_items_video ON playlist_items(video_id);
CREATE INDEX IF NOT EXISTS idx_chapters_status ON chapters(status);
CREATE INDEX IF NOT EXISTS idx_transcripts_status ON transcripts(status);
"""


class Database:
    """Thin SQLite wrapper used only from the main thread."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(_SCHEMA)
        self.connection.commit()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the SQLite connection."""

        self.connection.close()

    def upsert_channel(
        self,
        *,
        channel_id: str,
        title: str,
        description: str = "",
        url: str = "",
        uploads_playlist_id: str | None = None,
        fetched_at: str | None = None,
    ) -> None:
        """Insert or update a channel row."""

        timestamp = fetched_at or utc_now_iso()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO channels (id, title, description, url, uploads_playlist_id, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    description = excluded.description,
                    url = excluded.url,
                    uploads_playlist_id = excluded.uploads_playlist_id,
                    fetched_at = excluded.fetched_at
                """,
                (channel_id, title, description, url, uploads_playlist_id, timestamp),
            )

    def upsert_playlist(
        self,
        *,
        playlist_id: str,
        channel_id: str,
        title: str,
        description: str = "",
        item_count: int = 0,
        position: int = 0,
        kind: str = "regular",
        fetched_at: str | None = None,
    ) -> None:
        """Insert or update a playlist row."""

        _require(kind in PLAYLIST_KINDS, f"Unknown playlist kind: {kind}")
        timestamp = fetched_at or utc_now_iso()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO playlists
                    (id, channel_id, title, description, item_count, position, kind, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    channel_id = excluded.channel_id,
                    title = excluded.title,
                    description = excluded.description,
                    item_count = excluded.item_count,
                    position = excluded.position,
                    kind = excluded.kind,
                    fetched_at = excluded.fetched_at
                """,
                (
                    playlist_id,
                    channel_id,
                    title,
                    description,
                    int(item_count),
                    int(position),
                    kind,
                    timestamp,
                ),
            )

    def upsert_video(
        self,
        *,
        video_id: str,
        channel_id: str,
        title: str,
        description: str = "",
        published_at: str = "",
        duration_seconds: int = 0,
        default_audio_language: str | None = None,
        live_status: str = "none",
        fetched_at: str | None = None,
    ) -> None:
        """Insert or update a video row."""

        timestamp = fetched_at or utc_now_iso()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO videos
                    (id, channel_id, title, description, published_at, duration_seconds,
                     default_audio_language, live_status, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    channel_id = excluded.channel_id,
                    title = excluded.title,
                    description = excluded.description,
                    published_at = excluded.published_at,
                    duration_seconds = excluded.duration_seconds,
                    default_audio_language = excluded.default_audio_language,
                    live_status = excluded.live_status,
                    fetched_at = excluded.fetched_at
                """,
                (
                    video_id,
                    channel_id,
                    title,
                    description,
                    published_at,
                    int(duration_seconds),
                    default_audio_language,
                    live_status,
                    timestamp,
                ),
            )

    def replace_playlist_items(
        self,
        playlist_id: str,
        items: Iterable[tuple[str, int]],
    ) -> None:
        """Replace all memberships of a playlist with the supplied ordered items."""

        unique_items = {video_id: int(position) for video_id, position in items}
        with self.connection:
            self.connection.execute(
                "DELETE FROM playlist_items WHERE playlist_id = ?", (playlist_id,)
            )
            self.connection.executemany(
                "INSERT INTO playlist_items (playlist_id, video_id, position) VALUES (?, ?, ?)",
                ((playlist_id, video_id, position) for video_id, position in unique_items.items()),
            )

    def upsert_transcript(
        self,
        *,
        video_id: str,
        status: str,
        language: str | None = None,
        is_generated: bool | int | None = None,
        source: str | None = None,
        raw_path: str | None = None,
        text_path: str | None = None,
        error: str | None = None,
        fetched_at: str | None = None,
    ) -> None:
        """Insert or update transcript state for a video."""

        _require(status in TRANSCRIPT_STATUSES, f"Unknown transcript status: {status}")
        generated_value = int(is_generated) if is_generated is not None else None
        timestamp = fetched_at or utc_now_iso()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO transcripts
                    (video_id, language, is_generated, source, raw_path, text_path,
                     status, error, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    language = excluded.language,
                    is_generated = excluded.is_generated,
                    source = excluded.source,
                    raw_path = excluded.raw_path,
                    text_path = excluded.text_path,
                    status = excluded.status,
                    error = excluded.error,
                    fetched_at = excluded.fetched_at
                """,
                (
                    video_id,
                    language,
                    generated_value,
                    source,
                    raw_path,
                    text_path,
                    status,
                    error,
                    timestamp,
                ),
            )

    def upsert_chapter(
        self,
        *,
        video_id: str,
        status: str,
        cleaned_path: str | None = None,
        annotation: str = "",
        error: str | None = None,
        warnings: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        model: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        """Insert or update chapter state for a video."""

        _require(status in CHAPTER_STATUSES, f"Unknown chapter status: {status}")
        timestamp = updated_at or utc_now_iso()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO chapters
                    (video_id, cleaned_path, annotation, status, error, warnings,
                     tokens_in, tokens_out, cost_usd, model, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    cleaned_path = excluded.cleaned_path,
                    annotation = excluded.annotation,
                    status = excluded.status,
                    error = excluded.error,
                    warnings = excluded.warnings,
                    tokens_in = excluded.tokens_in,
                    tokens_out = excluded.tokens_out,
                    cost_usd = excluded.cost_usd,
                    model = excluded.model,
                    updated_at = excluded.updated_at
                """,
                (
                    video_id,
                    cleaned_path,
                    annotation,
                    status,
                    error,
                    warnings,
                    int(tokens_in),
                    int(tokens_out),
                    float(cost_usd),
                    model,
                    timestamp,
                ),
            )

    def get_channel(self) -> dict[str, Any] | None:
        """Return the most recently fetched channel."""

        row = self.connection.execute(
            "SELECT * FROM channels ORDER BY fetched_at DESC, id LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def get_channel_by_id(self, channel_id: str) -> dict[str, Any] | None:
        """Return one channel by ID."""

        row = self.connection.execute(
            "SELECT * FROM channels WHERE id = ?", (channel_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_channel_ids(self) -> list[str]:
        """Return every channel ID stored in the database."""

        rows = self.connection.execute("SELECT id FROM channels ORDER BY id ASC").fetchall()
        return [str(row["id"]) for row in rows]

    def delete_chapter(self, video_id: str) -> bool:
        """Delete a chapter row. Return ``True`` when a row existed."""

        with self.connection:
            cursor = self.connection.execute("DELETE FROM chapters WHERE video_id = ?", (video_id,))
        return cursor.rowcount > 0

    def reconcile_channel(
        self,
        channel_id: str,
        playlist_ids: Iterable[str],
        video_ids: Iterable[str],
    ) -> dict[str, int]:
        """Remove channel playlists/videos that were not seen in the latest fetch.

        Child rows (playlist items, transcripts, chapters) are removed with
        their parent. Other channels are never touched.
        """

        keep_playlists = {str(item) for item in playlist_ids if item}
        keep_videos = {str(item) for item in video_ids if item}
        deleted_playlists = 0
        deleted_videos = 0

        with self.connection:
            stored_playlists = {
                str(row["id"])
                for row in self.connection.execute(
                    "SELECT id FROM playlists WHERE channel_id = ?", (channel_id,)
                ).fetchall()
            }
            for chunk in _chunks(stored_playlists - keep_playlists):
                placeholders = ",".join("?" for _ in chunk)
                self.connection.execute(
                    f"DELETE FROM playlist_items WHERE playlist_id IN ({placeholders})",
                    chunk,
                )
                cursor = self.connection.execute(
                    f"DELETE FROM playlists WHERE id IN ({placeholders})",
                    chunk,
                )
                deleted_playlists += max(0, cursor.rowcount)

            stored_videos = {
                str(row["id"])
                for row in self.connection.execute(
                    "SELECT id FROM videos WHERE channel_id = ?", (channel_id,)
                ).fetchall()
            }
            for chunk in _chunks(stored_videos - keep_videos):
                placeholders = ",".join("?" for _ in chunk)
                self.connection.execute(
                    f"DELETE FROM chapters WHERE video_id IN ({placeholders})",
                    chunk,
                )
                self.connection.execute(
                    f"DELETE FROM transcripts WHERE video_id IN ({placeholders})",
                    chunk,
                )
                self.connection.execute(
                    f"DELETE FROM playlist_items WHERE video_id IN ({placeholders})",
                    chunk,
                )
                cursor = self.connection.execute(
                    f"DELETE FROM videos WHERE id IN ({placeholders})",
                    chunk,
                )
                deleted_videos += max(0, cursor.rowcount)

        return {"playlists": deleted_playlists, "videos": deleted_videos}

    def get_transcript(self, video_id: str) -> dict[str, Any] | None:
        """Return transcript state for a video."""

        row = self.connection.execute(
            "SELECT * FROM transcripts WHERE video_id = ?", (video_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_chapter(self, video_id: str) -> dict[str, Any] | None:
        """Return chapter state for a video."""

        row = self.connection.execute(
            "SELECT * FROM chapters WHERE video_id = ?", (video_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_videos(self) -> list[dict[str, Any]]:
        """Return all known videos ordered chronologically."""

        rows = self.connection.execute(
            "SELECT * FROM videos ORDER BY published_at ASC, id ASC"
        ).fetchall()
        return [dict(row) for row in rows]

    def list_playlists(self, *, kind: str | None = None) -> list[dict[str, Any]]:
        """Return playlists ordered by channel position."""

        if kind is None:
            rows = self.connection.execute(
                "SELECT * FROM playlists ORDER BY position ASC, id ASC"
            ).fetchall()
        else:
            _require(kind in PLAYLIST_KINDS, f"Unknown playlist kind: {kind}")
            rows = self.connection.execute(
                "SELECT * FROM playlists WHERE kind = ? ORDER BY position ASC, id ASC",
                (kind,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_playlist_items(self, playlist_id: str) -> list[dict[str, Any]]:
        """Return memberships for one playlist ordered by position."""

        rows = self.connection.execute(
            """
            SELECT video_id, position
            FROM playlist_items
            WHERE playlist_id = ?
            ORDER BY position ASC, video_id ASC
            """,
            (playlist_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def playlist_memberships(self) -> list[dict[str, Any]]:
        """Return regular playlist memberships enriched with playlist metadata."""

        rows = self.connection.execute(
            """
            SELECT
                pi.playlist_id,
                pi.video_id,
                pi.position,
                p.title,
                p.description,
                p.item_count,
                p.position AS playlist_position,
                COALESCE(NULLIF(p.item_count, 0), counts.actual_count, 0)
                    AS effective_item_count
            FROM playlist_items pi
            JOIN playlists p ON p.id = pi.playlist_id
            LEFT JOIN (
                SELECT playlist_id, COUNT(*) AS actual_count
                FROM playlist_items
                GROUP BY playlist_id
            ) counts ON counts.playlist_id = p.id
            WHERE p.kind = 'regular'
            ORDER BY p.position ASC, pi.position ASC, pi.video_id ASC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def primary_playlist_titles(self) -> dict[str, str]:
        """Map video ID to its most specific regular playlist title.

        "Most specific" means the playlist with the fewest effective items;
        ties are broken by the playlist's channel position.
        """

        best: dict[str, tuple[int, int, str]] = {}
        for membership in self.playlist_memberships():
            video_id = str(membership["video_id"])
            candidate = (
                int(membership["effective_item_count"]),
                int(membership["playlist_position"]),
                str(membership["title"]),
            )
            current = best.get(video_id)
            if current is None or candidate < current:
                best[video_id] = candidate
        return {video_id: value[2] for video_id, value in best.items()}

    def list_clean_candidates(self) -> list[dict[str, Any]]:
        """Return videos with a successful transcript and their existing chapter state."""

        rows = self.connection.execute(
            """
            SELECT
                v.id AS video_id,
                v.title,
                v.channel_id,
                v.published_at,
                t.text_path,
                t.language AS transcript_language,
                c.status AS chapter_status,
                c.cleaned_path AS existing_cleaned_path,
                c.annotation AS existing_annotation,
                c.warnings AS existing_warnings,
                c.tokens_in AS existing_tokens_in,
                c.tokens_out AS existing_tokens_out,
                c.cost_usd AS existing_cost_usd,
                c.model AS existing_model
            FROM videos v
            JOIN transcripts t ON t.video_id = v.id
            LEFT JOIN chapters c ON c.video_id = v.id
            WHERE t.status = 'ok'
              AND t.text_path IS NOT NULL
              AND t.text_path <> ''
            ORDER BY v.published_at ASC, v.id ASC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def list_ready_chapters(self) -> list[dict[str, Any]]:
        """Return cleaned/needs-review chapters with video metadata for EPUB building."""

        rows = self.connection.execute(
            """
            SELECT
                v.id AS video_id,
                v.title,
                v.channel_id,
                v.published_at,
                v.duration_seconds,
                v.default_audio_language,
                v.live_status,
                c.cleaned_path,
                c.annotation,
                c.status AS chapter_status,
                c.warnings
            FROM chapters c
            JOIN videos v ON v.id = c.video_id
            WHERE c.status IN ('cleaned', 'needs_review')
              AND c.cleaned_path IS NOT NULL
              AND c.cleaned_path <> ''
            ORDER BY v.published_at ASC, v.id ASC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def transcript_counts(self) -> dict[str, int]:
        """Count transcript rows by status, including ``total``."""

        return _count_by_status(self.connection, "transcripts", TRANSCRIPT_STATUSES)

    def chapter_counts(self) -> dict[str, int]:
        """Count chapter rows by status, including ``total``."""

        return _count_by_status(self.connection, "chapters", CHAPTER_STATUSES)

    def annotation_count(self) -> int:
        """Count successful chapters that have a non-empty annotation."""

        row = self.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM chapters
            WHERE status IN ('cleaned', 'needs_review')
              AND annotation IS NOT NULL
              AND TRIM(annotation) <> ''
            """
        ).fetchone()
        return int(row["count"])

    def count_playlists(self, *, kind: str | None = None) -> int:
        """Count playlists, optionally limited to a kind."""

        if kind is None:
            row = self.connection.execute("SELECT COUNT(*) AS count FROM playlists").fetchone()
        else:
            _require(kind in PLAYLIST_KINDS, f"Unknown playlist kind: {kind}")
            row = self.connection.execute(
                "SELECT COUNT(*) AS count FROM playlists WHERE kind = ?", (kind,)
            ).fetchone()
        return int(row["count"])

    def count_videos(self) -> int:
        """Count stored videos."""

        row = self.connection.execute("SELECT COUNT(*) AS count FROM videos").fetchone()
        return int(row["count"])

    def state_summary(self) -> dict[str, int]:
        """Return every count used by ``ybook status`` and final summaries."""

        transcripts = self.transcript_counts()
        chapters = self.chapter_counts()
        return {
            "channels": int(
                self.connection.execute("SELECT COUNT(*) AS count FROM channels").fetchone()[
                    "count"
                ]
            ),
            "playlists": self.count_playlists(),
            "regular_playlists": self.count_playlists(kind="regular"),
            "uploads_playlists": self.count_playlists(kind="uploads"),
            "videos": self.count_videos(),
            "transcripts_ok": transcripts["ok"],
            "transcripts_skipped": transcripts["skipped"],
            "transcripts_failed": transcripts["failed"],
            "chapters_cleaned": chapters["cleaned"],
            "chapters_needs_review": chapters["needs_review"],
            "chapters_failed": chapters["failed"],
            "annotations": self.annotation_count(),
        }


def _count_by_status(
    connection: sqlite3.Connection,
    table: str,
    statuses: frozenset[str],
) -> dict[str, int]:
    _require(table in {"transcripts", "chapters"}, f"Unsupported table: {table}")
    rows = connection.execute(
        f"SELECT status, COUNT(*) AS count FROM {table} GROUP BY status"  # noqa: S608
    ).fetchall()
    counts = {status: 0 for status in statuses}
    for row in rows:
        status = str(row["status"])
        if status in counts:
            counts[status] = int(row["count"])
    counts["total"] = sum(counts.values())
    return counts


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _chunks(values: set[str], size: int = 500) -> Iterable[list[str]]:
    ordered = sorted(values)
    for start in range(0, len(ordered), size):
        yield ordered[start : start + size]
