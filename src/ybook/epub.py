"""A small EPUB 3 writer and database-to-book assembler.

No third-party EPUB library is used: the package is a ZIP with a precise
``mimetype`` entry, an OPF package, an XHTML navigation document, and an NCX
fallback for older readers.  There is no cover and no output format other than
EPUB.
"""

from __future__ import annotations

import html
import logging
import os
import re
import tempfile
import uuid
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Config
from .db import Database
from .errors import BookBuildError
from .text import format_duration, to_paragraphs
from .utils import iso_date, truncate, utc_now

_XML_INVALID = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chapter:
    """One chapter rendered from a cleaned transcript."""

    title: str
    text: str
    annotation: str = ""
    published_at: str = ""
    duration_seconds: int = 0
    url: str = ""
    video_id: str = ""
    status: str = "cleaned"


@dataclass(frozen=True)
class Part:
    """A book part, usually a playlist."""

    title: str
    description: str = ""
    chapters: list[Chapter] = field(default_factory=list)


@dataclass(frozen=True)
class Book:
    """Everything the EPUB writer needs, independent of the database."""

    title: str
    author: str
    language: str = "en"
    description: str = ""
    source_url: str = ""
    channel_id: str = ""
    rights: str = "Personal use only. Compiled from YouTube transcripts."
    generated_at: datetime | None = None
    parts: list[Part] = field(default_factory=list)


@dataclass(frozen=True)
class _RenderedChapter:
    chapter: Chapter
    filename: str


@dataclass(frozen=True)
class _RenderedPart:
    part: Part
    filename: str
    chapters: list[_RenderedChapter]


def build_epub(book: Book, out_path: str | Path) -> Path:
    """Write ``book`` as a valid EPUB 3 file and return the output path."""

    parts = [part for part in book.parts if part.chapters]
    if not parts:
        raise BookBuildError(
            "No chapters to build. Run 'ybook clean' first and check that cleaned files exist."
        )
    if not book.title.strip():
        raise BookBuildError("Book title is empty")

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    generated_at = _normalize_datetime(book.generated_at)
    identifier = _book_uuid(book)
    identifier_urn = f"urn:uuid:{identifier}"

    rendered_parts: list[_RenderedPart] = []
    chapter_number = 0
    for part_number, part in enumerate(parts, start=1):
        rendered_chapters: list[_RenderedChapter] = []
        for chapter in part.chapters:
            chapter_number += 1
            rendered_chapters.append(
                _RenderedChapter(
                    chapter=chapter,
                    filename=f"chapter-{chapter_number:03d}.xhtml",
                )
            )
        rendered_parts.append(
            _RenderedPart(
                part=part,
                filename=f"part-{part_number:03d}.xhtml",
                chapters=rendered_chapters,
            )
        )

    documents: list[tuple[str, str]] = [
        ("META-INF/container.xml", _container_xml()),
        (
            "OEBPS/content.opf",
            _content_opf(
                book=book,
                identifier_urn=identifier_urn,
                generated_at=generated_at,
                rendered_parts=rendered_parts,
            ),
        ),
        (
            "OEBPS/nav.xhtml",
            _nav_xhtml(book=book, rendered_parts=rendered_parts),
        ),
        (
            "OEBPS/toc.ncx",
            _ncx(book=book, identifier_urn=identifier_urn, rendered_parts=rendered_parts),
        ),
        ("OEBPS/style.css", _STYLESHEET),
        ("OEBPS/text/title.xhtml", _title_page(book, generated_at)),
        ("OEBPS/text/about.xhtml", _about_page(book, len(parts), chapter_number)),
    ]
    for part in rendered_parts:
        documents.append((f"OEBPS/text/{part.filename}", _part_page(book, part)))
        for chapter in part.chapters:
            documents.append(
                (f"OEBPS/text/{chapter.filename}", _chapter_page(book, chapter.chapter))
            )

    _write_archive(target, documents)
    return target


def book_from_database(
    db: Database,
    config: Config,
    *,
    title: str | None = None,
    author: str | None = None,
    language: str | None = None,
    chronological: bool = False,
) -> Book:
    """Assemble a :class:`Book` from cleaned chapters in SQLite.

    This function performs no network access, so ``ybook build`` works offline.
    """

    channel = db.get_channel()
    current_channel_id = str((channel or {}).get("id") or "")
    rows = db.list_ready_chapters()
    if current_channel_id:
        other_channels = sorted(
            {
                str(row.get("channel_id") or "")
                for row in rows
                if str(row.get("channel_id") or "") != current_channel_id
            }
        )
        if other_channels:
            LOGGER.warning(
                "Ignoring cleaned chapters from non-current channel(s): %s",
                ", ".join(channel_id for channel_id in other_channels if channel_id) or "unknown",
            )
        rows = [row for row in rows if str(row.get("channel_id") or "") == current_channel_id]

    ready: dict[str, Chapter] = {}
    for row in rows:
        if not _passes_selection(row, config):
            continue
        cleaned_path = Path(str(row.get("cleaned_path") or ""))
        if not cleaned_path.is_file():
            continue
        try:
            text = cleaned_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if not text.strip():
            continue
        video_id = str(row["video_id"])
        ready[video_id] = Chapter(
            title=str(row.get("title") or "Untitled video"),
            text=text,
            annotation=str(row.get("annotation") or ""),
            published_at=str(row.get("published_at") or ""),
            duration_seconds=int(row.get("duration_seconds") or 0),
            url=f"https://www.youtube.com/watch?v={video_id}",
            video_id=video_id,
            status=str(row.get("chapter_status") or "cleaned"),
        )

    if not ready:
        raise BookBuildError(
            "No cleaned chapters are available. Run 'ybook transcripts' and "
            "'ybook clean' first (or check include_shorts/include_live filters)."
        )

    channel_title = str((channel or {}).get("title") or "")
    title_candidate = title if title and title.strip() else config.book.title
    book_title = (title_candidate or channel_title).strip()
    if not book_title:
        raise BookBuildError("Book title is empty. Pass --title or set book.title in the config.")
    author_candidate = author if author and author.strip() else config.book.author
    book_author = (author_candidate or channel_title or book_title).strip()
    language_candidate = language if language and language.strip() else config.book.language
    book_language = language_candidate.strip() or "en"
    description = (config.book.description or str((channel or {}).get("description") or "")).strip()
    source_url = str((channel or {}).get("url") or "")
    channel_id = str((channel or {}).get("id") or "")

    chronological_mode = chronological or not config.book.group_by_playlist
    if chronological_mode:
        chapters = sorted(
            ready.values(), key=lambda chapter: (chapter.published_at, chapter.video_id)
        )
        parts = [Part(title="All chapters", chapters=chapters)]
    else:
        parts = _group_by_playlist(db, ready)

    return Book(
        title=book_title,
        author=book_author,
        language=book_language,
        description=description,
        source_url=source_url,
        channel_id=channel_id,
        generated_at=utc_now(),
        parts=parts,
    )


def _group_by_playlist(db: Database, ready: dict[str, Chapter]) -> list[Part]:
    playlists = db.list_playlists(kind="regular")
    memberships = db.playlist_memberships()

    assignments: dict[str, str] = {}
    best_keys: dict[str, tuple[int, int, str]] = {}
    for membership in memberships:
        video_id = str(membership["video_id"])
        if video_id not in ready:
            continue
        playlist_id = str(membership["playlist_id"])
        key = (
            int(membership["effective_item_count"]),
            int(membership["playlist_position"]),
            playlist_id,
        )
        if video_id not in best_keys or key < best_keys[video_id]:
            best_keys[video_id] = key
            assignments[video_id] = playlist_id

    chapters_by_playlist: dict[str, list[Chapter]] = {
        str(playlist["id"]): [] for playlist in playlists
    }
    for membership in memberships:
        video_id = str(membership["video_id"])
        playlist_id = str(membership["playlist_id"])
        if assignments.get(video_id) != playlist_id or video_id not in ready:
            continue
        chapters_by_playlist.setdefault(playlist_id, []).append(ready[video_id])

    parts: list[Part] = []
    for playlist in playlists:
        playlist_id = str(playlist["id"])
        chapters = chapters_by_playlist.get(playlist_id, [])
        if not chapters:
            continue
        parts.append(
            Part(
                title=str(playlist.get("title") or "Untitled playlist"),
                description=str(playlist.get("description") or ""),
                chapters=chapters,
            )
        )

    assigned_videos = set(assignments)
    other = [chapter for video_id, chapter in ready.items() if video_id not in assigned_videos]
    if other:
        other.sort(key=lambda chapter: (chapter.published_at, chapter.video_id))
        parts.append(Part(title="Other videos", chapters=other))

    return parts


def _passes_selection(row: dict[str, Any], config: Config) -> bool:
    duration = int(row.get("duration_seconds") or 0)
    if not config.book.include_shorts and 0 < duration <= 180:
        return False
    if not config.book.include_live and str(row.get("live_status") or "none") in {
        "live",
        "upcoming",
    }:
        return False
    return True


def _content_opf(
    *,
    book: Book,
    identifier_urn: str,
    generated_at: datetime,
    rendered_parts: list[_RenderedPart],
) -> str:
    lang = _xml(book.language)
    date = generated_at.strftime("%Y-%m-%d")
    modified = generated_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" '
            f'unique-identifier="pub-id" xml:lang="{lang}" '
            'prefix="dcterms: http://purl.org/dc/terms/">'
        ),
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">',
        f'    <dc:identifier id="pub-id">{_xml(identifier_urn)}</dc:identifier>',
        f"    <dc:title>{_xml(book.title)}</dc:title>",
        f"    <dc:creator>{_xml(book.author)}</dc:creator>",
        f"    <dc:language>{lang}</dc:language>",
        f"    <dc:description>{_xml(book.description)}</dc:description>",
        f"    <dc:source>{_xml(book.source_url)}</dc:source>",
        f"    <dc:rights>{_xml(book.rights)}</dc:rights>",
        f"    <dc:date>{date}</dc:date>",
        f'    <meta property="dcterms:modified">{modified}</meta>',
        "  </metadata>",
        "  <manifest>",
        '    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        '    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
        '    <item id="style" href="style.css" media-type="text/css"/>',
        '    <item id="titlepage" href="text/title.xhtml" media-type="application/xhtml+xml"/>',
        '    <item id="about" href="text/about.xhtml" media-type="application/xhtml+xml"/>',
    ]
    for part in rendered_parts:
        part_id = part.filename.removesuffix(".xhtml")
        rows.append(
            f'    <item id="{part_id}" href="text/{part.filename}" '
            'media-type="application/xhtml+xml"/>'
        )
        for chapter in part.chapters:
            chapter_id = chapter.filename.removesuffix(".xhtml")
            rows.append(
                f'    <item id="{chapter_id}" href="text/{chapter.filename}" '
                'media-type="application/xhtml+xml"/>'
            )
    rows.extend(
        [
            "  </manifest>",
            '  <spine toc="ncx">',
            '    <itemref idref="titlepage"/>',
            '    <itemref idref="about"/>',
        ]
    )
    for part in rendered_parts:
        rows.append(f'    <itemref idref="{part.filename.removesuffix(".xhtml")}"/>')
        for chapter in part.chapters:
            rows.append(f'    <itemref idref="{chapter.filename.removesuffix(".xhtml")}"/>')
    rows.extend(["  </spine>", "</package>"])
    return "\n".join(rows) + "\n"


def _nav_xhtml(*, book: Book, rendered_parts: list[_RenderedPart]) -> str:
    lang = _xml(book.language)
    first_chapter = next(
        (chapter for part in rendered_parts for chapter in part.chapters),
        None,
    )
    bodymatter_href = (
        f"text/{first_chapter.filename}" if first_chapter is not None else "text/title.xhtml"
    )
    rows = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            '<html xmlns="http://www.w3.org/1999/xhtml" '
            'xmlns:epub="http://www.idpf.org/2007/ops" '
            f'xml:lang="{lang}" lang="{lang}">'
        ),
        "  <head>",
        '    <meta charset="utf-8"/>',
        "    <title>Contents</title>",
        '    <link rel="stylesheet" type="text/css" href="style.css"/>',
        "  </head>",
        "  <body>",
        '    <nav epub:type="toc" id="toc">',
        "      <h1>Contents</h1>",
        "      <ol>",
        '        <li><a href="text/title.xhtml">Title page</a></li>',
        '        <li><a href="text/about.xhtml">About this book</a></li>',
    ]
    for part in rendered_parts:
        rows.append("        <li>")
        rows.append(f'          <a href="text/{part.filename}">{_xml(part.part.title)}</a>')
        rows.append("          <ol>")
        for chapter in part.chapters:
            rows.append(
                f'            <li><a href="text/{chapter.filename}">'
                f"{_xml(chapter.chapter.title)}</a></li>"
            )
        rows.append("          </ol>")
        rows.append("        </li>")
    rows.extend(
        [
            "      </ol>",
            "    </nav>",
            '    <nav epub:type="landmarks" id="landmarks" hidden="hidden">',
            "      <h1>Landmarks</h1>",
            "      <ol>",
            (f'        <li><a epub:type="bodymatter" href="{bodymatter_href}">' "Begin</a></li>"),
            '        <li><a epub:type="toc" href="nav.xhtml#toc">Contents</a></li>',
            "      </ol>",
            "    </nav>",
            "  </body>",
            "</html>",
        ]
    )
    return "\n".join(rows) + "\n"


def _ncx(
    *,
    book: Book,
    identifier_urn: str,
    rendered_parts: list[_RenderedPart],
) -> str:
    play_order = 0
    nav_points: list[str] = []

    def point_block(order: int, label: str, src: str, children: list[str] | None = None) -> str:
        lines = [
            f'    <navPoint id="navpoint-{order}" playOrder="{order}">',
            f"      <navLabel><text>{_xml(label)}</text></navLabel>",
            f'      <content src="{_xml(src)}"/>',
        ]
        if children:
            lines.extend(children)
        lines.append("    </navPoint>")
        return "\n".join(lines)

    play_order += 1
    nav_points.append(point_block(play_order, "Title page", "text/title.xhtml"))
    play_order += 1
    nav_points.append(point_block(play_order, "About this book", "text/about.xhtml"))

    for part in rendered_parts:
        play_order += 1
        parent_order = play_order
        child_points: list[str] = []
        for chapter in part.chapters:
            play_order += 1
            child_points.append(
                point_block(
                    play_order,
                    chapter.chapter.title,
                    f"text/{chapter.filename}",
                )
            )
        nav_points.append(
            point_block(
                parent_order,
                part.part.title,
                f"text/{part.filename}",
                child_points,
            )
        )

    rows = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1" '
            f'xml:lang="{_xml(book.language)}">'
        ),
        "  <head>",
        f'    <meta name="dtb:uid" content="{_xml(identifier_urn)}"/>',
        '    <meta name="dtb:depth" content="2"/>',
        '    <meta name="dtb:totalPageCount" content="0"/>',
        '    <meta name="dtb:maxPageNumber" content="0"/>',
        "  </head>",
        f"  <docTitle><text>{_xml(book.title)}</text></docTitle>",
        f"  <docAuthor><text>{_xml(book.author)}</text></docAuthor>",
        "  <navMap>",
        *nav_points,
        "  </navMap>",
        "</ncx>",
    ]
    return "\n".join(rows) + "\n"


def _title_page(book: Book, generated_at: datetime) -> str:
    source_line = ""
    if book.source_url:
        source_line = (
            f'<p class="source">Source: <a href="{_xml(book.source_url)}">'
            f"{_xml(book.source_url)}</a></p>"
        )
    body = "\n".join(
        [
            '<section class="title-page" epub:type="titlepage">',
            f'  <h1 class="book-title">{_xml(book.title)}</h1>',
            f'  <p class="book-author">by {_xml(book.author)}</p>',
            '  <p class="compiled">Compiled from YouTube transcripts</p>',
            f'  <p class="generated">Generated {generated_at.strftime("%Y-%m-%d")}</p>',
            source_line,
            "</section>",
        ]
    )
    return _xhtml_document(book.title, body, language=book.language, depth=1)


def _about_page(book: Book, part_count: int, chapter_count: int) -> str:
    paragraphs = "".join(
        f"<p>{_xml(paragraph)}</p>" for paragraph in to_paragraphs(book.description)
    )
    if not paragraphs:
        paragraphs = "<p>This book was compiled automatically from YouTube transcripts.</p>"
    source_line = ""
    if book.source_url:
        source_line = (
            f'<p class="source">Channel: <a href="{_xml(book.source_url)}">'
            f"{_xml(book.source_url)}</a></p>"
        )
    body = "\n".join(
        [
            '<section class="about" epub:type="preface">',
            "  <h1>About this book</h1>",
            paragraphs,
            f"  <p>{chapter_count} chapters in {part_count} parts.</p>",
            source_line,
            (
                "  <p>Chapter texts are edited transcripts of spoken YouTube videos. "
                "They are not verbatim publications and may contain editing "
                "imperfections.</p>"
            ),
            "</section>",
        ]
    )
    return _xhtml_document("About this book", body, language=book.language, depth=1)


def _part_page(book: Book, rendered: _RenderedPart) -> str:
    description = ""
    if rendered.part.description.strip():
        snippet = truncate(re.sub(r"\s+", " ", rendered.part.description).strip(), 600)
        description = f'  <p class="part-description">{_xml(snippet)}</p>'
    body = "\n".join(
        [
            '<section class="part" epub:type="part">',
            f"  <h1>{_xml(rendered.part.title)}</h1>",
            description,
            "</section>",
        ]
    )
    return _xhtml_document(rendered.part.title, body, language=book.language, depth=1)


def _chapter_page(book: Book, chapter: Chapter) -> str:
    meta_parts: list[str] = []
    published = iso_date(chapter.published_at)
    if published:
        meta_parts.append(f"Published {_xml(published)}")
    if chapter.duration_seconds > 0:
        meta_parts.append(_xml(format_duration(chapter.duration_seconds)))
    video_url = chapter.url or (
        f"https://www.youtube.com/watch?v={chapter.video_id}" if chapter.video_id else ""
    )
    if video_url:
        meta_parts.append(f'<a href="{_xml(video_url)}">Watch on YouTube</a>')
    meta = f'  <p class="meta">{" · ".join(meta_parts)}</p>' if meta_parts else ""
    annotation = ""
    if chapter.annotation.strip():
        annotation = f'  <p class="annotation"><em>{_xml(chapter.annotation.strip())}</em></p>'
    paragraphs = "\n".join(
        f"  <p>{_xml(paragraph)}</p>" for paragraph in to_paragraphs(chapter.text)
    )
    body = "\n".join(
        [
            '<section class="chapter" epub:type="chapter">',
            f"  <h1>{_xml(chapter.title)}</h1>",
            meta,
            annotation,
            paragraphs,
            "</section>",
        ]
    )
    return _xhtml_document(chapter.title, body, language=book.language, depth=1)


def _xhtml_document(title: str, body: str, *, language: str, depth: int) -> str:
    stylesheet = "../style.css" if depth else "style.css"
    lang = _xml(language)
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                '<html xmlns="http://www.w3.org/1999/xhtml" '
                'xmlns:epub="http://www.idpf.org/2007/ops" '
                f'xml:lang="{lang}" lang="{lang}">'
            ),
            "  <head>",
            '    <meta charset="utf-8"/>',
            f"    <title>{_xml(title)}</title>",
            f'    <link rel="stylesheet" type="text/css" href="{stylesheet}"/>',
            "  </head>",
            "  <body>",
            body,
            "  </body>",
            "</html>",
            "",
        ]
    )


def _container_xml() -> str:
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">',
            "  <rootfiles>",
            (
                '    <rootfile full-path="OEBPS/content.opf" '
                'media-type="application/oebps-package+xml"/>'
            ),
            "  </rootfiles>",
            "</container>",
            "",
        ]
    )


def _book_uuid(book: Book) -> str:
    identity = book.channel_id or book.source_url or book.author
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ybook:{identity}:{book.title}"))


def _normalize_datetime(value: datetime | None) -> datetime:
    if value is None:
        return utc_now()
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _xml(value: Any) -> str:
    text = _XML_INVALID.sub("", str(value if value is not None else ""))
    return html.escape(text, quote=True)


def _write_archive(target: Path, documents: Iterable[tuple[str, str]]) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            mimetype_info = zipfile.ZipInfo("mimetype", date_time=_ZIP_DATE)
            mimetype_info.compress_type = zipfile.ZIP_STORED
            mimetype_info.external_attr = 0o644 << 16
            archive.writestr(mimetype_info, "application/epub+zip")

            for name, content in documents:
                info = zipfile.ZipInfo(name, date_time=_ZIP_DATE)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, content.encode("utf-8"))
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


_STYLESHEET = """\
@charset "utf-8";

html {
  font-size: 100%;
}

body {
  font-family: Georgia, "Iowan Old Style", "Palatino Linotype", "Times New Roman", serif;
  line-height: 1.65;
  margin: 0 5%;
  color: #1d1d1b;
  background: #fffdf8;
  hyphens: auto;
  -epub-hyphens: auto;
}

h1 {
  font-family: "Palatino Linotype", Palatino, Georgia, serif;
  font-size: 1.65em;
  line-height: 1.2;
  font-weight: 600;
  margin: 1.4em 0 0.9em;
  text-indent: 0;
  letter-spacing: 0.01em;
}

p {
  margin: 0 0 0.55em;
  text-indent: 1.25em;
  orphans: 2;
  widows: 2;
}

p.meta,
p.annotation,
p.part-description,
p.source,
p.compiled,
p.generated,
p.book-author {
  text-indent: 0;
}

section.part,
section.chapter,
section.about,
section.title-page {
  page-break-before: always;
  break-before: page;
}

section.title-page {
  page-break-before: avoid;
  break-before: avoid;
  padding-top: 18%;
}

h1.book-title {
  font-size: 2.1em;
  margin-bottom: 0.35em;
}

p.book-author {
  font-size: 1.15em;
  font-style: italic;
  margin-bottom: 2.2em;
}

p.compiled,
p.generated,
p.source {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 0.88em;
  color: #66605a;
  margin-bottom: 0.45em;
}

p.meta {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 0.82em;
  color: #6b6258;
  border-bottom: 1px solid #ddd4c7;
  padding-bottom: 0.7em;
  margin-bottom: 1.2em;
}

p.annotation {
  font-style: italic;
  color: #3c352e;
  border-left: 3px solid #b8a78d;
  padding: 0.15em 0 0.15em 0.9em;
  margin: 1.2em 0 1.5em;
}

p.part-description {
  font-style: italic;
  color: #5b534a;
  margin-bottom: 1.2em;
}

a {
  color: #4d5b63;
  text-decoration: underline;
}

section.part h1 {
  margin-top: 20%;
  font-size: 2em;
}

@media screen {
  body {
    max-width: 46em;
    margin: 2em auto;
    padding: 0 1.5em;
  }
}
"""
