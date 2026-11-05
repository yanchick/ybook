"""EPUB package structure, metadata, nested TOC, escaping, and determinism."""

from __future__ import annotations

import zipfile
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from ybook.epub import Book, Chapter, Part, build_epub
from ybook.errors import BookBuildError

NS = {
    "container": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
    "xhtml": "http://www.w3.org/1999/xhtml",
    "epub": "http://www.idpf.org/2007/ops",
    "ncx": "http://www.daisy.org/z3986/2005/ncx/",
}


def _book() -> Book:
    return Book(
        title="My <Book> & More",
        author="A & B",
        language="en",
        description="About <this> & that",
        source_url="https://www.youtube.com/channel/UCexample",
        channel_id="UCexample",
        generated_at=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        parts=[
            Part(
                title="Part & One",
                description="Description <one> & more",
                chapters=[
                    Chapter(
                        title="First <Chapter>",
                        text="Tom & Jerry <3 live.\n\nSecond paragraph.",
                        annotation="An <annotation> & note.",
                        published_at="2024-01-01T10:00:00Z",
                        duration_seconds=65,
                        video_id="vid1",
                    ),
                    Chapter(
                        title="Second Chapter",
                        text="Plain text.",
                        published_at="2024-01-02T10:00:00Z",
                        duration_seconds=0,
                        video_id="vid2",
                    ),
                ],
            ),
            Part(
                title="Part Two",
                chapters=[
                    Chapter(
                        title="Third Chapter",
                        text="Final text.",
                        annotation="Last note.",
                        video_id="vid3",
                    )
                ],
            ),
        ],
    )


def _build(tmp_path: Path, name: str = "book.epub") -> tuple[Path, zipfile.ZipFile]:
    path = tmp_path / name
    build_epub(_book(), path)
    return path, zipfile.ZipFile(path)


def test_mimetype_is_first_and_stored(tmp_path: Path) -> None:
    path, archive = _build(tmp_path)
    try:
        infos = archive.infolist()
        assert infos[0].filename == "mimetype"
        assert infos[0].compress_type == zipfile.ZIP_STORED
        assert archive.read("mimetype") == b"application/epub+zip"
        names = archive.namelist()
        for required in (
            "META-INF/container.xml",
            "OEBPS/content.opf",
            "OEBPS/nav.xhtml",
            "OEBPS/toc.ncx",
            "OEBPS/style.css",
            "OEBPS/text/title.xhtml",
            "OEBPS/text/about.xhtml",
            "OEBPS/text/part-001.xhtml",
            "OEBPS/text/chapter-001.xhtml",
            "OEBPS/text/chapter-002.xhtml",
            "OEBPS/text/chapter-003.xhtml",
            "OEBPS/text/part-002.xhtml",
        ):
            assert required in names
        assert not any("cover" in name.lower() for name in names)
    finally:
        archive.close()


def test_container_points_to_opf(tmp_path: Path) -> None:
    _, archive = _build(tmp_path)
    try:
        root = ET.fromstring(archive.read("META-INF/container.xml"))
        rootfile = root.find("container:rootfiles/container:rootfile", NS)
        assert rootfile is not None
        assert rootfile.get("full-path") == "OEBPS/content.opf"
        assert rootfile.get("media-type") == "application/oebps-package+xml"
    finally:
        archive.close()


def test_opf_metadata_manifest_and_no_cover(tmp_path: Path) -> None:
    _, archive = _build(tmp_path)
    try:
        opf_text = archive.read("OEBPS/content.opf").decode("utf-8")
        root = ET.fromstring(opf_text)
        assert root.tag == f"{{{NS['opf']}}}package"
        assert root.get("version") == "3.0"
        assert root.get("unique-identifier") == "pub-id"

        metadata = root.find("opf:metadata", NS)
        assert metadata is not None
        assert metadata.findtext("dc:title", namespaces=NS) == "My <Book> & More"
        assert metadata.findtext("dc:creator", namespaces=NS) == "A & B"
        assert metadata.findtext("dc:language", namespaces=NS) == "en"
        assert metadata.findtext("dc:description", namespaces=NS) == "About <this> & that"
        assert (
            metadata.findtext("dc:source", namespaces=NS)
            == "https://www.youtube.com/channel/UCexample"
        )
        assert metadata.findtext("dc:rights", namespaces=NS)
        assert metadata.findtext("dc:date", namespaces=NS) == "2024-01-02"
        identifier = metadata.find("dc:identifier", NS)
        assert identifier is not None
        assert identifier.get("id") == "pub-id"
        assert identifier.text.startswith("urn:uuid:")
        modified = metadata.find("opf:meta[@property='dcterms:modified']", NS)
        assert modified is not None
        assert modified.text == "2024-01-02T03:04:05Z"

        manifest = root.find("opf:manifest", NS)
        assert manifest is not None
        items = {item.get("id"): item for item in manifest.findall("opf:item", NS)}
        assert items["nav"].get("properties") == "nav"
        assert items["ncx"].get("media-type") == "application/x-dtbncx+xml"
        assert all(item.get("id") != "cover-image" for item in items.values())
        assert "cover" not in opf_text.lower()

        spine = root.find("opf:spine", NS)
        assert spine is not None
        assert spine.get("toc") == "ncx"
        refs = [item.get("idref") for item in spine.findall("opf:itemref", NS)]
        assert refs == [
            "titlepage",
            "about",
            "part-001",
            "chapter-001",
            "chapter-002",
            "part-002",
            "chapter-003",
        ]
    finally:
        archive.close()


def test_nested_navigation_and_existing_links(tmp_path: Path) -> None:
    _, archive = _build(tmp_path)
    try:
        nav_text = archive.read("OEBPS/nav.xhtml").decode("utf-8")
        root = ET.fromstring(nav_text)
        navs = list(root.iter(f"{{{NS['xhtml']}}}nav"))
        toc = next(nav for nav in navs if nav.get(f"{{{NS['epub']}}}type") == "toc")
        top_ol = toc.find("xhtml:ol", NS)
        assert top_ol is not None
        top_items = top_ol.findall("xhtml:li", NS)
        assert len(top_items) == 4

        first_part_link = top_items[2].find("xhtml:a", NS)
        assert first_part_link is not None
        assert first_part_link.text == "Part & One"
        nested = top_items[2].find("xhtml:ol", NS)
        assert nested is not None
        nested_links = nested.findall("xhtml:li/xhtml:a", NS)
        assert [link.text for link in nested_links] == [
            "First <Chapter>",
            "Second Chapter",
        ]

        second_nested = top_items[3].find("xhtml:ol", NS)
        assert second_nested is not None
        assert len(second_nested.findall("xhtml:li/xhtml:a", NS)) == 1

        landmarks = next(nav for nav in navs if nav.get(f"{{{NS['epub']}}}type") == "landmarks")
        assert landmarks is not None
        bodymatter = next(
            anchor
            for anchor in landmarks.iter(f"{{{NS['xhtml']}}}a")
            if anchor.get(f"{{{NS['epub']}}}type") == "bodymatter"
        )
        assert bodymatter.get("href") == "text/chapter-001.xhtml"

        for anchor in root.iter(f"{{{NS['xhtml']}}}a"):
            href = anchor.get("href")
            assert href
            target = href.split("#", 1)[0]
            assert f"OEBPS/{target}" in archive.namelist()
    finally:
        archive.close()


def test_ncx_is_nested_and_play_order_is_sequential(tmp_path: Path) -> None:
    _, archive = _build(tmp_path)
    try:
        root = ET.fromstring(archive.read("OEBPS/toc.ncx"))
        nav_map = root.find("ncx:navMap", NS)
        assert nav_map is not None
        top_points = nav_map.findall("ncx:navPoint", NS)
        assert len(top_points) == 4
        assert top_points[2].find("ncx:navLabel/ncx:text", NS).text == "Part & One"
        children = top_points[2].findall("ncx:navPoint", NS)
        assert len(children) == 2

        orders = [
            int(point.get("playOrder", "0")) for point in root.iter(f"{{{NS['ncx']}}}navPoint")
        ]
        assert orders == list(range(1, len(orders) + 1))
        depth = root.find("ncx:head/ncx:meta[@name='dtb:depth']", NS)
        assert depth is not None
        assert depth.get("content") == "2"
    finally:
        archive.close()


def test_title_about_and_chapter_pages_are_escaped(tmp_path: Path) -> None:
    _, archive = _build(tmp_path)
    try:
        title_raw = archive.read("OEBPS/text/title.xhtml").decode("utf-8")
        assert "My &lt;Book&gt; &amp; More" in title_raw
        assert "A &amp; B" in title_raw
        assert "Compiled from YouTube transcripts" in title_raw
        assert "Generated 2024-01-02" in title_raw
        ET.fromstring(title_raw)

        about_raw = archive.read("OEBPS/text/about.xhtml").decode("utf-8")
        assert "About &lt;this&gt; &amp; that" in about_raw
        assert "3 chapters in 2 parts." in about_raw
        ET.fromstring(about_raw)

        chapter_raw = archive.read("OEBPS/text/chapter-001.xhtml").decode("utf-8")
        assert "Tom &amp; Jerry &lt;3 live." in chapter_raw
        assert "An &lt;annotation&gt; &amp; note." in chapter_raw
        assert "1:05" in chapter_raw
        assert "https://www.youtube.com/watch?v=vid1" in chapter_raw
        assert 'class="annotation"' in chapter_raw
        chapter_root = ET.fromstring(chapter_raw)
        chapter_section = chapter_root.find(".//xhtml:section[@class='chapter']", NS)
        assert chapter_section is not None
        annotation = chapter_section.find("xhtml:p[@class='annotation']", NS)
        assert annotation is not None
        annotation_em = annotation.find("xhtml:em", NS)
        assert annotation_em is not None
        assert annotation_em.text == "An <annotation> & note."
        paragraphs = [
            node.text
            for node in chapter_section.findall("xhtml:p", NS)
            if node.get("class") is None
        ]
        assert "Tom & Jerry <3 live." in paragraphs
        assert "Second paragraph." in paragraphs
    finally:
        archive.close()


def test_uuid_is_deterministic(tmp_path: Path) -> None:
    first_path = tmp_path / "first.epub"
    second_path = tmp_path / "second.epub"
    build_epub(_book(), first_path)
    build_epub(_book(), second_path)

    def identifier(path: Path) -> str:
        with zipfile.ZipFile(path) as archive:
            root = ET.fromstring(archive.read("OEBPS/content.opf"))
            node = root.find("opf:metadata/dc:identifier", NS)
            assert node is not None
            return node.text or ""

    assert identifier(first_path) == identifier(second_path)


def test_all_xhtml_documents_are_well_formed_xml(tmp_path: Path) -> None:
    _, archive = _build(tmp_path)
    try:
        for name in archive.namelist():
            if name.endswith(".xhtml") or name.endswith(".opf") or name.endswith(".ncx"):
                ET.fromstring(archive.read(name))
    finally:
        archive.close()


def test_empty_book_raises(tmp_path: Path) -> None:
    with pytest.raises(BookBuildError, match="No chapters"):
        build_epub(
            Book(title="Empty", author="Nobody", parts=[Part(title="None")]),
            tmp_path / "empty.epub",
        )


def test_empty_title_raises(tmp_path: Path) -> None:
    with pytest.raises(BookBuildError, match="title"):
        build_epub(
            Book(
                title="",
                author="Nobody",
                parts=[Part(title="Part", chapters=[Chapter(title="C", text="Text.")])],
            ),
            tmp_path / "no-title.epub",
        )


def test_part_description_is_truncated_and_escaped(tmp_path: Path) -> None:
    book = Book(
        title="T",
        author="A",
        parts=[
            Part(
                title="Part",
                description="<b>" + "long text " * 100 + "</b>",
                chapters=[Chapter(title="C", text="Body.")],
            )
        ],
    )
    path = tmp_path / "truncated.epub"
    build_epub(book, path)
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("OEBPS/text/part-001.xhtml").decode("utf-8")
        assert "&lt;b&gt;" in raw
        assert "long text" in raw
        assert raw.count("long text") < 100
