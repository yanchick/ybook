"""Typer command-line interface for ybook."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .config import Config
from .db import Database
from .errors import YbookError
from .pipeline import (
    build_book,
    fetch_channel_data,
    run_clean,
    run_transcripts,
)
from .utils import setup_logging

app = typer.Typer(
    name="ybook",
    help=(
        "Turn a YouTube channel into a personal EPUB book from edited transcripts.\n\n"
        "Typical flow: ybook all --channel URL"
    ),
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
error_console = Console(stderr=True)


@dataclass(frozen=True)
class Runtime:
    """Objects shared by all commands."""

    config: Config


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to ybook.toml (default: ./ybook.toml if present).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    version: bool = typer.Option(
        False,
        "--version",
        is_eager=True,
        help="Show ybook version and exit.",
    ),
) -> None:
    """Load configuration and environment before running a command."""

    if version:
        typer.echo(f"ybook {__version__}")
        raise typer.Exit()

    setup_logging(verbose)
    _load_dotenv_files(config)
    try:
        loaded = Config.load(config)
    except YbookError as exc:
        _print_error(exc)
        raise typer.Exit(1) from exc
    ctx.obj = Runtime(config=loaded)
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command()
def fetch(
    ctx: typer.Context,
    channel: Optional[str] = typer.Option(
        None, "--channel", help="Channel URL, @handle, or UC... ID."
    ),
) -> None:
    """Fetch channel metadata, playlists, uploads, and videos."""

    runtime = _runtime(ctx)
    config = runtime.config
    try:
        with Database(config.db_path) as db:
            summary = fetch_channel_data(config, db, channel=channel)
    except (YbookError, OSError, sqlite3.Error) as exc:
        _print_error(exc)
        raise typer.Exit(1) from exc

    table = Table(title="Fetch complete", show_header=True, header_style="bold")
    table.add_column("Item")
    table.add_column("Count", justify="right")
    table.add_row("Channel", escape(summary.channel_title))
    table.add_row("Regular playlists", str(summary.playlists))
    table.add_row("Uploads entries", str(summary.uploads_videos))
    table.add_row("Videos", str(summary.videos))
    console.print(table)


@app.command()
def transcripts(
    ctx: typer.Context,
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Single transcript language code (default: config, en).",
    ),
    limit: Optional[int] = typer.Option(None, "--limit", min=1, help="Process at most N videos."),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-fetch even if a transcript already exists."
    ),
    workers: Optional[int] = typer.Option(
        None, "--workers", min=1, help="Override transcript worker count."
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="auto, youtube-transcript-api, or yt-dlp (debugging).",
    ),
) -> None:
    """Download subtitle transcripts without audio or speech recognition."""

    _validate_provider(provider)
    runtime = _runtime(ctx)
    config = runtime.config
    try:
        with Database(config.db_path) as db:
            summary = run_transcripts(
                config,
                db,
                language=language,
                limit=limit,
                force=force,
                workers=workers,
                provider=provider,
            )
    except (YbookError, OSError, sqlite3.Error) as exc:
        _print_error(exc)
        raise typer.Exit(1) from exc

    table = Table(title="Transcripts", show_header=True, header_style="bold")
    table.add_column("Result")
    table.add_column("Count", justify="right")
    table.add_row("Processed", str(summary.selected))
    table.add_row("OK", str(summary.ok))
    table.add_row("Skipped", str(summary.skipped))
    table.add_row("Failed", str(summary.failed))
    table.add_row("Already done", str(summary.already_done))
    console.print(table)
    if summary.selected == 0:
        console.print("Nothing to do.")


@app.command()
def clean(
    ctx: typer.Context,
    limit: Optional[int] = typer.Option(None, "--limit", min=1, help="Process at most N chapters."),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help=(
            "Re-clean and regenerate annotations; with --no-annotations or --no-llm "
            "clears them after successful cleaning."
        ),
    ),
    llm: Optional[bool] = typer.Option(
        None,
        "--llm/--no-llm",
        help=(
            "Use the LLM for cleaning and annotations (default: config.llm.enabled). "
            "--no-llm copies transcripts unchanged."
        ),
    ),
    model: Optional[str] = typer.Option(None, "--model", help="Override the configured LLM model."),
    workers: Optional[int] = typer.Option(
        None, "--workers", min=1, help="Override LLM worker count."
    ),
    no_annotations: bool = typer.Option(
        False, "--no-annotations", help="Skip annotation requests."
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Override the annotation language (default: book.language).",
    ),
) -> None:
    """Clean transcripts with the LLM, or copy them unchanged with --no-llm."""

    runtime = _runtime(ctx)
    config = runtime.config
    try:
        with Database(config.db_path) as db:
            summary = run_clean(
                config,
                db,
                limit=limit,
                force=force,
                model=model,
                workers=workers,
                annotations=False if no_annotations else None,
                annotation_language=language,
                llm_enabled=llm,
            )
    except (YbookError, OSError, sqlite3.Error) as exc:
        _print_error(exc)
        raise typer.Exit(1) from exc

    table = Table(title="Cleaning", show_header=True, header_style="bold")
    table.add_column("Result")
    table.add_column("Count", justify="right")
    table.add_row("Processed", str(summary.selected))
    table.add_row("Cleaned", str(summary.cleaned))
    table.add_row("Needs review", str(summary.needs_review))
    table.add_row("Failed", str(summary.failed))
    table.add_row("Annotations", str(summary.annotations))
    console.print(table)
    if summary.selected == 0:
        console.print("Nothing to do.")


@app.command()
def build(
    ctx: typer.Context,
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Output EPUB path (default: out/book.epub)."
    ),
    title: Optional[str] = typer.Option(None, "--title", help="Book title override."),
    author: Optional[str] = typer.Option(None, "--author", help="Book author override."),
    language: Optional[str] = typer.Option(
        None, "--language", "-l", help="Book language override."
    ),
    chronological: bool = typer.Option(
        False,
        "--chronological",
        help="Ignore playlists and build one chronological part.",
    ),
) -> None:
    """Build the EPUB offline from the SQLite database."""

    runtime = _runtime(ctx)
    config = runtime.config
    try:
        with Database(config.db_path) as db:
            summary = build_book(
                config,
                db,
                out=out,
                title=title,
                author=author,
                language=language,
                chronological=chronological,
            )
    except (YbookError, OSError, sqlite3.Error) as exc:
        _print_error(exc)
        raise typer.Exit(1) from exc

    table = Table(title="EPUB built", show_header=True, header_style="bold")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("Path", escape(str(summary.path)))
    table.add_row("Parts", str(summary.parts))
    table.add_row("Chapters", str(summary.chapters))
    console.print(table)


@app.command(name="all")
def run_all(
    ctx: typer.Context,
    channel: Optional[str] = typer.Option(
        None, "--channel", help="Channel URL, @handle, or UC... ID."
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Output EPUB path (default: out/book.epub)."
    ),
    language: Optional[str] = typer.Option(
        None, "--language", "-l", help="Transcript and book language."
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", min=1, help="Process at most N videos per stage."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-fetch transcripts and re-clean chapters."
    ),
    llm: Optional[bool] = typer.Option(
        None,
        "--llm/--no-llm",
        help=(
            "Use the LLM for cleaning and annotations (default: config.llm.enabled). "
            "--no-llm copies transcripts unchanged."
        ),
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="auto, youtube-transcript-api, or yt-dlp (debugging).",
    ),
    title: Optional[str] = typer.Option(None, "--title", help="Book title override."),
    author: Optional[str] = typer.Option(None, "--author", help="Book author override."),
    chronological: bool = typer.Option(
        False, "--chronological", help="Build one chronological part."
    ),
    no_annotations: bool = typer.Option(
        False, "--no-annotations", help="Skip annotation requests."
    ),
) -> None:
    """Run fetch, transcripts, clean, and build in one go.

    Pass --no-llm to build from normalized transcripts without any LLM calls
    or API key.
    """

    _validate_provider(provider)
    runtime = _runtime(ctx)
    config = runtime.config
    try:
        with Database(config.db_path) as db:
            fetch_summary = fetch_channel_data(config, db, channel=channel)
            console.print(
                f"[bold]Channel:[/bold] {escape(fetch_summary.channel_title)} "
                f"({fetch_summary.playlists} playlists, {fetch_summary.videos} videos)"
            )

            transcript_summary = run_transcripts(
                config,
                db,
                language=language,
                limit=limit,
                force=force,
                provider=provider,
            )
            console.print(
                "Transcripts: "
                f"ok={transcript_summary.ok}, "
                f"skipped={transcript_summary.skipped}, "
                f"failed={transcript_summary.failed}, "
                f"already done={transcript_summary.already_done}"
            )

            clean_summary = run_clean(
                config,
                db,
                limit=limit,
                force=force,
                annotations=False if no_annotations else None,
                annotation_language=language,
                llm_enabled=llm,
            )
            console.print(
                "Cleaning: "
                f"cleaned={clean_summary.cleaned}, "
                f"needs_review={clean_summary.needs_review}, "
                f"failed={clean_summary.failed}, "
                f"annotations={clean_summary.annotations}"
            )

            build_summary = build_book(
                config,
                db,
                out=out,
                title=title,
                author=author,
                language=language,
                chronological=chronological,
            )
            _print_final_summary(db, build_summary.path)
    except (YbookError, OSError, sqlite3.Error) as exc:
        _print_error(exc)
        raise typer.Exit(1) from exc


@app.command()
def status(ctx: typer.Context) -> None:
    """Print channel, transcript, chapter, and annotation counts."""

    runtime = _runtime(ctx)
    config = runtime.config
    try:
        with Database(config.db_path) as db:
            channel = db.get_channel()
            counts = db.state_summary()
    except (YbookError, OSError, sqlite3.Error) as exc:
        _print_error(exc)
        raise typer.Exit(1) from exc

    if channel:
        console.print(
            f"[bold]Channel:[/bold] {escape(str(channel.get('title') or channel.get('id')))}"
        )
    else:
        console.print("[yellow]No channel fetched yet.[/yellow]")
    console.print(f"[dim]Database:[/dim] {escape(str(config.db_path))}")
    _print_counts_table(counts)


def _runtime(ctx: typer.Context) -> Runtime:
    runtime = ctx.obj
    if not isinstance(runtime, Runtime):
        _print_error("Internal error: configuration context is missing")
        raise typer.Exit(1)
    return runtime


def _print_final_summary(db: Database, epub_path: Path) -> None:
    console.print("[bold green]Pipeline complete[/bold green]")
    counts = db.state_summary()
    _print_counts_table(counts, title="Final summary", extra={"EPUB": str(epub_path)})


def _print_counts_table(
    counts: dict[str, int],
    *,
    title: str = "Status",
    extra: dict[str, str] | None = None,
) -> None:
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Item")
    table.add_column("Count", justify="right")
    table.add_row("Playlists (regular)", str(counts.get("regular_playlists", 0)))
    table.add_row("Videos", str(counts.get("videos", 0)))
    table.add_row("Transcripts OK", str(counts.get("transcripts_ok", 0)))
    table.add_row("Transcripts skipped", str(counts.get("transcripts_skipped", 0)))
    table.add_row("Transcripts failed", str(counts.get("transcripts_failed", 0)))
    table.add_row("Chapters cleaned", str(counts.get("chapters_cleaned", 0)))
    table.add_row("Chapters needs review", str(counts.get("chapters_needs_review", 0)))
    table.add_row("Chapters failed", str(counts.get("chapters_failed", 0)))
    table.add_row("Annotations", str(counts.get("annotations", 0)))
    for label, value in (extra or {}).items():
        table.add_row(label, escape(value))
    console.print(table)


def _validate_provider(provider: str | None) -> None:
    if provider in {None, "auto", "youtube-transcript-api", "yt-dlp"}:
        return
    _print_error(
        f"Unknown transcript provider '{provider}'. " "Use auto, youtube-transcript-api, or yt-dlp."
    )
    raise typer.Exit(2)


def _load_dotenv_files(config: Path | None) -> None:
    candidates: list[Path] = [Path.cwd() / ".env"]
    if config is not None:
        path = Path(config)
        directory = path if path.is_dir() else path.parent
        candidates.insert(0, directory / ".env")
    for candidate in candidates:
        load_dotenv(candidate, override=False)


def _print_error(exc: object) -> None:
    error_console.print(f"Error: {exc}", style="red", markup=False)
