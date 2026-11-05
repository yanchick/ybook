"""CLI option wiring tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ybook.cli import app
from ybook.pipeline import BuildSummary, CleanSummary, FetchSummary, TranscriptSummary


def _config_file(tmp_path: Path) -> Path:
    config_path = tmp_path / "ybook.toml"
    config_path.write_text(
        f"""
[paths]
data_dir = "{(tmp_path / "data").as_posix()}"
output_dir = "{(tmp_path / "out").as_posix()}"
""",
        encoding="utf-8",
    )
    return config_path


def _empty_clean_summary() -> CleanSummary:
    return CleanSummary(
        selected=0,
        cleaned=0,
        needs_review=0,
        failed=0,
        annotations=0,
    )


def test_clean_language_option_is_forwarded(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "ybook.toml"
    config_path.write_text(
        f"""
[paths]
data_dir = "{(tmp_path / "data").as_posix()}"
output_dir = "{(tmp_path / "out").as_posix()}"
""",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run_clean(config, db, **kwargs) -> CleanSummary:
        captured.update(kwargs)
        return CleanSummary(
            selected=0,
            cleaned=0,
            needs_review=0,
            failed=0,
            annotations=0,
        )

    monkeypatch.setattr("ybook.cli.run_clean", fake_run_clean)
    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config_path),
            "clean",
            "--language",
            "ru",
            "--limit",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["annotation_language"] == "ru"
    assert captured["limit"] == 5
    assert captured["annotations"] is None
    assert captured["llm_enabled"] is None


def test_clean_llm_flag_is_forwarded(tmp_path: Path, monkeypatch) -> None:
    config_path = _config_file(tmp_path)
    captured: dict[str, object] = {}

    def fake_run_clean(config, db, **kwargs) -> CleanSummary:
        captured.update(kwargs)
        return _empty_clean_summary()

    monkeypatch.setattr("ybook.cli.run_clean", fake_run_clean)
    runner = CliRunner()
    for extra_args, expected in (
        ([], None),
        (["--llm"], True),
        (["--no-llm"], False),
    ):
        captured.clear()
        result = runner.invoke(
            app,
            ["--config", str(config_path), "clean", *extra_args],
        )
        assert result.exit_code == 0, result.output
        assert captured["llm_enabled"] is expected


def test_all_llm_flag_is_forwarded(tmp_path: Path, monkeypatch) -> None:
    config_path = _config_file(tmp_path)
    captured: dict[str, object] = {}

    def fake_fetch(config, db, **kwargs) -> FetchSummary:
        return FetchSummary(
            channel_title="Test channel",
            playlists=0,
            uploads_videos=0,
            videos=0,
        )

    def fake_transcripts(config, db, **kwargs) -> TranscriptSummary:
        return TranscriptSummary(
            selected=0,
            ok=0,
            skipped=0,
            failed=0,
            already_done=0,
        )

    def fake_run_clean(config, db, **kwargs) -> CleanSummary:
        captured.update(kwargs)
        return _empty_clean_summary()

    def fake_build(config, db, **kwargs) -> BuildSummary:
        return BuildSummary(path=tmp_path / "out" / "book.epub", parts=0, chapters=0)

    monkeypatch.setattr("ybook.cli.fetch_channel_data", fake_fetch)
    monkeypatch.setattr("ybook.cli.run_transcripts", fake_transcripts)
    monkeypatch.setattr("ybook.cli.run_clean", fake_run_clean)
    monkeypatch.setattr("ybook.cli.build_book", fake_build)

    runner = CliRunner()
    for extra_args, expected in (
        ([], None),
        (["--llm"], True),
        (["--no-llm"], False),
    ):
        captured.clear()
        result = runner.invoke(
            app,
            ["--config", str(config_path), "all", *extra_args],
        )
        assert result.exit_code == 0, result.output
        assert captured["llm_enabled"] is expected
