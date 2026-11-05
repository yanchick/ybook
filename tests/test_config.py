"""Configuration loading, defaults, environment overrides, and paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from ybook.config import Config
from ybook.errors import ConfigError

_ENV_NAMES = [
    "YBOOK_CHANNEL",
    "OPENAI_BASE_URL",
    "YBOOK_LLM_MODEL",
    "YOUTUBE_API_KEY",
    "OPENAI_API_KEY",
    "YBOOK_LLM_API_KEY",
    "CUSTOM_YOUTUBE_KEY",
    "CUSTOM_LLM_KEY",
]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_load_toml_and_resolve_relative_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "ybook.toml"
    config_path.write_text(
        """
channel = "https://www.youtube.com/@example"

[paths]
data_dir = "var/state"
output_dir = "books"

[youtube]
api_key_env = "CUSTOM_YOUTUBE_KEY"

[transcripts]
languages = ["ru", "en"]
provider = "yt-dlp"
workers = 2

[llm]
base_url = "https://llm.example/v1/"
api_key_env = "CUSTOM_LLM_KEY"
model = "custom-model"

[book]
language = "ru"
annotations = false
""",
        encoding="utf-8",
    )

    config = Config.load(config_path)

    assert config.channel == "https://www.youtube.com/@example"
    assert config.paths.data_dir == (tmp_path / "var/state").resolve()
    assert config.paths.output_dir == (tmp_path / "books").resolve()
    assert config.db_path == (tmp_path / "var/state/state.sqlite").resolve()
    assert config.transcripts.languages == ["ru", "en"]
    assert config.transcripts.provider == "yt-dlp"
    assert config.llm.base_url == "https://llm.example/v1"
    assert config.book.language == "ru"
    assert config.book.annotations is False
    assert config.base_dir == tmp_path.resolve()


def test_defaults_without_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = Config.load()

    assert config.config_path is None
    assert config.channel == ""
    assert config.transcripts.languages == ["en"]
    assert config.transcripts.provider == "auto"
    assert config.book.language == "en"
    assert config.book.group_by_playlist is True
    assert config.llm.enabled is True
    assert config.llm.max_words_per_chunk == 2500
    assert config.llm.annotation_max_words == 12000
    assert config.paths.data_dir == (tmp_path / "data").resolve()
    assert config.paths.output_dir == (tmp_path / "out").resolve()


def test_llm_can_be_disabled_in_config(tmp_path: Path) -> None:
    config_path = tmp_path / "ybook.toml"
    config_path.write_text("[llm]\nenabled = false\n", encoding="utf-8")

    config = Config.load(config_path)

    assert config.llm.enabled is False


def test_environment_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "ybook.toml"
    config_path.write_text(
        """
channel = "https://www.youtube.com/@fromfile"

[llm]
base_url = "https://old.example/v1"
model = "old-model"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("YBOOK_CHANNEL", "https://www.youtube.com/@fromenv")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1/")
    monkeypatch.setenv("YBOOK_LLM_MODEL", "new-model")

    config = Config.load(config_path)

    assert config.channel == "https://www.youtube.com/@fromenv"
    assert config.llm.base_url == "http://localhost:11434/v1"
    assert config.llm.model == "new-model"


def test_absolute_paths_are_not_joined_to_config_dir(tmp_path: Path) -> None:
    absolute_data = tmp_path / "elsewhere"
    config_path = tmp_path / "ybook.toml"
    config_path.write_text(
        f"""
[paths]
data_dir = "{absolute_data.as_posix()}"
output_dir = "{tmp_path.as_posix()}"
""",
        encoding="utf-8",
    )
    config = Config.load(config_path)
    assert config.paths.data_dir == absolute_data.resolve()


def test_api_key_resolution_and_missing_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "ybook.toml"
    config_path.write_text(
        """
[youtube]
api_key_env = "CUSTOM_YOUTUBE_KEY"

[llm]
api_key_env = "CUSTOM_LLM_KEY"
""",
        encoding="utf-8",
    )
    config = Config.load(config_path)

    with pytest.raises(ConfigError, match="CUSTOM_YOUTUBE_KEY"):
        config.youtube_api_key()
    monkeypatch.setenv("CUSTOM_YOUTUBE_KEY", "yt-secret")
    assert config.youtube_api_key() == "yt-secret"

    with pytest.raises(ConfigError, match="CUSTOM_LLM_KEY"):
        config.llm_api_key()
    monkeypatch.setenv("YBOOK_LLM_API_KEY", "fallback-llm")
    assert config.llm_api_key() == "fallback-llm"


def test_youtube_api_key_falls_back_to_default_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "ybook.toml"
    config_path.write_text(
        """
[youtube]
api_key_env = "CUSTOM_YOUTUBE_KEY"
""",
        encoding="utf-8",
    )
    config = Config.load(config_path)
    monkeypatch.setenv("YOUTUBE_API_KEY", "default-key")
    assert config.youtube_api_key() == "default-key"


def test_invalid_config_reports_validation_error(tmp_path: Path) -> None:
    config_path = tmp_path / "ybook.toml"
    config_path.write_text(
        """
[transcripts]
languages = []
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Invalid configuration"):
        Config.load(config_path)


def test_annotation_max_words_has_a_minimum(tmp_path: Path) -> None:
    config_path = tmp_path / "ybook.toml"
    config_path.write_text(
        """
[llm]
annotation_max_words = 999
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="annotation_max_words"):
        Config.load(config_path)


def test_missing_explicit_config_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        Config.load(tmp_path / "missing.toml")


def test_invalid_toml_is_an_error(tmp_path: Path) -> None:
    config_path = tmp_path / "ybook.toml"
    config_path.write_text("[llm\nbase_url = 1", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        Config.load(config_path)
