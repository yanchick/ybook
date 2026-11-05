"""Typed application configuration loaded from TOML and environment variables.

The loader is deliberately small and explicit: TOML is read with ``tomllib``,
validated by pydantic, then the few documented environment overrides are
applied.  Relative paths are anchored to the directory containing the config
file, which keeps a project portable when the whole directory is moved.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .errors import ConfigError

ProviderName = Literal["auto", "youtube-transcript-api", "yt-dlp"]
MaxTokensParameter = Literal["max_tokens", "max_completion_tokens"]

DEFAULT_CONFIG_FILENAME = "ybook.toml"


class _Section(BaseModel):
    """Base model for configuration sections; unknown TOML keys are ignored."""

    model_config = ConfigDict(extra="ignore")


class PathsConfig(_Section):
    """Filesystem locations. Relative values are resolved from the config file."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("out")


class YouTubeConfig(_Section):
    """YouTube Data API settings."""

    api_key_env: str = "YOUTUBE_API_KEY"

    @field_validator("api_key_env")
    @classmethod
    def _validate_api_key_env(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("youtube.api_key_env must not be empty")
        return cleaned


class TranscriptsConfig(_Section):
    """Transcript download settings."""

    languages: list[str] = Field(default_factory=lambda: ["en"])
    provider: ProviderName = "auto"
    cookies_from_browser: str = ""
    workers: int = Field(default=3, ge=1)
    sleep_requests: float = Field(default=1.0, ge=0.0)
    timeout_seconds: int = Field(default=180, gt=0)

    @field_validator("languages")
    @classmethod
    def _normalize_languages(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item and item.strip()]
        if not normalized:
            raise ValueError("transcripts.languages must contain at least one language code")
        return normalized

    @field_validator("cookies_from_browser")
    @classmethod
    def _strip_cookies(cls, value: str) -> str:
        return value.strip()


class LLMConfig(_Section):
    """OpenAI-compatible chat-completions settings."""

    enabled: bool = True
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4.1-mini"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8192, gt=0)
    max_words_per_chunk: int = Field(default=2500, ge=1)
    annotation_max_words: int = Field(default=12000, ge=1000)
    workers: int = Field(default=3, ge=1)
    timeout_seconds: float = Field(default=180.0, gt=0.0)
    max_tokens_param: MaxTokensParameter = "max_tokens"
    input_cost_per_million: float = Field(default=0.0, ge=0.0)
    output_cost_per_million: float = Field(default=0.0, ge=0.0)
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        cleaned = value.strip().rstrip("/")
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError("llm.base_url must start with http:// or https://")
        return cleaned

    @field_validator("api_key_env", "model")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be empty")
        return cleaned


class BookConfig(_Section):
    """EPUB metadata and selection settings."""

    title: str = ""
    author: str = ""
    language: str = "en"
    description: str = ""
    group_by_playlist: bool = True
    include_shorts: bool = True
    include_live: bool = True
    annotations: bool = True

    @field_validator("title", "author", "description")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("book.language must not be empty")
        return cleaned


class Config(BaseModel):
    """Complete, validated ybook configuration."""

    model_config = ConfigDict(extra="ignore")

    channel: str = ""
    paths: PathsConfig = Field(default_factory=PathsConfig)
    youtube: YouTubeConfig = Field(default_factory=YouTubeConfig)
    transcripts: TranscriptsConfig = Field(default_factory=TranscriptsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    book: BookConfig = Field(default_factory=BookConfig)

    config_path: Path | None = Field(default=None, exclude=True)
    base_dir: Path = Field(default_factory=Path.cwd, exclude=True)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Load ``ybook.toml`` (or an explicit path) plus environment overrides.

        When ``path`` is omitted, ``./ybook.toml`` is used if it exists.  Without
        a config file all defaults apply and paths are anchored at the current
        working directory.
        """

        config_path = _resolve_config_path(path)
        base_dir = config_path.parent if config_path is not None else Path.cwd()
        raw: dict[str, Any] = {}

        if config_path is not None:
            try:
                with config_path.open("rb") as handle:
                    loaded = tomllib.load(handle)
            except OSError as exc:
                raise ConfigError(f"Cannot read config file {config_path}: {exc}") from exc
            except tomllib.TOMLDecodeError as exc:
                raise ConfigError(f"Invalid TOML in {config_path}: {exc}") from exc
            if not isinstance(loaded, dict):
                raise ConfigError(f"Config file {config_path} must contain a TOML table")
            raw = loaded

        raw = _apply_environment_overrides(raw)

        try:
            config = cls.model_validate(raw)
        except ValidationError as exc:
            source = str(config_path) if config_path else "defaults"
            raise ConfigError(f"Invalid configuration ({source}):\n{exc}") from exc

        base_dir = base_dir.resolve()
        config.config_path = config_path
        config.base_dir = base_dir
        config.paths.data_dir = _resolve_path(config.paths.data_dir, base_dir)
        config.paths.output_dir = _resolve_path(config.paths.output_dir, base_dir)
        return config

    def ensure_directories(self) -> None:
        """Create state, transcript, cleaned, cache, API, and output directories."""

        for directory in (
            self.paths.data_dir,
            self.transcripts_dir,
            self.cleaned_dir,
            self.llm_cache_dir,
            self.api_dir,
            self.paths.output_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        """SQLite state database path."""

        return self.paths.data_dir / "state.sqlite"

    @property
    def transcripts_dir(self) -> Path:
        """Directory for raw and normalized transcript files."""

        return self.paths.data_dir / "transcripts"

    @property
    def cleaned_dir(self) -> Path:
        """Directory for cleaned chapter text and annotations."""

        return self.paths.data_dir / "cleaned"

    @property
    def cache_dir(self) -> Path:
        """Root cache directory."""

        return self.paths.data_dir / "cache"

    @property
    def llm_cache_dir(self) -> Path:
        """Per-chunk LLM response cache."""

        return self.cache_dir / "llm"

    @property
    def api_dir(self) -> Path:
        """Directory for sanitized YouTube API debug snapshots."""

        return self.paths.data_dir / "api"

    def youtube_api_key(self) -> str:
        """Return the YouTube API key or raise a clear configuration error."""

        env_name = self.youtube.api_key_env
        key = os.getenv(env_name, "").strip()
        if not key and env_name != "YOUTUBE_API_KEY":
            key = os.getenv("YOUTUBE_API_KEY", "").strip()
        if not key:
            raise ConfigError(
                f"{env_name} is not set. Create an API key in Google Cloud, enable "
                "YouTube Data API v3, and export it (or put it in .env)."
            )
        return key

    def llm_api_key(self) -> str:
        """Return the LLM API key or raise a clear configuration error."""

        env_name = self.llm.api_key_env
        key = os.getenv(env_name, "").strip()
        if not key:
            key = os.getenv("YBOOK_LLM_API_KEY", "").strip()
        if not key:
            raise ConfigError(
                f"{env_name} is not set. Provide an API key for the configured "
                "OpenAI-compatible endpoint (YBOOK_LLM_API_KEY also works as a fallback)."
            )
        return key


def _resolve_config_path(path: str | Path | None) -> Path | None:
    if path is not None:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if candidate.is_dir():
            candidate = candidate / DEFAULT_CONFIG_FILENAME
        if not candidate.exists():
            raise ConfigError(f"Config file not found: {candidate}")
        return candidate.resolve()

    default = Path.cwd() / DEFAULT_CONFIG_FILENAME
    return default.resolve() if default.is_file() else None


def _resolve_path(value: Path, base_dir: Path) -> Path:
    expanded = value.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (base_dir / expanded).resolve()


def _apply_environment_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``raw`` with documented environment overrides applied."""

    result = dict(raw)

    channel = os.getenv("YBOOK_CHANNEL", "").strip()
    if channel:
        result["channel"] = channel

    llm_raw = result.get("llm")
    if llm_raw is not None and not isinstance(llm_raw, dict):
        # Leave the invalid value in place so pydantic reports the real problem.
        return result

    llm = dict(llm_raw) if isinstance(llm_raw, dict) else {}

    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        llm["base_url"] = base_url

    model = os.getenv("YBOOK_LLM_MODEL", "").strip()
    if model:
        llm["model"] = model

    result["llm"] = llm
    return result
