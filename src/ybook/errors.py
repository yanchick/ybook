"""Project-specific exception hierarchy.

The CLI catches :class:`YbookError` so every expected failure produces a short,
actionable message instead of a traceback.  Unexpected exceptions still surface
as tracebacks, which is intentional: they signal a bug rather than a bad input.
"""

from __future__ import annotations


class YbookError(Exception):
    """Base class for all expected ybook failures."""


class ConfigError(YbookError):
    """Configuration is missing, unreadable, or inconsistent."""


class YouTubeError(YbookError):
    """YouTube Data API request or response failed."""


class ChannelInputError(YouTubeError):
    """The supplied channel URL or handle is not a supported channel reference."""


class QuotaExceededError(YouTubeError):
    """The YouTube Data API daily quota has been exhausted."""


class ChannelNotFoundError(YouTubeError):
    """A channel could not be resolved from the supplied URL or handle."""


class TranscriptError(YbookError):
    """Transcript download failed for a reason worth surfacing."""


class LLMError(YbookError):
    """Base class for OpenAI-compatible API failures."""


class LLMConfigurationError(LLMError):
    """LLM configuration is incomplete or invalid."""


class LLMRequestError(LLMError):
    """The LLM endpoint returned a permanent HTTP or transport error."""


class LLMResponseError(LLMError):
    """The LLM response was malformed or unusable."""


class LLMCleanError(LLMError):
    """A transcript chunk could not be cleaned after retrying."""


class BookBuildError(YbookError):
    """The EPUB cannot be assembled from the current database state."""
