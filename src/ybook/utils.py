"""Small shared helpers: logging, atomic writes, hashing, and redaction."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9]+")


def setup_logging(verbose: bool) -> None:
    """Configure stdlib logging for CLI use without enabling noisy HTTP logs."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    for noisy_logger in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def utc_now() -> datetime:
    """Current timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def utc_now_iso() -> str:
    """Current UTC timestamp in a sortable ISO-8601 form."""

    return utc_now().isoformat(timespec="seconds")


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write text atomically by replacing a temporary file in the same directory."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return target


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Serialize ``payload`` as pretty UTF-8 JSON using an atomic write."""

    return atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def sha256_text(value: str) -> str:
    """Hex SHA-256 digest of a UTF-8 string."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def redact_secret(text: str, secret: str) -> str:
    """Replace a secret with a marker, guarding logs and error messages."""

    if not secret:
        return text
    return text.replace(secret, "[redacted]")


def slugify(value: str, *, max_length: int = 80) -> str:
    """Create a conservative ASCII-ish filename slug."""

    slug = _SLUG_PATTERN.sub("-", value.strip().lower()).strip("-")
    return slug[:max_length].rstrip("-") or "item"


def truncate(value: str, limit: int) -> str:
    """Truncate at a word boundary when possible, adding an ellipsis."""

    if len(value) <= limit:
        return value
    head = value[: max(0, limit - 1)]
    if " " in head:
        head = head.rsplit(" ", 1)[0]
    return head.rstrip() + "…"


def iso_date(value: str | None) -> str:
    """Best-effort ISO date extraction from a YouTube timestamp."""

    if not value:
        return ""
    match = re.match(r"(\d{4}-\d{2}-\d{2})", value)
    return match.group(1) if match else ""


def language_matches(stored: str | None, requested: Iterable[str]) -> bool:
    """Return whether a stored language code satisfies a requested language.

    A stored regional/generated variant (``en-US``, ``en-orig``) satisfies a
    request for its base language (``en``). Missing stored languages never match.
    """

    stored_code = (stored or "").strip().casefold()
    if not stored_code:
        return False
    for language in requested:
        requested_code = language.strip().casefold()
        if not requested_code:
            continue
        if stored_code == requested_code or stored_code.startswith(f"{requested_code}-"):
            return True
    return False
