"""
Logging redaction helpers.
Redacts sensitive tokens/keys from log messages.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable


_PATTERNS: Iterable[tuple[re.Pattern, str]] = (
    # Telegram bot token in URL: /bot<token>/
    (re.compile(r"bot\d+:[A-Za-z0-9_-]{20,}"), "bot[REDACTED]"),
    # Authorization: Bearer <token>
    (re.compile(r"(Bearer\s+)([A-Za-z0-9\-\._]+)"), r"\1[REDACTED]"),
    # Generic access token key/value
    (re.compile(r"(?i)(access_token|token)\s*[:=]\s*([A-Za-z0-9\-\._]+)"), r"\1=[REDACTED]"),
    # Upstox API key/secret headers or config output
    (re.compile(r"(?i)(api[_-]?key|api[_-]?secret)\s*[:=]\s*([A-Za-z0-9\-\._]+)"), r"\1=[REDACTED]"),
)


def redact_message(message: str) -> str:
    redacted = message
    for pattern, replacement in _PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


class RedactingFilter(logging.Filter):
    """Filter that redacts sensitive data from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
            redacted = redact_message(message)
            record.msg = redacted
            record.args = ()
        except Exception:
            # If redaction fails, allow log through unmodified
            pass
        return True


def install_redaction_filter() -> None:
    root = logging.getLogger()
    # Avoid duplicate filters
    for existing in root.filters:
        if isinstance(existing, RedactingFilter):
            return
    root.addFilter(RedactingFilter())
