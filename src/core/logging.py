"""Structured logging helpers."""
from __future__ import annotations

import logging
import sys
import time
from typing import Any, Optional

import orjson

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Format log records as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = record.stack_info
        if extra := getattr(record, "extra_fields", None):
            payload.update(extra)
        return orjson.dumps(payload).decode("utf-8")


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once for the process."""

    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), handlers=[handler])
    _CONFIGURED = True


def get_logger(name: str, *, level: Optional[str] = None) -> logging.Logger:
    """Return a module-specific logger."""

    logger = logging.getLogger(name)
    if level:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
