"""
fusion/utils/logging.py — STUB. See fusion/__init__.py for why this exists.

``get_logger`` is imported by loader.py / downloader.py / converter.py
as ``from fusion.utils.logging import get_logger``.

``log_event`` is imported by every Phase 2 encoder file (base.py's own
style — ``logger.info("event=X key=%s", val)`` — implies the same
"event=<name> key=value ..." structured format; this makes that pattern
callable as a single helper instead of hand-formatted at each call site).
Not included in the Encoders.zip upload — reverse-engineered purely from
call sites like ``log_event(logger, logging.INFO, "model_loading",
encoder="bert", model=model_name)``. Replace with the real one if it
turns out to live somewhere else.
"""

from __future__ import annotations

import logging
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Return a standard library logger configured with a basic handler.

    Args:
        name (str): Usually ``__name__`` of the calling module.

    Returns:
        logging.Logger: A logger; safe to call repeatedly for the same name.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit one structured log line: ``event=<event> key=value key2=value2 ...``.

    Args:
        logger: Logger to write to (usually from :func:`get_logger`).
        level: A ``logging`` level constant, e.g. ``logging.INFO``.
        event: Short event name, e.g. ``"model_loading"``.
        **fields: Arbitrary key/value context appended to the line.
    """
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    message = f"event={event}" + (f" {suffix}" if suffix else "")
    logger.log(level, message)
