"""
fusion/utils/logging.py — STUB. See fusion/__init__.py for why this exists.

``get_logger`` is imported by loader.py / downloader.py / converter.py
as ``from fusion.utils.logging import get_logger``.
"""

from __future__ import annotations

import logging


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
