"""
fusion/exceptions.py — STUB. See fusion/__init__.py for why this exists.

Base exception hierarchy, inferred from usage in loader.py / downloader.py
/ converter.py (``PretrainedModelError(FusionError)``,
``DownloadError(FusionError)``, ``ModalityError`` — each constructed with
a message and an optional structured ``details`` dict).
"""

from __future__ import annotations

from typing import Optional


class FusionError(Exception):
    """Base class for all FUSION framework exceptions.

    Args:
        message (str): Human-readable description of what went wrong.
        details (Optional[dict]): Structured diagnostic payload.
    """

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ModalityError(FusionError):
    """Raised when an unsupported or invalid modality is used."""
