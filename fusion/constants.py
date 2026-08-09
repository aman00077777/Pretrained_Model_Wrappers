"""
fusion/constants.py — STUB. See fusion/__init__.py for why this exists.

``Modality`` is referenced by converter.py (``Modality.VISION`` /
``.LANGUAGE`` / ``.AUDIO``) but was not defined anywhere in this repo.
"""

from __future__ import annotations

import enum


class Modality(str, enum.Enum):
    """The data modalities FUSION encoders operate on."""

    VISION = "vision"
    LANGUAGE = "language"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
