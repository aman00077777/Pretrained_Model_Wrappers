"""Shared constants used across the FUSION framework."""
from enum import Enum


class Modality(Enum):
    VISION = "vision"
    LANGUAGE = "language"
    AUDIO = "audio"