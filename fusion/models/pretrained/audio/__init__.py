"""fusion.models.pretrained.audio — audio encoder wrappers.

Exported classes:
    - ASTEncoder
    - CLAPEncoder
    - HuBERTEncoder
    - Wav2Vec2Encoder
    - WhisperEncoder
"""

from fusion.models.pretrained.audio.ast import ASTEncoder
from fusion.models.pretrained.audio.clap import CLAPEncoder
from fusion.models.pretrained.audio.hubert import HuBERTEncoder
from fusion.models.pretrained.audio.wav2vec2 import Wav2Vec2Encoder
from fusion.models.pretrained.audio.whisper import WhisperEncoder

__all__ = [
    "ASTEncoder",
    "CLAPEncoder",
    "HuBERTEncoder",
    "Wav2Vec2Encoder",
    "WhisperEncoder",
]
