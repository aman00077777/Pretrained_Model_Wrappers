"""
fusion.models.pretrained — pretrained model wrappers package.

Public API:

    from fusion.models.pretrained import load_pretrained
    from fusion.models.pretrained import download_pretrained
    from fusion.models.pretrained import convert_to_fusion_format
    from fusion.models.pretrained import PRETRAINED_REGISTRY

Language encoder wrappers (fusion.models.pretrained.language):

    from fusion.models.pretrained.language import (
        CLIPTextEncoder,
        BERTEncoder,
        RoBERTaEncoder,
        T5Encoder,
        GPTEncoder,
        XLMEncoder,
    )

Submodule imports for vision / audio / multimodal will be uncommented
once those modules are merged by their respective owners.
"""

from fusion.models.pretrained.loader import (
    PRETRAINED_REGISTRY,
    PretrainedModelError,
    load_pretrained,
)
from fusion.models.pretrained.downloader import (
    DownloadError,
    download_pretrained,
)
from fusion.models.pretrained.converter import convert_to_fusion_format

__all__ = [
    # Core loader
    "PRETRAINED_REGISTRY",
    "PretrainedModelError",
    "load_pretrained",
    # Downloader
    "DownloadError",
    "download_pretrained",
    # Converter
    "convert_to_fusion_format",
]
