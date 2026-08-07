"""
Task P0 — fusion/models/pretrained package init

Re-exports the three public entry-points (load_pretrained,
download_pretrained, convert_to_fusion_format) and the shared registry
(PRETRAINED_REGISTRY) so teammates can use a single import path:

    from fusion.models.pretrained import load_pretrained, download_pretrained
    from fusion.models.pretrained import convert_to_fusion_format
    from fusion.models.pretrained import PRETRAINED_REGISTRY

# -----------------------------------------------------------------------
# Submodule imports — to be uncommented by submodule owners once merged:
# -----------------------------------------------------------------------

# Vision wrappers (owner: <vision-team>)
# from fusion.models.pretrained.vision import (
#     CLIPVisionWrapper,
#     DINOv2Wrapper,
#     ViTWrapper,
#     EfficientNetWrapper,
#     ResNetWrapper,
#     ConvNextWrapper,
#     SwinWrapper,
#     BEiTWrapper,
# )

# Language wrappers (owner: <language-team>)
# from fusion.models.pretrained.language import (
#     CLIPTextWrapper,
#     BERTWrapper,
#     RoBERTaWrapper,
#     T5Wrapper,
#     GPT2Wrapper,
#     XLMWrapper,
# )

# Audio wrappers (owner: <audio-team>)
# from fusion.models.pretrained.audio import (
#     Wav2Vec2Wrapper,
#     HuBERTWrapper,
#     WhisperWrapper,
#     CLAPWrapper,
# )

# Multimodal wrappers (owner: <multimodal-team>)
# from fusion.models.pretrained.multimodal import (
#     CLIPWrapper,
#     ALIGNWrapper,
#     ImageBindWrapper,
#     BLIPWrapper,
# )
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
