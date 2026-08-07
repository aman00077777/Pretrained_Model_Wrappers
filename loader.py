"""
Task P1 — Pretrained Model Loader

Maps friendly short-names (e.g. "clip", "bert-base") to Hugging Face model IDs
and their corresponding FUSION encoder wrapper class, then instantiates the
wrapper via its ``from_pretrained()`` classmethod.

Consumers::

    from fusion.models.pretrained import load_pretrained

    encoder = load_pretrained("clip", cache_dir="/tmp/hf_cache")
"""
from __future__ import annotations

from typing import Any, Optional, Tuple, Type

from fusion.encoders.base import BaseEncoder
from fusion.encoders.registry import Registry
from fusion.exceptions import FusionError
from fusion.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PretrainedModelError
# ---------------------------------------------------------------------------

class PretrainedModelError(FusionError):
    """Raised when a pretrained model cannot be loaded or resolved.

    Args:
        message (str): Human-readable description of what went wrong.
        details (Optional[dict]): Structured diagnostic payload
            (e.g. ``{"name": "clip", "available": ["bert-base", ...]}``).
    """

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message, details)


# ---------------------------------------------------------------------------
# Registry instance
# ---------------------------------------------------------------------------

PRETRAINED_REGISTRY: Registry = Registry("pretrained")


# ---------------------------------------------------------------------------
# Registry population
#
# Each entry maps a friendly name to a zero-argument factory that returns a
# (hf_model_id, wrapper_class) tuple.  Using factories keeps the imports lazy
# so that missing optional dependencies don't break the import of this module
# when only a subset of encoders is installed.
# ---------------------------------------------------------------------------

def _register_defaults() -> None:
    """Populate PRETRAINED_REGISTRY with all built-in friendly-name entries."""

    # ---- Vision encoders ---------------------------------------------------
    @PRETRAINED_REGISTRY.register("clip")
    def _clip() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.vision.clip_vision import CLIPVisionEncoder
        return ("openai/clip-vit-base-patch32", CLIPVisionEncoder)

    @PRETRAINED_REGISTRY.register("clip-large")
    def _clip_large() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.vision.clip_vision import CLIPVisionEncoder
        return ("openai/clip-vit-large-patch14", CLIPVisionEncoder)

    @PRETRAINED_REGISTRY.register("dinov2-base")
    def _dinov2_base() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.vision.dinov2 import DINOv2Encoder
        return ("facebook/dinov2-base", DINOv2Encoder)

    @PRETRAINED_REGISTRY.register("dinov2-large")
    def _dinov2_large() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.vision.dinov2 import DINOv2Encoder
        return ("facebook/dinov2-large", DINOv2Encoder)

    @PRETRAINED_REGISTRY.register("vit-base")
    def _vit_base() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.vision.vit import ViTEncoder
        return ("google/vit-base-patch16-224", ViTEncoder)

    @PRETRAINED_REGISTRY.register("efficientnet-b0")
    def _efficientnet_b0() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.vision.efficientnet import EfficientNetEncoder
        return ("google/efficientnet-b0", EfficientNetEncoder)

    @PRETRAINED_REGISTRY.register("resnet-50")
    def _resnet50() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.vision.resnet import ResNetEncoder
        return ("microsoft/resnet-50", ResNetEncoder)

    @PRETRAINED_REGISTRY.register("convnext-base")
    def _convnext_base() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.vision.convnext import ConvNextEncoder
        return ("facebook/convnext-base-224-22k", ConvNextEncoder)

    @PRETRAINED_REGISTRY.register("swin-base")
    def _swin_base() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.vision.swin import SwinEncoder
        return ("microsoft/swin-base-patch4-window7-224", SwinEncoder)

    # ---- Language encoders -------------------------------------------------
    @PRETRAINED_REGISTRY.register("bert-base")
    def _bert_base() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.language.bert import BERTEncoder
        return ("bert-base-uncased", BERTEncoder)

    @PRETRAINED_REGISTRY.register("bert-large")
    def _bert_large() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.language.bert import BERTEncoder
        return ("bert-large-uncased", BERTEncoder)

    @PRETRAINED_REGISTRY.register("roberta-base")
    def _roberta_base() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.language.roberta import RoBERTaEncoder
        return ("roberta-base", RoBERTaEncoder)

    @PRETRAINED_REGISTRY.register("clip-text")
    def _clip_text() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.language.clip_text import CLIPTextEncoder
        return ("openai/clip-vit-base-patch32", CLIPTextEncoder)

    @PRETRAINED_REGISTRY.register("t5-base")
    def _t5_base() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.language.t5 import T5Encoder
        return ("t5-base", T5Encoder)

    @PRETRAINED_REGISTRY.register("gpt2")
    def _gpt2() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.language.gpt import GPTEncoder
        return ("gpt2", GPTEncoder)

    @PRETRAINED_REGISTRY.register("xlm-roberta-base")
    def _xlm_roberta_base() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.language.xlm import XLMEncoder
        return ("xlm-roberta-base", XLMEncoder)

    # ---- Audio encoders ----------------------------------------------------
    @PRETRAINED_REGISTRY.register("wav2vec2-base")
    def _wav2vec2_base() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.audio.wav2vec2 import Wav2Vec2Encoder
        return ("facebook/wav2vec2-base", Wav2Vec2Encoder)

    @PRETRAINED_REGISTRY.register("hubert-base")
    def _hubert_base() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.audio.hubert import HuBERTEncoder
        return ("facebook/hubert-base-ls960", HuBERTEncoder)

    @PRETRAINED_REGISTRY.register("whisper-base")
    def _whisper_base() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.audio.whisper import WhisperEncoder
        return ("openai/whisper-base", WhisperEncoder)

    @PRETRAINED_REGISTRY.register("clap-base")
    def _clap_base() -> Tuple[str, Type[BaseEncoder]]:
        from fusion.encoders.audio.clap import CLAPEncoder
        return ("laion/clap-htsat-unfused", CLAPEncoder)


_register_defaults()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pretrained(
    name: str,
    cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> BaseEncoder:
    """Load a pretrained encoder by its registered friendly name.

    Looks up *name* in :data:`PRETRAINED_REGISTRY`, resolves the Hugging Face
    model ID and wrapper class, then delegates to
    ``wrapper_class.from_pretrained(model_id, cache_dir=cache_dir, **kwargs)``.

    Args:
        name: Friendly registry name, e.g. ``"clip"``, ``"bert-base"``,
            ``"dinov2-base"``.  Call ``PRETRAINED_REGISTRY.list()`` for the
            full list of registered names.
        cache_dir: Optional path where HF hub caches downloaded weights.
            Forwarded verbatim to the underlying ``from_pretrained`` call.
        **kwargs: Extra keyword arguments forwarded to the wrapper's
            ``from_pretrained`` classmethod (e.g. ``torch_dtype``,
            ``device_map``).

    Returns:
        A fully-initialised :class:`~fusion.encoders.base.BaseEncoder`
        instance wrapping the requested pretrained model.

    Raises:
        PretrainedModelError: If *name* is not registered, or if the
            underlying ``from_pretrained`` call fails.

    Example::

        encoder = load_pretrained("clip", cache_dir="/tmp/hf_cache")
        encoder = load_pretrained("bert-base", torch_dtype=torch.float16)
    """
    logger.info("event=load_pretrained_started name=%s", name)

    # ---- Registry look-up --------------------------------------------------
    if name not in PRETRAINED_REGISTRY:
        available = PRETRAINED_REGISTRY.list()
        raise PretrainedModelError(
            f"Unknown pretrained model name: '{name}'. "
            f"Available names: {available}",
            details={"name": name, "available": available},
        )

    # Each registry entry is a zero-argument factory returning (hf_model_id, cls).
    factory = PRETRAINED_REGISTRY.get(name)
    hf_model_id, wrapper_cls = factory()

    logger.debug(
        "event=registry_resolved name=%s hf_model_id=%s wrapper=%s",
        name, hf_model_id, wrapper_cls.__name__,
    )

    # ---- Instantiation via from_pretrained ---------------------------------
    try:
        if cache_dir is not None:
            encoder = wrapper_cls.from_pretrained(
                hf_model_id, cache_dir=cache_dir, **kwargs
            )
        else:
            encoder = wrapper_cls.from_pretrained(hf_model_id, **kwargs)
    except Exception as exc:
        logger.error(
            "event=load_pretrained_failed name=%s hf_model_id=%s error_type=%s",
            name, hf_model_id, type(exc).__name__,
        )
        raise PretrainedModelError(
            f"Failed to load pretrained model '{name}' "
            f"(HF id: {hf_model_id}): {exc}",
            details={
                "name": name,
                "hf_model_id": hf_model_id,
                "wrapper": wrapper_cls.__name__,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        ) from exc

    logger.info(
        "event=load_pretrained_complete name=%s hf_model_id=%s wrapper=%s",
        name, hf_model_id, wrapper_cls.__name__,
    )
    return encoder


__all__ = [
    "PRETRAINED_REGISTRY",
    "PretrainedModelError",
    "load_pretrained",
]
