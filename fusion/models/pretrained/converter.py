"""
Task P3 — Pretrained Model Converter

Wraps an arbitrary ``nn.Module`` in the appropriate FUSION Custom*Encoder
shell based on the target modality, so that any pretrained backbone
participates in the standard BaseEncoder contract (encode, freeze,
get_output_dim, etc.).

Consumers::

    from fusion.models.pretrained import convert_to_fusion_format
    from fusion.constants import Modality

    encoder = convert_to_fusion_format(my_model, Modality.VISION, output_dim=768)
"""
from __future__ import annotations

from typing import Optional

import torch.nn as nn

from fusion.constants import Modality
from fusion.encoders.base import BaseEncoder
from fusion.exceptions import ModalityError
from fusion.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_to_fusion_format(
    model: nn.Module,
    modality: Modality,
    output_dim: int,
    config: Optional[dict] = None,
) -> BaseEncoder:
    """Wrap *model* in the FUSION Custom*Encoder for *modality*.

    Selects the correct wrapper class based on *modality*:

    * :data:`~fusion.constants.Modality.VISION`   →
      :class:`~fusion.encoders.vision.custom.CustomVisionEncoder`
    * :data:`~fusion.constants.Modality.LANGUAGE` →
      :class:`~fusion.encoders.language.custom.CustomLanguageEncoder`
    * :data:`~fusion.constants.Modality.AUDIO`    →
      :class:`~fusion.encoders.audio.custom.CustomAudioEncoder`

    Args:
        model: Any :class:`torch.nn.Module` whose ``forward()`` accepts the
            modality's canonical input and returns a tensor of shape
            ``[batch, output_dim]``.
        modality: One of the :class:`~fusion.constants.Modality` enum members.
        output_dim: Dimensionality of the pooled embedding produced by
            *model*.  Must be a positive integer.
        config: Optional configuration dict forwarded to the wrapper's
            ``__init__``.  Ignored by ``CustomLanguageEncoder`` (which does
            not accept a ``config`` kwarg); kept for forward-compatibility.

    Returns:
        A fully-initialised :class:`~fusion.encoders.base.BaseEncoder`
        that satisfies ``encode()``, ``get_output_dim()``, ``freeze()``,
        ``unfreeze()``, and ``freeze_except_last_n()``.

    Raises:
        ModalityError: If *modality* is not one of the three supported values.
        TypeError: If *model* is not an ``nn.Module`` (propagated from the
            underlying wrapper constructor).
        ValueError: If *output_dim* ≤ 0 (propagated from the underlying
            wrapper constructor).

    Example::

        import torch.nn as nn
        from fusion.constants import Modality
        from fusion.models.pretrained import convert_to_fusion_format

        backbone = nn.Linear(512, 768)
        encoder = convert_to_fusion_format(backbone, Modality.VISION, 768)
        print(encoder.get_output_dim())  # 768
    """
    logger.info(
        "event=convert_started modality=%s output_dim=%d",
        modality, output_dim,
    )

    try:
        if modality is Modality.VISION or modality == Modality.VISION:
            from fusion.encoders.vision.custom import CustomVisionEncoder
            encoder: BaseEncoder = CustomVisionEncoder(
                module=model, output_dim=output_dim, config=config
            )

        elif modality is Modality.LANGUAGE or modality == Modality.LANGUAGE:
            from fusion.encoders.language.custom import CustomLanguageEncoder
            # CustomLanguageEncoder does not accept a config kwarg.
            encoder = CustomLanguageEncoder(module=model, output_dim=output_dim)

        elif modality is Modality.AUDIO or modality == Modality.AUDIO:
            from fusion.encoders.audio.custom import CustomAudioEncoder
            encoder = CustomAudioEncoder(
                module=model, output_dim=output_dim, config=config
            )

        else:
            supported = [m.value for m in Modality]
            raise ModalityError(
                f"Unsupported modality: '{modality}'. "
                f"Supported modalities: {supported}",
                details={
                    "received": str(modality),
                    "supported": supported,
                },
            )

    except ModalityError:
        logger.error(
            "event=convert_failed modality=%s error_type=ModalityError",
            modality,
        )
        raise
    except Exception as exc:
        logger.error(
            "event=convert_failed modality=%s error_type=%s",
            modality, type(exc).__name__,
        )
        raise

    logger.info(
        "event=convert_complete modality=%s wrapper=%s output_dim=%d",
        modality, type(encoder).__name__, output_dim,
    )
    return encoder


__all__ = ["convert_to_fusion_format"]
