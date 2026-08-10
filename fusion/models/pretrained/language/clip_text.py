"""
CLIP text-tower language encoder wrapper.

Usage:
    from fusion.encoders.language import CLIPTextEncoder

Uses ``CLIPTextModelWithProjection`` rather than the bare
``CLIPTextModel`` so the output lands in CLIP's joint image-text
embedding space (``.text_embeds``) — this is what makes it comparable
against ``CLIPVisionEncoder`` (vision/clip_vision.py, Gauri's file) for
similarity scoring. HF's implementation already pools internally via the
EOS-token position, so no separate pooling helper is needed here.
"""

from __future__ import annotations

from typing import Any, List, Optional

from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from fusion.constants import Modality
from fusion.encoders.base import BaseEncoder
from fusion.types import ModalTensor


class CLIPTextEncoder(BaseEncoder):
    """Wraps CLIP's text tower as a FUSION language encoder.

    Args:
        model: Loaded ``CLIPTextModelWithProjection``.
        tokenizer: The matching ``CLIPTokenizer``.
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        super().__init__(output_dim=model.config.projection_dim)
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "openai/clip-vit-base-patch32",
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> "CLIPTextEncoder":
        """Load CLIP's text tower + tokenizer from Hugging Face.

        Args:
            model_name_or_path: HF Hub id or local path, e.g.
                ``"openai/clip-vit-base-patch32"``.
            cache_dir: Optional local cache directory for downloaded weights.
            **kwargs: Forwarded to ``CLIPTextModelWithProjection.from_pretrained``.

        Returns:
            An initialised CLIPTextEncoder.
        """
        tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        model = CLIPTextModelWithProjection.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, **kwargs
        )
        return cls(model=model, tokenizer=tokenizer)

    def encode(self, texts: List[str], max_length: int = 77) -> ModalTensor:
        """Encode a batch of raw text into CLIP's joint embedding space.

        Args:
            texts: List of strings.
            max_length: Truncation length; CLIP's own tokenizer is
                trained with a 77-token context window, so this default
                differs from the other 5 wrappers' 512.

        Returns:
            ModalTensor wrapping a ``(B, output_dim)`` embedding tensor.
        """
        device = next(self.model.parameters()).device
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        ).to(device)
        outputs = self.model(**batch)
        return ModalTensor(tensor=outputs.text_embeds, modality=Modality.LANGUAGE)
