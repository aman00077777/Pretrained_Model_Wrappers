"""
CLIP text encoder wrapper.

Usage:
    from fusion.models.pretrained.language.clip_text import CLIPTextEncoder
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import CLIPTextModel, CLIPTokenizer

from fusion.constants import Modality
from fusion.encoders.base import BaseEncoder
from fusion.encoders.modal_tensor import ModalTensor
from fusion.encoders.registry import register_encoder


@register_encoder("clip_text")
class CLIPTextEncoder(BaseEncoder):
    """Wraps the CLIP text tower as a FUSION language encoder."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        output_dim: int,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)

        if output_dim <= 0:
            raise ValueError(
                f"`output_dim` must be positive, got {output_dim}"
            )

        self.model = model
        self.tokenizer = tokenizer
        self._output_dim = int(output_dim)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "openai/clip-vit-base-patch32",
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> "CLIPTextEncoder":
        """Load the CLIP text model and tokenizer.

        Args:
            model_name_or_path: Hugging Face Hub ID or local path.
            cache_dir: Optional local cache directory.
            **kwargs: Forwarded to CLIPTextModel.from_pretrained.

        Returns:
            An initialized CLIPTextEncoder.
        """
        tokenizer = CLIPTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )

        model = CLIPTextModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            **kwargs,
        )

        output_dim = model.config.hidden_size

        return cls(
            model=model,
            tokenizer=tokenizer,
            output_dim=output_dim,
        )

    def encode(
        self,
        inputs: list[str] | str,
        max_length: int = 77,
    ) -> ModalTensor:
        """Encode a batch of strings into CLIP text embeddings.

        CLIP uses the embedding at the [EOS] position (the last token)
        as the pooled sentence representation.

        Args:
            inputs: A string or list of strings to encode.
            max_length: Maximum token sequence length (CLIP default: 77).

        Returns:
            ModalTensor of shape (B, output_dim) tagged as LANGUAGE.
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        if not inputs:
            raise ValueError("Received an empty list of text inputs.")

        device = next(self.model.parameters()).device

        batch = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = self.model(**batch)

        # CLIP pools at the EOS token position.
        pooled = outputs.pooler_output

        return ModalTensor(data=pooled, modality=Modality.LANGUAGE)

    def get_output_dim(self) -> int:
        """Return the dimensionality of the text embedding."""
        return self._output_dim
