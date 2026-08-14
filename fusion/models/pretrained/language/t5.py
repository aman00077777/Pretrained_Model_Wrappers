"""
T5 language encoder wrapper.

Usage:
    from fusion.models.pretrained.language.t5 import T5Encoder
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import AutoTokenizer, T5EncoderModel

from fusion.constants import Modality
from fusion.encoders.base import BaseEncoder
from fusion.encoders.modal_tensor import ModalTensor
from fusion.encoders.registry import register_encoder


@register_encoder("t5")
class T5Encoder(BaseEncoder):
    """Wraps the T5 encoder stack as a FUSION language encoder.

    Only the encoder half of T5 is used; the decoder is never instantiated,
    keeping memory footprint low and inference fast.
    """

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
        model_name_or_path: str = "t5-base",
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> "T5Encoder":
        """Load the T5 encoder and tokenizer.

        Uses ``T5EncoderModel`` which loads only the encoder stack —
        no decoder weights are downloaded.

        Args:
            model_name_or_path: Hugging Face Hub ID or local path.
            cache_dir: Optional local cache directory.
            **kwargs: Forwarded to T5EncoderModel.from_pretrained.

        Returns:
            An initialized T5Encoder.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )

        model = T5EncoderModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            **kwargs,
        )

        output_dim = model.config.d_model

        return cls(
            model=model,
            tokenizer=tokenizer,
            output_dim=output_dim,
        )

    def encode(
        self,
        inputs: list[str] | str,
        max_length: int = 128,
    ) -> ModalTensor:
        """Encode a batch of strings into T5 encoder embeddings.

        Mean-pools the encoder hidden states over the token dimension to
        produce one fixed-size vector per input string.

        Args:
            inputs: A string or list of strings to encode.
            max_length: Maximum token sequence length.

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

        hidden_states = outputs.last_hidden_state  # (B, T, D)

        # Masked mean pooling — ignore padding tokens.
        attention_mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (
            (hidden_states * attention_mask).sum(dim=1)
            / attention_mask.sum(dim=1).clamp(min=1e-9)
        )

        return ModalTensor(data=pooled, modality=Modality.LANGUAGE)

    def get_output_dim(self) -> int:
        """Return the dimensionality of the language embedding."""
        return self._output_dim
