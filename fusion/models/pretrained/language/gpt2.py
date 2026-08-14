"""
GPT-2 language encoder wrapper.

Usage:
    from fusion.models.pretrained.language.gpt2 import GPT2Encoder
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import AutoTokenizer, GPT2Model

from fusion.constants import Modality
from fusion.encoders.base import BaseEncoder
from fusion.encoders.modal_tensor import ModalTensor
from fusion.encoders.registry import register_encoder


@register_encoder("gpt2")
class GPT2Encoder(BaseEncoder):
    """Wraps a Hugging Face GPT-2 model as a FUSION language encoder.

    GPT-2 is a decoder-only model with no dedicated pooling head.
    The last non-padding token's hidden state is used as the sentence
    representation (consistent with the causal LM convention).
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
        model_name_or_path: str = "gpt2",
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> "GPT2Encoder":
        """Load a pretrained GPT-2 model and tokenizer.

        Args:
            model_name_or_path: Hugging Face Hub ID or local path.
            cache_dir: Optional local cache directory.
            **kwargs: Forwarded to GPT2Model.from_pretrained.

        Returns:
            An initialized GPT2Encoder.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )

        # GPT-2's tokenizer has no default pad token — use EOS.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = GPT2Model.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            **kwargs,
        )

        output_dim = model.config.n_embd

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
        """Encode a batch of strings into GPT-2 embeddings.

        Uses the hidden state of the last non-padding token as the pooled
        sentence representation.

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

        # Find the index of the last real (non-padding) token per sample.
        attention_mask = batch["attention_mask"]  # (B, T)
        seq_lengths = attention_mask.sum(dim=1) - 1  # last real token idx

        pooled = hidden_states[
            torch.arange(hidden_states.size(0), device=device),
            seq_lengths,
        ]

        return ModalTensor(data=pooled, modality=Modality.LANGUAGE)

    def get_output_dim(self) -> int:
        """Return the dimensionality of the language embedding."""
        return self._output_dim
