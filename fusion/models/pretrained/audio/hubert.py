"""
HuBERT audio encoder wrapper.

Usage:
from fusion.models.pretrained.audio.hubert import HuBERTEncoder
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import AutoFeatureExtractor, AutoModel

from fusion.constants import Modality
from fusion.encoders.base import BaseEncoder
from fusion.encoders.modal_tensor import ModalTensor
from fusion.encoders.registry import register_encoder


@register_encoder("hubert")
class HuBERTEncoder(BaseEncoder):
    """Wraps a Hugging Face HuBERT model as a FUSION audio encoder."""

    def __init__(
        self,
        model: Any,
        feature_extractor: Any,
        output_dim: int,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)

        if output_dim <= 0:
            raise ValueError(
                f"`output_dim` must be positive, got {output_dim}"
            )

        self.model = model
        self.feature_extractor = feature_extractor
        self._output_dim = int(output_dim)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "facebook/hubert-base-ls960",
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> "HuBERTEncoder":
        """Load a pretrained HuBERT model and feature extractor.

        Args:
            model_name_or_path: Hugging Face Hub ID or local model path.
            cache_dir: Optional local cache directory.
            **kwargs: Forwarded to AutoModel.from_pretrained.

        Returns:
            An initialized HuBERTEncoder.
        """
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )

        model = AutoModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            **kwargs,
        )

        output_dim = model.config.hidden_size

        return cls(
            model=model,
            feature_extractor=feature_extractor,
            output_dim=output_dim,
        )

    def encode(
        self,
        inputs: torch.Tensor,
        sample_rate: int = 16000,
    ) -> ModalTensor:
        """Encode a batch of raw audio into pooled embeddings.

        Args:
            inputs: Audio waveform tensor with shape (B, T), or
                (B, C, T) for mono/stereo audio.
            sample_rate: Sampling rate of the input audio.

        Returns:
            ModalTensor containing a (B, output_dim) embedding tensor.
        """
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(
                f"HuBERTEncoder expects a torch.Tensor, got {type(inputs)}"
            )

        if inputs.dim() not in (2, 3):
            raise ValueError(
                "Audio input must have shape (B, T) or (B, C, T), "
                f"got shape {tuple(inputs.shape)}"
            )

        if inputs.shape[0] == 0:
            raise ValueError("Received an empty batch of audio inputs.")

        # Convert stereo/multi-channel audio to mono.
        if inputs.dim() == 3:
            inputs = inputs.mean(dim=1)

        # Move audio to the same device as the model.
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        # HuBERT expects floating-point waveform values.
        inputs = inputs.float()

        # Extract the input features required by HuBERT.
        batch = self.feature_extractor(
            inputs,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )

        batch = {
            key: value.to(device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }

        outputs = self.model(**batch)

        # HuBERT produces one hidden vector per audio timestep.
        hidden_states = outputs.last_hidden_state

        # Mean-pool the sequence into one embedding per audio sample.
        pooled = hidden_states.mean(dim=1)

        return ModalTensor(
            data=pooled,
            modality=Modality.AUDIO,
        )

    def get_output_dim(self) -> int:
        """Return the dimensionality of the audio embedding."""
        return self._output_dim