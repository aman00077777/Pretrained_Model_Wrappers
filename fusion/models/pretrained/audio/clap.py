"""
CLAP audio encoder wrapper.

Usage:
from fusion.models.pretrained.audio.clap import CLAPEncoder
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import ClapFeatureExtractor, ClapModel

from fusion.constants import Modality
from fusion.encoders.base import BaseEncoder
from fusion.encoders.modal_tensor import ModalTensor
from fusion.encoders.registry import register_encoder


@register_encoder("clap")
class CLAPEncoder(BaseEncoder):
    """Wraps a Hugging Face CLAP model as a FUSION audio encoder."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        output_dim: int,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)

        if output_dim <= 0:
            raise ValueError(
                f"`output_dim` must be positive, got {output_dim}"
            )

        self.model = model
        self.processor = processor
        self._output_dim = int(output_dim)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "laion/clap-htsat-unfused",
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> "CLAPEncoder":
        """Load a pretrained CLAP model and feature extractor."""

        processor = ClapFeatureExtractor.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )

        model = ClapModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            **kwargs,
        )

        output_dim = model.config.projection_dim

        return cls(
            model=model,
            processor=processor,
            output_dim=output_dim,
        )

    def encode(
        self,
        inputs: torch.Tensor,
        sample_rate: int = 48000,
    ) -> ModalTensor:
        """Encode a batch of raw audio into CLAP embeddings.

        Args:
            inputs: Audio waveform tensor with shape (B, T), or
                (B, C, T) for mono/stereo audio.
            sample_rate: Sampling rate of the input audio.

        Returns:
            ModalTensor containing a (B, output_dim) embedding tensor.
        """
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(
                f"CLAPEncoder expects a torch.Tensor, got {type(inputs)}"
            )

        if inputs.dim() not in (2, 3):
            raise ValueError(
                "Audio input must have shape (B, T) or (B, C, T), "
                f"got shape {tuple(inputs.shape)}"
            )

        if inputs.shape[0] == 0:
            raise ValueError("Received an empty batch of audio inputs.")

        # Convert multi-channel audio to mono.
        if inputs.dim() == 3:
            inputs = inputs.mean(dim=1)

        inputs = inputs.float()

        # Keep the model and inputs on the same device.
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        # Convert raw waveforms into CLAP model inputs.
        batch = self.processor(
            inputs,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        batch = {
            key: value.to(device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }

        # CLAP provides a dedicated audio embedding method.
        audio_features = self.model.get_audio_features(**batch)

        if audio_features.dim() != 2:
            raise ValueError(
                "CLAP audio features must have shape (B, output_dim), "
                f"got shape {tuple(audio_features.shape)}"
            )

        return ModalTensor(
            data=audio_features,
            modality=Modality.AUDIO,
        )

    def get_output_dim(self) -> int:
        """Return the dimensionality of the audio embedding."""
        return self._output_dim