"""
Wav2Vec2 audio encoder wrapper.

Usage:
from fusion.models.pretrained.audio.wav2vec2 import Wav2Vec2Encoder
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor


from fusion.constants import Modality, DEFAULT_AUDIO_SAMPLE_RATE
from fusion.encoders.base import BaseEncoder
from fusion.encoders.modal_tensor import ModalTensor
from fusion.encoders.registry import register_encoder


@register_encoder("wav2vec2")
class Wav2Vec2Encoder(BaseEncoder):
    """Wraps a Hugging Face Wav2Vec2 model as a FUSION audio encoder."""

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
        model_name_or_path: str = "facebook/wav2vec2-base",
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> "Wav2Vec2Encoder":
        """Load a pretrained Wav2Vec2 model and processor.

        Args:
            model_name_or_path: Hugging Face Hub ID or local model path.
            cache_dir: Optional local cache directory.
            **kwargs: Forwarded to AutoModel.from_pretrained.

        Returns:
            An initialized Wav2Vec2Encoder.
        """
        processor = AutoProcessor.from_pretrained(
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
            processor=processor,
            output_dim=output_dim,
        )

    def encode(
        self,
        inputs: torch.Tensor,
        sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
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
                f"Wav2Vec2Encoder expects a torch.Tensor, got {type(inputs)}"
            )

        if inputs.dim() not in (2, 3):
            raise ValueError(
                "Audio input must have shape (B, T) or (B, C, T), "
                f"got shape {tuple(inputs.shape)}"
            )

        if inputs.shape[0] == 0:
            raise ValueError("Received an empty batch of audio inputs.")

        if sample_rate <= 0:
            raise ValueError(
                f"`sample_rate` must be positive, got {sample_rate}"
            )

        # Convert multi-channel audio to mono.
        if inputs.dim() == 3:
            inputs = inputs.mean(dim=1)

        # Wav2Vec2 expects floating-point waveform values.
        inputs = inputs.float()

        # Resample to the model's expected sampling rate when necessary.
        target_sample_rate = DEFAULT_AUDIO_SAMPLE_RATE

        if sample_rate != target_sample_rate:
            new_length = int(
                inputs.shape[-1] * target_sample_rate / sample_rate
            )

            inputs = F.interpolate(
                inputs.unsqueeze(1),
                size=new_length,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

            sample_rate = target_sample_rate

        # Move audio to the same device as the model.
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        # Convert raw waveforms into model inputs.
        batch = self.processor(
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

        hidden_states = outputs.last_hidden_state

        # Masked mean pooling when an attention mask is available.
        attention_mask = batch.get("attention_mask")

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)

            pooled = (
                (hidden_states * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1e-9)
            )
        else:
            pooled = hidden_states.mean(dim=1)

        return ModalTensor(
            data=pooled,
            modality=Modality.AUDIO,
        )

    def get_output_dim(self) -> int:
        """Return the dimensionality of the audio embedding."""
        return self._output_dim