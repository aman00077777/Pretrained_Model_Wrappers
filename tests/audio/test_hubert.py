"""
Unit tests for the HuBERT audio encoder wrapper.

The tests use mocked Hugging Face model loading so that:
- no pretrained model is downloaded;
- no Hugging Face Hub access is required;
- the actual encode(), pooling, validation, and output-dimension logic
  are still exercised using a small real PyTorch model.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from fusion.constants import Modality
from fusion.encoders.modal_tensor import ModalTensor
from fusion.models.pretrained.audio.hubert import HuBERTEncoder


HIDDEN_SIZE = 32


class FakeHuBERTModel(nn.Module):
    """Small real model that mimics the HuBERT output structure."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()

        self.config = SimpleNamespace(hidden_size=hidden_size)

        # Real parameter so device handling works.
        self.proj = nn.Linear(1, hidden_size)

    def forward(self, input_values, attention_mask=None, **kwargs):
        batch_size = input_values.shape[0]

        # Simulate HuBERT's sequence output.
        sequence_length = 5

        # Reduce each waveform to one scalar.
        summary = input_values.mean(dim=1, keepdim=True)

        # [B, 1] -> [B, T, 1]
        x = summary.unsqueeze(1).expand(
            batch_size,
            sequence_length,
            1,
        )

        # [B, T, 1] -> [B, T, H]
        hidden = self.proj(x)

        return SimpleNamespace(
            last_hidden_state=hidden
        )


class FakeFeatureExtractor:
    """Small processor replacement for AutoFeatureExtractor."""

    def __call__(
        self,
        waveforms,
        sampling_rate=None,
        return_tensors="pt",
        padding=True,
        **kwargs,
    ):
        if isinstance(waveforms, torch.Tensor):
            batch = waveforms.float()

            if batch.dim() == 1:
                batch = batch.unsqueeze(0)
        else:
            batch = torch.tensor(
                waveforms,
                dtype=torch.float32,
            )

        attention_mask = torch.ones(
            batch.shape,
            dtype=torch.long,
        )

        return {
            "input_values": batch,
            "attention_mask": attention_mask,
        }


@pytest.fixture
def fake_model():
    """Return a small real HuBERT-like model."""
    return FakeHuBERTModel(hidden_size=HIDDEN_SIZE)


@pytest.fixture
def fake_feature_extractor():
    """Return the fake feature extractor."""
    return FakeFeatureExtractor()


@pytest.fixture
def encoder(fake_model, fake_feature_extractor):
    """Construct HuBERTEncoder without downloading anything."""
    return HuBERTEncoder(
        model=fake_model,
        feature_extractor=fake_feature_extractor,
        output_dim=HIDDEN_SIZE,
    )


def test_wrapper_loads_model_and_feature_extractor(
    fake_model,
    fake_feature_extractor,
):
    """from_pretrained() must load both required components."""

    with patch(
        "fusion.models.pretrained.audio.hubert.AutoFeatureExtractor.from_pretrained",
        return_value=fake_feature_extractor,
    ) as mock_extractor, patch(
        "fusion.models.pretrained.audio.hubert.AutoModel.from_pretrained",
        return_value=fake_model,
    ) as mock_model:

        encoder = HuBERTEncoder.from_pretrained(
            "facebook/hubert-base-ls960"
        )

    mock_extractor.assert_called_once()
    mock_model.assert_called_once()

    assert isinstance(encoder, HuBERTEncoder)
    assert encoder.get_output_dim() == HIDDEN_SIZE


def test_encode_returns_modal_tensor(encoder):
    """encode() must return a ModalTensor."""

    audio = torch.randn(2, 16000)

    result = encoder.encode(audio)

    assert isinstance(result, ModalTensor)


def test_encode_returns_audio_modality(encoder):
    """The returned ModalTensor must be tagged as AUDIO."""

    audio = torch.randn(2, 16000)

    result = encoder.encode(audio)

    assert result.modality == Modality.AUDIO


def test_encode_output_shape_matches_expected_dimension(encoder):
    """The final embedding dimension must match get_output_dim()."""

    audio = torch.randn(2, 16000)

    result = encoder.encode(audio)

    assert result.data.shape == (
        2,
        HIDDEN_SIZE,
    )

    assert result.data.shape[-1] == encoder.get_output_dim()


def test_encode_preserves_batch_dimension(encoder):
    """A batch of N audio samples must produce N embeddings."""

    batch_size = 4

    audio = torch.randn(batch_size, 16000)

    result = encoder.encode(audio)

    assert result.data.shape[0] == batch_size


def test_encode_stereo_audio_is_converted_to_mono(encoder):
    """Audio with shape (B, C, T) must be accepted."""

    audio = torch.randn(2, 2, 16000)

    result = encoder.encode(audio)

    assert isinstance(result, ModalTensor)
    assert result.data.shape == (
        2,
        HIDDEN_SIZE,
    )


def test_encode_empty_batch_raises_error(encoder):
    """An empty batch must raise ValueError."""

    audio = torch.empty(0, 16000)

    with pytest.raises(ValueError):
        encoder.encode(audio)


def test_encode_invalid_dimensions_raise_error(encoder):
    """Audio tensors other than (B,T) or (B,C,T) must be rejected."""

    audio = torch.randn(16000)

    with pytest.raises(ValueError):
        encoder.encode(audio)


def test_encode_non_tensor_input_raises_error(encoder):
    """Non-tensor input must raise TypeError."""

    with pytest.raises(TypeError):
        encoder.encode(
            ["not", "a", "tensor"]
        )


def test_get_output_dim():
    """get_output_dim() must return the configured embedding dimension."""

    model = FakeHuBERTModel(hidden_size=HIDDEN_SIZE)
    extractor = FakeFeatureExtractor()

    encoder = HuBERTEncoder(
        model=model,
        feature_extractor=extractor,
        output_dim=HIDDEN_SIZE,
    )

    assert encoder.get_output_dim() == HIDDEN_SIZE