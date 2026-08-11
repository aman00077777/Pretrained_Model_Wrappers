"""
Unit tests for the Whisper audio encoder wrapper.

The pretrained Whisper model and processor are mocked so that tests:
- never download a pretrained model;
- do not access the Hugging Face Hub;
- exercise the actual encoding and pooling logic;
- verify that only the Whisper encoder is used.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from fusion.constants import Modality
from fusion.encoders.modal_tensor import ModalTensor
from fusion.models.pretrained.audio.whisper import WhisperEncoder


HIDDEN_SIZE = 32
N_MELS = 80


class FakeWhisperEncoder(nn.Module):
    """Small real encoder that mimics Whisper encoder output."""

    def __init__(self, d_model: int = HIDDEN_SIZE):
        super().__init__()

        self.proj = nn.Linear(N_MELS, d_model)

    def forward(self, input_features, **kwargs):
        # Whisper input:
        # [B, n_mels, T]
        #
        # Convert to:
        # [B, T, n_mels]
        x = input_features.transpose(1, 2)

        hidden = self.proj(x)

        return SimpleNamespace(
            last_hidden_state=hidden
        )


class DecoderTripwire:
    """
    Ensures that the Whisper decoder is never called.

    The audio wrapper should use model.encoder() directly.
    """

    def __call__(self, *args, **kwargs):
        raise AssertionError(
            "Whisper decoder must not be used during encoding"
        )


class FakeWhisperModel(nn.Module):
    """Small model mimicking WhisperModel."""

    def __init__(self, d_model: int = HIDDEN_SIZE):
        super().__init__()

        self.config = SimpleNamespace(
            d_model=d_model
        )

        self.encoder = FakeWhisperEncoder(
            d_model=d_model
        )

        self.decoder = DecoderTripwire()


class FakeWhisperProcessor:
    """Small processor replacement for AutoProcessor."""

    def __call__(
        self,
        waveforms,
        sampling_rate=None,
        return_tensors="pt",
        **kwargs,
    ):
        if isinstance(waveforms, torch.Tensor):
            batch_size = waveforms.shape[0]
        else:
            batch_size = len(waveforms)

        sequence_length = 20

        input_features = torch.randn(
            batch_size,
            N_MELS,
            sequence_length,
        )

        return {
            "input_features": input_features
        }


@pytest.fixture
def fake_model():
    return FakeWhisperModel(
        d_model=HIDDEN_SIZE
    )


@pytest.fixture
def fake_processor():
    return FakeWhisperProcessor()


@pytest.fixture
def encoder(fake_model, fake_processor):
    return WhisperEncoder(
        model=fake_model,
        processor=fake_processor,
        output_dim=HIDDEN_SIZE,
    )


def test_wrapper_loads_processor_and_model(
    fake_model,
    fake_processor,
):
    """from_pretrained() must load both processor and model."""

    with patch(
        "fusion.models.pretrained.audio.whisper"
        ".AutoProcessor.from_pretrained",
        return_value=fake_processor,
    ) as mock_processor, patch(
        "fusion.models.pretrained.audio.whisper"
        ".WhisperModel.from_pretrained",
        return_value=fake_model,
    ) as mock_model:

        encoder = WhisperEncoder.from_pretrained(
            "openai/whisper-base"
        )

    mock_processor.assert_called_once()
    mock_model.assert_called_once()

    assert isinstance(
        encoder,
        WhisperEncoder,
    )

    assert (
        encoder.get_output_dim()
        == HIDDEN_SIZE
    )


def test_encode_returns_modal_tensor(
    encoder,
):
    """encode() must return a ModalTensor."""

    audio = torch.randn(
        2,
        16000,
    )

    result = encoder.encode(audio)

    assert isinstance(
        result,
        ModalTensor,
    )


def test_encode_returns_audio_modality(
    encoder,
):
    """The output ModalTensor must be tagged as AUDIO."""

    audio = torch.randn(
        2,
        16000,
    )

    result = encoder.encode(audio)

    assert result.modality == Modality.AUDIO


def test_encode_output_shape_matches_expected_dimension(
    encoder,
):
    """The final embedding dimension must match get_output_dim()."""

    audio = torch.randn(
        2,
        16000,
    )

    result = encoder.encode(audio)

    assert result.data.shape == (
        2,
        HIDDEN_SIZE,
    )

    assert (
        result.data.shape[-1]
        == encoder.get_output_dim()
    )


def test_encode_preserves_batch_dimension(
    encoder,
):
    """N audio samples must produce N embeddings."""

    batch_size = 4

    audio = torch.randn(
        batch_size,
        16000,
    )

    result = encoder.encode(audio)

    assert result.data.shape[0] == batch_size


def test_encode_stereo_audio(
    encoder,
):
    """Audio with shape (B, C, T) must be accepted."""

    audio = torch.randn(
        2,
        2,
        16000,
    )

    result = encoder.encode(audio)

    assert isinstance(
        result,
        ModalTensor,
    )

    assert result.data.shape == (
        2,
        HIDDEN_SIZE,
    )


def test_encode_uses_encoder_only(
    encoder,
):
    """
    Encoding must use the Whisper encoder and never the decoder.
    """

    audio = torch.randn(
        2,
        16000,
    )

    result = encoder.encode(audio)

    assert result.data.shape == (
        2,
        HIDDEN_SIZE,
    )


def test_encode_empty_batch_raises_error(
    encoder,
):
    """An empty batch must raise ValueError."""

    audio = torch.empty(
        0,
        16000,
    )

    with pytest.raises(ValueError):
        encoder.encode(audio)


def test_encode_invalid_dimensions_raise_error(
    encoder,
):
    """Only (B,T) and (B,C,T) audio tensors are valid."""

    audio = torch.randn(
        16000,
    )

    with pytest.raises(ValueError):
        encoder.encode(audio)


def test_encode_non_tensor_input_raises_error(
    encoder,
):
    """Non-tensor input must raise TypeError."""

    with pytest.raises(TypeError):
        encoder.encode(
            ["audio", "audio"]
        )


def test_get_output_dim():
    """get_output_dim() must return the configured dimension."""

    model = FakeWhisperModel(
        d_model=HIDDEN_SIZE
    )

    processor = FakeWhisperProcessor()

    encoder = WhisperEncoder(
        model=model,
        processor=processor,
        output_dim=HIDDEN_SIZE,
    )

    assert (
        encoder.get_output_dim()
        == HIDDEN_SIZE
    )