"""
Unit tests for the AST audio encoder wrapper.

The pretrained AST model and feature extractor are mocked so that tests:
- never download a pretrained model;
- do not access the Hugging Face Hub;
- exercise the actual encoding and pooling logic;
- verify output shapes and input validation.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.constants import Modality
from fusion.encoders.modal_tensor import ModalTensor
from fusion.models.pretrained.audio.ast import ASTEncoder


HIDDEN_SIZE = 32


class FakeASTModel(nn.Module):
    """Small real model that mimics ASTModel."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()

        self.config = SimpleNamespace(
            hidden_size=hidden_size
        )

        self.proj = nn.Linear(
            64,
            hidden_size,
        )

    def forward(self, input_values, **kwargs):
        batch_size = input_values.shape[0]

        flat = input_values.reshape(
            batch_size,
            -1,
        )

        if flat.shape[1] < 64:
            flat = F.pad(
                flat,
                (0, 64 - flat.shape[1]),
            )
        else:
            flat = flat[:, :64]

        pooled = self.proj(flat)

        # Produce a short sequence so the wrapper's
        # mean-pooling logic is exercised.
        hidden_states = pooled.unsqueeze(1).expand(
            batch_size,
            5,
            HIDDEN_SIZE,
        )

        return SimpleNamespace(
            last_hidden_state=hidden_states
        )


class FakeASTFeatureExtractor:
    """Small processor replacement for ASTFeatureExtractor."""

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

        # AST normally receives log-Mel spectrogram features.
        input_values = torch.randn(
            batch_size,
            1024,
            128,
        )

        return {
            "input_values": input_values
        }


@pytest.fixture
def fake_model():
    return FakeASTModel(
        hidden_size=HIDDEN_SIZE
    )


@pytest.fixture
def fake_processor():
    return FakeASTFeatureExtractor()


@pytest.fixture
def encoder(fake_model, fake_processor):
    return ASTEncoder(
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
        "fusion.models.pretrained.audio.ast"
        ".ASTFeatureExtractor.from_pretrained",
        return_value=fake_processor,
    ) as mock_processor, patch(
        "fusion.models.pretrained.audio.ast"
        ".ASTModel.from_pretrained",
        return_value=fake_model,
    ) as mock_model:

        encoder = ASTEncoder.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

    mock_processor.assert_called_once()
    mock_model.assert_called_once()

    assert isinstance(
        encoder,
        ASTEncoder,
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
    """The pooled embedding dimension must match get_output_dim()."""

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


def test_encode_non_tensor_input_raises_error(
    encoder,
):
    """Non-tensor input must raise TypeError."""

    with pytest.raises(TypeError):
        encoder.encode(
            ["audio", "audio"]
        )


def test_encode_invalid_dimensions_raise_error(
    encoder,
):
    """Only (B,T) and (B,C,T) inputs are valid."""

    audio = torch.randn(
        16000,
    )

    with pytest.raises(ValueError):
        encoder.encode(audio)


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


def test_get_output_dim(
    fake_model,
    fake_processor,
):
    """get_output_dim() must return the configured dimension."""

    encoder = ASTEncoder(
        model=fake_model,
        processor=fake_processor,
        output_dim=HIDDEN_SIZE,
    )

    assert (
        encoder.get_output_dim()
        == HIDDEN_SIZE
    )


def test_encode_produces_finite_embeddings(
    encoder,
):
    """The pooled AST embeddings must contain finite values."""

    audio = torch.randn(
        2,
        16000,
    )

    result = encoder.encode(audio)

    assert torch.isfinite(
        result.data
    ).all()