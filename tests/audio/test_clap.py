"""
Unit tests for the CLAP audio encoder wrapper.

The pretrained CLAP model and feature extractor are mocked so that tests:
- never download a pretrained model;
- do not access the Hugging Face Hub;
- exercise the actual encoding logic;
- verify output shapes and validation behavior.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from fusion.constants import Modality
from fusion.encoders.modal_tensor import ModalTensor
from fusion.models.pretrained.audio.clap import CLAPEncoder


OUTPUT_DIM = 32


class FakeCLAPModel(nn.Module):
    """Small real model that mimics the CLAP audio feature API."""

    def __init__(self, projection_dim: int = OUTPUT_DIM):
        super().__init__()

        self.config = SimpleNamespace(
            projection_dim=projection_dim
        )

        self.proj = nn.Linear(4, projection_dim)

    def get_audio_features(
        self,
        input_features=None,
        is_longer=None,
        **kwargs,
    ):
        batch_size = input_features.shape[0]

        flat = input_features.reshape(
            batch_size,
            -1,
        )

        if flat.shape[1] < 4:
            flat = torch.nn.functional.pad(
                flat,
                (0, 4 - flat.shape[1]),
            )
        else:
            flat = flat[:, :4]

        return self.proj(flat)


class FakeCLAPFeatureExtractor:
    """Small processor replacement for ClapFeatureExtractor."""

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

        input_features = torch.randn(
            batch_size,
            1,
            64,
            64,
        )

        is_longer = torch.zeros(
            batch_size,
            1,
            dtype=torch.bool,
        )

        return {
            "input_features": input_features,
            "is_longer": is_longer,
        }


@pytest.fixture
def fake_model():
    return FakeCLAPModel(
        projection_dim=OUTPUT_DIM
    )


@pytest.fixture
def fake_processor():
    return FakeCLAPFeatureExtractor()


@pytest.fixture
def encoder(fake_model, fake_processor):
    return CLAPEncoder(
        model=fake_model,
        processor=fake_processor,
        output_dim=OUTPUT_DIM,
    )


def test_wrapper_loads_processor_and_model(
    fake_model,
    fake_processor,
):
    """from_pretrained() must load both processor and model."""

    with patch(
        "fusion.models.pretrained.audio.clap"
        ".ClapFeatureExtractor.from_pretrained",
        return_value=fake_processor,
    ) as mock_processor, patch(
        "fusion.models.pretrained.audio.clap"
        ".ClapModel.from_pretrained",
        return_value=fake_model,
    ) as mock_model:

        encoder = CLAPEncoder.from_pretrained(
            "laion/clap-htsat-unfused"
        )

    mock_processor.assert_called_once()
    mock_model.assert_called_once()

    assert isinstance(
        encoder,
        CLAPEncoder,
    )

    assert (
        encoder.get_output_dim()
        == OUTPUT_DIM
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
        OUTPUT_DIM,
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
        OUTPUT_DIM,
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

    encoder = CLAPEncoder(
        model=fake_model,
        processor=fake_processor,
        output_dim=OUTPUT_DIM,
    )

    assert (
        encoder.get_output_dim()
        == OUTPUT_DIM
    )


def test_encode_uses_get_audio_features(
    encoder,
    monkeypatch,
):
    """CLAP encoding must use the model's audio feature method."""

    called = {"value": False}

    original = encoder.model.get_audio_features

    def tracked_get_audio_features(*args, **kwargs):
        called["value"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(
        encoder.model,
        "get_audio_features",
        tracked_get_audio_features,
    )

    audio = torch.randn(
        2,
        16000,
    )

    result = encoder.encode(audio)

    assert called["value"] is True
    assert result.data.shape == (
        2,
        OUTPUT_DIM,
    )