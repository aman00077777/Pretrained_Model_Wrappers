"""Unit tests for the Swin Vision encoder wrapper."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from PIL import Image

import fusion

# Inject core fusion package paths to resolve imports like fusion.encoders
core_fusion_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Fusion-"))
if os.path.exists(core_fusion_dir):
    fusion_pkg_path = os.path.join(core_fusion_dir, "fusion")
    if fusion_pkg_path not in fusion.__path__:
        fusion.__path__.append(fusion_pkg_path)

from fusion.constants import Modality
from fusion.encoders.modal_tensor import ModalTensor
from fusion.exceptions import DataError, EncoderError
from fusion.models.pretrained.vision.swin import SwinEncoder

HIDDEN_SIZE = 32


class FakeConfig:
    """Fake config for Hugging Face Swin."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE) -> None:
        self.hidden_size = hidden_size


class FakeSwinModel(nn.Module):
    """Fake Swin Model."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE) -> None:
        super().__init__()
        self.config = FakeConfig(hidden_size=hidden_size)
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self, pixel_values: torch.Tensor, **kwargs: Any) -> SimpleNamespace:
        B = pixel_values.shape[0]
        C = self.config.hidden_size
        return SimpleNamespace(pooler_output=torch.randn(B, C))


class FakeSwinProcessor:
    """Fake Swin Processor."""

    def __call__(self, images: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        if isinstance(images, list):
            batch_size = len(images)
        else:
            batch_size = 1
        return {"pixel_values": torch.randn(batch_size, 3, 224, 224)}


@pytest.fixture
def fake_model() -> FakeSwinModel:
    return FakeSwinModel(hidden_size=HIDDEN_SIZE)


@pytest.fixture
def fake_processor() -> FakeSwinProcessor:
    return FakeSwinProcessor()


@pytest.fixture
def encoder(fake_model: FakeSwinModel, fake_processor: FakeSwinProcessor) -> SwinEncoder:
    with patch(
        "fusion.models.pretrained.vision.swin.AutoImageProcessor.from_pretrained",
        return_value=fake_processor,
    ), patch(
        "fusion.models.pretrained.vision.swin.SwinModel.from_pretrained",
        return_value=fake_model,
    ):
        return SwinEncoder.from_pretrained("microsoft/swin-base-patch4-window7-224")


def test_wrapper_loads_model_successfully(fake_model: FakeSwinModel, fake_processor: FakeSwinProcessor) -> None:
    """Verify that the wrapper loads the HuggingFace model and processor successfully."""
    with patch(
        "fusion.models.pretrained.vision.swin.AutoImageProcessor.from_pretrained",
        return_value=fake_processor,
    ) as mock_proc, patch(
        "fusion.models.pretrained.vision.swin.SwinModel.from_pretrained",
        return_value=fake_model,
    ) as mock_model:
        enc = SwinEncoder("microsoft/swin-base-patch4-window7-224")

    mock_proc.assert_called_once()
    mock_model.assert_called_once()
    assert isinstance(enc, SwinEncoder)
    assert enc.get_output_dim() == HIDDEN_SIZE


def test_encode_returns_modal_tensor_type(encoder: SwinEncoder) -> None:
    """Verify that encode() returns a ModalTensor."""
    img = Image.new("RGB", (224, 224))
    res = encoder.encode(img)
    assert isinstance(res, ModalTensor)
    assert res.modality == Modality.VISION


def test_encode_output_shape_matches_expected_dim(encoder: SwinEncoder) -> None:
    """Verify that the output shape matches the expected output dimension."""
    img = Image.new("RGB", (224, 224))
    res = encoder.encode(img)
    assert res.shape == (1, HIDDEN_SIZE)
    assert res.shape[-1] == encoder.get_output_dim()


def test_encode_batch_of_images_returns_batch_dim(encoder: SwinEncoder) -> None:
    """Verify that encoding a batch of images returns the correct batch dimension."""
    imgs = [Image.new("RGB", (224, 224)) for _ in range(3)]
    res = encoder.encode(imgs)
    assert res.shape == (3, HIDDEN_SIZE)
    assert res.batch_size == 3


def test_encode_single_image_handled_correctly(encoder: SwinEncoder) -> None:
    """Verify that a single PIL Image is handled correctly."""
    img = Image.new("RGB", (224, 224))
    res = encoder.encode(img)
    assert res.shape == (1, HIDDEN_SIZE)


def test_invalid_image_input_raises_clear_error(encoder: SwinEncoder) -> None:
    """Verify that corrupt or invalid image inputs raise a clear FusionError subclass (DataError)."""
    with pytest.raises(DataError):
        encoder.encode("not_an_image")  # type: ignore[arg-type]

    with pytest.raises(DataError):
        encoder.encode([])

    with pytest.raises(DataError):
        # Corrupt image of size 0x0
        corrupt_img = Image.new("RGB", (0, 0))
        encoder.encode(corrupt_img)
