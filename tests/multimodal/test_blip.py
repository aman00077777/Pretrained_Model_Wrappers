"""
Unit tests for the BLIP multimodal encoder wrapper.

BlipForConditionalGeneration and BlipProcessor are mocked so that tests:
- never download pretrained weights;
- do not access the Hugging Face Hub;
- exercise encode_image, encode_text, generate_caption, and error handling;
- verify output shapes against the model's vision hidden size.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from PIL import Image

from fusion.constants import Modality
from fusion.encoders.modal_tensor import ModalTensor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VISION_HIDDEN_SIZE = 32   # small stand-in for 768
TEXT_HIDDEN_SIZE = 32
BATCH_SIZE = 2

def _get_blip():
    """
    Lazily import the blip submodule.

    conftest.py has already stubbed the missing sibling modules
    (clip, align) in sys.modules, so multimodal/__init__.py no longer
    raises ModuleNotFoundError when this runs.
    """
    import fusion.models.pretrained.multimodal.blip as _mod
    return _mod


# ---------------------------------------------------------------------------
# Fake model components
# ---------------------------------------------------------------------------

class FakeVisionModel(nn.Module):
    """Mimics BlipVisionModel — returns pooler_output of shape (B, D)."""

    def __init__(self, hidden_size: int = VISION_HIDDEN_SIZE):
        super().__init__()
        self.proj = nn.Linear(3, hidden_size)

    def forward(self, pixel_values, **kwargs):
        b = pixel_values.shape[0]
        pooler_output = torch.randn(b, VISION_HIDDEN_SIZE)
        return SimpleNamespace(pooler_output=pooler_output)


class FakeBertModel(nn.Module):
    """Mimics the BERT encoder inside BLIP's text decoder."""

    def __init__(self, hidden_size: int = TEXT_HIDDEN_SIZE):
        super().__init__()
        self.proj = nn.Linear(1, hidden_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        b, seq = input_ids.shape
        last_hidden_state = torch.randn(b, seq, TEXT_HIDDEN_SIZE)
        return SimpleNamespace(last_hidden_state=last_hidden_state)


class FakeTextDecoder(nn.Module):
    """Mimics BLIP's text_decoder, exposing .bert."""

    def __init__(self):
        super().__init__()
        self.bert = FakeBertModel()


class FakeBLIPModel(nn.Module):
    """Mimics BlipForConditionalGeneration at the surface used by BLIPWrapper."""

    def __init__(self):
        super().__init__()

        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(hidden_size=VISION_HIDDEN_SIZE)
        )

        self.vision_model = FakeVisionModel(VISION_HIDDEN_SIZE)
        self.text_decoder = FakeTextDecoder()

    def generate(self, pixel_values=None, **kwargs):
        """Return a plausible token-id tensor."""
        return torch.tensor([[101, 1037, 3899, 102, 0]])

    def eval(self):
        return self


class FakeBLIPProcessor:
    """Mimics BlipProcessor — handles both images-only and text calls."""

    def __call__(
        self,
        images=None,
        text=None,
        return_tensors="pt",
        padding=False,
        truncation=False,
        **kwargs,
    ):
        result = {}

        if images is not None:
            img_list = images if isinstance(images, list) else [images]
            result["pixel_values"] = torch.randn(len(img_list), 3, 224, 224)

        if text is not None:
            text_list = text if isinstance(text, list) else [text]
            b = len(text_list)
            seq = 10
            result["input_ids"] = torch.randint(0, 100, (b, seq))
            result["attention_mask"] = torch.ones(b, seq, dtype=torch.long)

        return result

    def decode(self, token_ids, skip_special_tokens=True):
        return "a dog sitting on a beach"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_model():
    return FakeBLIPModel()


@pytest.fixture
def fake_processor():
    return FakeBLIPProcessor()


@pytest.fixture
def wrapper(fake_model, fake_processor):
    """
    Build a BLIPWrapper by patching the HuggingFace constructors directly
    on the already-imported blip submodule so no network call is made.
    """
    _blip_mod = _get_blip()
    with patch.object(
        _blip_mod.BlipForConditionalGeneration,
        "from_pretrained",
        return_value=fake_model,
    ), patch.object(
        _blip_mod.BlipProcessor,
        "from_pretrained",
        return_value=fake_processor,
    ):
        return _blip_mod.BLIPWrapper(
            config={"model_name": "Salesforce/blip-image-captioning-base"}
        )


@pytest.fixture
def dummy_image():
    """A small solid-colour PIL image."""
    return Image.new("RGB", (224, 224), color=(128, 64, 32))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_blip_loads_model_and_processor(fake_model, fake_processor):
    """BLIPWrapper.__init__ must call both from_pretrained constructors."""
    _blip_mod = _get_blip()
    BLIPWrapper = _blip_mod.BLIPWrapper
    with patch.object(
        _blip_mod.BlipForConditionalGeneration,
        "from_pretrained",
        return_value=fake_model,
    ) as mock_model_ctor, patch.object(
        _blip_mod.BlipProcessor,
        "from_pretrained",
        return_value=fake_processor,
    ) as mock_processor_ctor:
        blip = BLIPWrapper(
            config={"model_name": "Salesforce/blip-image-captioning-base"}
        )

    mock_model_ctor.assert_called_once()
    mock_processor_ctor.assert_called_once()
    assert isinstance(blip, BLIPWrapper)


def test_encode_image_returns_correct_shape(wrapper, dummy_image):
    """encode_image() must return a (B, vision_hidden_size) VISION ModalTensor."""
    images = [dummy_image, dummy_image]

    result = wrapper.encode_image(images)

    assert isinstance(result, ModalTensor)
    assert result.modality == Modality.VISION
    assert result.data.shape == (BATCH_SIZE, VISION_HIDDEN_SIZE)
    assert result.data.shape[-1] == wrapper.get_output_dim()


def test_encode_text_returns_correct_shape(wrapper):
    """encode_text() must return a (B, hidden_size) LANGUAGE ModalTensor."""
    texts = ["a dog on a beach", "a cat in a hat"]

    result = wrapper.encode_text(texts)

    assert isinstance(result, ModalTensor)
    assert result.modality == Modality.LANGUAGE
    assert result.data.ndim == 2
    assert result.data.shape[0] == BATCH_SIZE


def test_generate_caption_returns_non_empty_string(wrapper, dummy_image):
    """generate_caption() must return a non-empty string for a valid image."""
    caption = wrapper.generate_caption(dummy_image)

    assert isinstance(caption, str)
    assert len(caption.strip()) > 0


def test_generate_caption_handles_invalid_image_gracefully(wrapper):
    """generate_caption(None) must raise ValueError."""
    with pytest.raises(ValueError):
        wrapper.generate_caption(None)
