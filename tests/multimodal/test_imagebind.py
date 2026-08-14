"""
Unit tests for the ImageBind multimodal encoder wrapper.

The imagebind package and its heavy pretrained model are fully mocked so
that tests:
- never download weights;
- do not require the imagebind package to be installed;
- exercise wrapper routing and output-shape logic;
- verify that all modality embeddings share the same dimension.

Strategy
--------
conftest.py pre-stubs the imagebind.* packages at collection time so that
``multimodal/__init__.py`` can be imported without the real package.  Each
test fixture then *replaces* those stub entries in sys.modules with richer
fakes that return deterministic tensors.  Because ``imagebind.py`` has
already been imported (and ``@register_encoder`` has already run once), we
MUST NOT reload the module — doing so would re-execute the decorator and
raise a duplicate-key error in ENCODER_REGISTRY.  Instead we just
instantiate ImageBindWrapper directly; its ``__init__`` reads
``from imagebind import data`` / ``from imagebind.models import …`` at
runtime, so replacing sys.modules entries is sufficient.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from fusion.constants import Modality
from fusion.encoders.modal_tensor import ModalTensor

# Import the wrapper once at module level (conftest stubs are already active).
from fusion.models.pretrained.multimodal.imagebind import ImageBindWrapper


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIM = 1024   # ImageBind projects every modality to 1024-d.
BATCH_SIZE = 2


# ---------------------------------------------------------------------------
# Rich fake imagebind package (replaces conftest stubs per-test)
# ---------------------------------------------------------------------------

def _install_rich_fakes(monkeypatch):
    """
    Replace the bare conftest stubs with richer fakes that return
    deterministic tensors of the expected shape.
    """
    ModalityType = SimpleNamespace(
        VISION="vision",
        TEXT="text",
        AUDIO="audio",
    )

    # imagebind.models.imagebind_model
    ib_model_mod = types.ModuleType("imagebind.models.imagebind_model")
    ib_model_mod.ModalityType = ModalityType

    # Fake heavy model — __call__ returns per-modality random tensors.
    fake_model = MagicMock()

    def _fake_forward(inputs: dict):
        result = {}
        for key, tensor in inputs.items():
            b = tensor.shape[0] if isinstance(tensor, torch.Tensor) else BATCH_SIZE
            result[key] = torch.randn(b, OUTPUT_DIM)
        return result

    fake_model.side_effect = _fake_forward
    ib_model_mod.imagebind_huge = MagicMock(return_value=fake_model)

    # imagebind.models
    ib_models_mod = types.ModuleType("imagebind.models")
    ib_models_mod.imagebind_model = ib_model_mod

    # imagebind.data
    ib_data_mod = types.ModuleType("imagebind.data")
    ib_data_mod.load_and_transform_vision_data = (
        lambda paths, device: torch.randn(len(paths), 3, 224, 224)
    )
    ib_data_mod.load_and_transform_text = (
        lambda texts, device: torch.randint(0, 100, (len(texts), 77))
    )
    ib_data_mod.load_and_transform_audio_data = (
        lambda paths, device: torch.randn(len(paths), 1, 128, 204)
    )

    # imagebind root
    ib_root = types.ModuleType("imagebind")
    ib_root.data = ib_data_mod
    ib_root.models = ib_models_mod

    for name, mod in [
        ("imagebind", ib_root),
        ("imagebind.data", ib_data_mod),
        ("imagebind.models", ib_models_mod),
        ("imagebind.models.imagebind_model", ib_model_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wrapper(monkeypatch):
    """
    Return an ImageBindWrapper backed by rich fake imagebind modules.

    We install the fakes *before* constructing the wrapper so that the
    ``__init__`` body (which does ``from imagebind import data`` etc.)
    picks up our deterministic tensors.
    """
    _install_rich_fakes(monkeypatch)
    return ImageBindWrapper(
        config={"model_name": "imagebind_huge", "device": "cpu"}
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_imagebind_loads_model_successfully(monkeypatch):
    """ImageBindWrapper.__init__ must complete without errors."""
    _install_rich_fakes(monkeypatch)
    instance = ImageBindWrapper(
        config={"model_name": "imagebind_huge", "device": "cpu"}
    )
    assert instance is not None
    assert instance.get_output_dim() == OUTPUT_DIM


def test_encode_vision_returns_correct_shape(wrapper):
    """encode_vision() must return a (B, 1024) ModalTensor tagged VISION."""
    result = wrapper.encode_vision(["img1.jpg", "img2.jpg"])

    assert isinstance(result, ModalTensor)
    assert result.modality == Modality.VISION
    assert result.data.shape == (BATCH_SIZE, OUTPUT_DIM)


def test_encode_language_returns_correct_shape(wrapper):
    """encode_language() must return a (B, 1024) ModalTensor tagged LANGUAGE."""
    result = wrapper.encode_language(["a dog on a beach", "a cat in a hat"])

    assert isinstance(result, ModalTensor)
    assert result.modality == Modality.LANGUAGE
    assert result.data.shape == (BATCH_SIZE, OUTPUT_DIM)


def test_encode_audio_returns_correct_shape(wrapper):
    """encode_audio() must return a (B, 1024) ModalTensor tagged AUDIO."""
    result = wrapper.encode_audio(["clip1.wav", "clip2.wav"])

    assert isinstance(result, ModalTensor)
    assert result.modality == Modality.AUDIO
    assert result.data.shape == (BATCH_SIZE, OUTPUT_DIM)


def test_all_modality_embeddings_share_same_dim(wrapper):
    """
    Vision, language, and audio embeddings must all share the same last
    dimension — the whole point of ImageBind's joint embedding space.
    """
    vision_result = wrapper.encode_vision(["img1.jpg", "img2.jpg"])
    language_result = wrapper.encode_language(["hello world", "goodbye world"])
    audio_result = wrapper.encode_audio(["a.wav", "b.wav"])

    assert (
        vision_result.data.shape[-1]
        == language_result.data.shape[-1]
        == audio_result.data.shape[-1]
        == OUTPUT_DIM
    )
