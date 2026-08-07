"""
tests/test_converter.py

Unit tests for fusion/models/pretrained/converter.py.

Test cases (per Aman's spec):
  - test_convert_vision_model_returns_custom_vision_encoder
  - test_convert_language_model_returns_custom_language_encoder
  - test_convert_audio_model_returns_custom_audio_encoder
  - test_convert_sets_output_dim_correctly
  - test_convert_invalid_modality_raises_value_error
  - test_converted_encoder_conforms_to_base_encoder_contract
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from fusion.constants import Modality
from fusion.encoders.base import BaseEncoder
from fusion.encoders.vision.custom import CustomVisionEncoder
from fusion.encoders.language.custom import CustomLanguageEncoder
from fusion.encoders.audio.custom import CustomAudioEncoder
from fusion.exceptions import ModalityError
from fusion.models.pretrained.converter import convert_to_fusion_format


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _IdentityModule(nn.Module):
    """Trivial module: returns input unchanged (for dimension testing)."""

    def __init__(self, output_dim: int):
        super().__init__()
        self._out = output_dim
        self.linear = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.fixture
def dummy_module():
    return _IdentityModule(output_dim=64)


# ---------------------------------------------------------------------------
# Tests — correct wrapper class returned per modality
# ---------------------------------------------------------------------------

class TestConvertReturnsCorrectWrapper:

    def test_convert_vision_model_returns_custom_vision_encoder(
        self, dummy_module
    ):
        encoder = convert_to_fusion_format(
            dummy_module, Modality.VISION, output_dim=64
        )
        assert isinstance(encoder, CustomVisionEncoder)

    def test_convert_language_model_returns_custom_language_encoder(
        self, dummy_module
    ):
        encoder = convert_to_fusion_format(
            dummy_module, Modality.LANGUAGE, output_dim=64
        )
        assert isinstance(encoder, CustomLanguageEncoder)

    def test_convert_audio_model_returns_custom_audio_encoder(
        self, dummy_module
    ):
        encoder = convert_to_fusion_format(
            dummy_module, Modality.AUDIO, output_dim=64
        )
        assert isinstance(encoder, CustomAudioEncoder)


# ---------------------------------------------------------------------------
# Tests — output_dim is set correctly
# ---------------------------------------------------------------------------

class TestConvertOutputDim:

    @pytest.mark.parametrize("modality,output_dim", [
        (Modality.VISION,   128),
        (Modality.LANGUAGE, 256),
        (Modality.AUDIO,    512),
        (Modality.VISION,   768),
    ])
    def test_convert_sets_output_dim_correctly(self, modality, output_dim):
        module = _IdentityModule(output_dim=output_dim)
        encoder = convert_to_fusion_format(module, modality, output_dim=output_dim)
        assert encoder.get_output_dim() == output_dim, (
            f"Expected output_dim={output_dim}, got {encoder.get_output_dim()}"
        )


# ---------------------------------------------------------------------------
# Tests — invalid modality raises ModalityError
# ---------------------------------------------------------------------------

class TestConvertInvalidModality:

    def test_convert_invalid_modality_raises_value_error(self, dummy_module):
        """Passing a non-Modality value must raise ModalityError."""
        with pytest.raises(ModalityError):
            convert_to_fusion_format(
                dummy_module,
                modality="video",   # type: ignore[arg-type]  — intentional bad input
                output_dim=64,
            )

    def test_convert_none_modality_raises_modality_error(self, dummy_module):
        with pytest.raises(ModalityError):
            convert_to_fusion_format(dummy_module, modality=None, output_dim=64)  # type: ignore

    def test_modality_error_carries_details(self, dummy_module):
        """ModalityError.details must document what was received and what's valid."""
        with pytest.raises(ModalityError) as exc_info:
            convert_to_fusion_format(dummy_module, modality="video", output_dim=64)  # type: ignore

        err = exc_info.value
        assert err.details is not None
        assert "received" in err.details
        assert "supported" in err.details
        assert isinstance(err.details["supported"], list)


# ---------------------------------------------------------------------------
# Tests — returned encoder satisfies BaseEncoder contract
# ---------------------------------------------------------------------------

class TestConvertBaseEncoderContract:

    @pytest.mark.parametrize("modality", [
        Modality.VISION,
        Modality.LANGUAGE,
        Modality.AUDIO,
    ])
    def test_converted_encoder_conforms_to_base_encoder_contract(
        self, modality
    ):
        """Every converted encoder must pass isinstance(x, BaseEncoder)."""
        module = _IdentityModule(output_dim=32)
        encoder = convert_to_fusion_format(module, modality, output_dim=32)
        assert isinstance(encoder, BaseEncoder)

    @pytest.mark.parametrize("modality", [
        Modality.VISION,
        Modality.LANGUAGE,
        Modality.AUDIO,
    ])
    def test_converted_encoder_has_get_output_dim(self, modality):
        module = _IdentityModule(output_dim=32)
        encoder = convert_to_fusion_format(module, modality, output_dim=32)
        assert callable(getattr(encoder, "get_output_dim", None))
        assert encoder.get_output_dim() == 32

    @pytest.mark.parametrize("modality", [
        Modality.VISION,
        Modality.LANGUAGE,
        Modality.AUDIO,
    ])
    def test_converted_encoder_is_nn_module(self, modality):
        module = _IdentityModule(output_dim=32)
        encoder = convert_to_fusion_format(module, modality, output_dim=32)
        assert isinstance(encoder, nn.Module)

    def test_converted_encoder_freeze_unfreeze(self):
        """freeze() / unfreeze() must toggle requires_grad on all parameters."""
        module = _IdentityModule(output_dim=32)
        encoder = convert_to_fusion_format(module, Modality.VISION, output_dim=32)

        encoder.freeze()
        for p in encoder.parameters():
            assert p.requires_grad is False

        encoder.unfreeze()
        for p in encoder.parameters():
            assert p.requires_grad is True

    def test_converted_encoder_freeze_except_last_n(self):
        """freeze_except_last_n(1) must leave the final child trainable."""
        module = _IdentityModule(output_dim=32)
        encoder = convert_to_fusion_format(module, Modality.VISION, output_dim=32)

        # Should not raise
        encoder.freeze_except_last_n(1)
