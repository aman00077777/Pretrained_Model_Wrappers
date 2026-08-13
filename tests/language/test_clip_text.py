"""
tests/language/test_clip_text.py

CLIP's real implementation is structurally different from the other 5:
it loads the FULL CLIPModel and pulls out .text_model + .text_projection,
plus a separate CLIPProcessor — not AutoModel/AutoTokenizer. Mocking
reflects that.
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from fusion.encoders.language.clip_text import CLIPTextEncoder
from fusion.encoders.modal_tensor import ModalTensor, Modality
from fusion.encoders.registry import ENCODER_REGISTRY

PROJECTION_DIM = 32


class TestCLIPTextEncoder:
    def _load(self, fake_tokenizer, make_fake_clip_model) -> CLIPTextEncoder:
        with patch(
            "fusion.encoders.language.clip_text.CLIPModel.from_pretrained",
            return_value=make_fake_clip_model(text_hidden_size=PROJECTION_DIM, projection_dim=PROJECTION_DIM),
        ), patch(
            "fusion.encoders.language.clip_text.CLIPProcessor.from_pretrained",
            return_value=fake_tokenizer,
        ):
            return CLIPTextEncoder(config={"model_name": "openai/clip-vit-base-patch32"})

    def test_registered_in_encoder_registry(self) -> None:
        assert "clip_text" in ENCODER_REGISTRY
        assert ENCODER_REGISTRY.get("clip_text") is CLIPTextEncoder

    def test_wrapper_loads_tokenizer_and_model(self, fake_tokenizer, make_fake_clip_model) -> None:
        """__init__ must load the full CLIPModel (for text_model +
        text_projection) and a separate CLIPProcessor."""
        with patch(
            "fusion.encoders.language.clip_text.CLIPModel.from_pretrained",
            return_value=make_fake_clip_model(text_hidden_size=PROJECTION_DIM, projection_dim=PROJECTION_DIM),
        ) as mock_model, patch(
            "fusion.encoders.language.clip_text.CLIPProcessor.from_pretrained",
            return_value=fake_tokenizer,
        ) as mock_proc:
            encoder = CLIPTextEncoder(config={"model_name": "openai/clip-vit-base-patch32"})

        mock_model.assert_called_once()
        mock_proc.assert_called_once()
        assert isinstance(encoder, CLIPTextEncoder)
        assert encoder.get_output_dim() == PROJECTION_DIM

    def test_encode_returns_modal_tensor_type(self, fake_tokenizer, make_fake_clip_model) -> None:
        encoder = self._load(fake_tokenizer, make_fake_clip_model)
        result = encoder.encode(["a photo of a cat"])
        assert isinstance(result, ModalTensor)
        assert result.modality == Modality.LANGUAGE

    def test_encode_output_shape_matches_expected_dim(self, fake_tokenizer, make_fake_clip_model) -> None:
        encoder = self._load(fake_tokenizer, make_fake_clip_model)
        result = encoder.encode(["a photo of a dog"])
        assert result.data.shape[-1] == encoder.get_output_dim()

    def test_encode_list_of_texts_returns_batch_dim(self, fake_tokenizer, make_fake_clip_model) -> None:
        encoder = self._load(fake_tokenizer, make_fake_clip_model)
        texts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
        result = encoder.encode(texts)
        assert result.data.shape[0] == len(texts)

    def test_encode_empty_string_handled_gracefully(self, fake_tokenizer, make_fake_clip_model) -> None:
        encoder = self._load(fake_tokenizer, make_fake_clip_model)
        result = encoder.encode([""])
        assert result.data.shape[0] == 1
        assert torch.isfinite(result.data).all()

    def test_encode_empty_list_raises_value_error(self, fake_tokenizer, make_fake_clip_model) -> None:
        encoder = self._load(fake_tokenizer, make_fake_clip_model)
        try:
            encoder.encode([])
            assert False, "expected ValueError for an empty batch"
        except ValueError:
            pass

    def test_encode_long_text_truncates_without_error(self, fake_tokenizer, make_fake_clip_model) -> None:
        encoder = self._load(fake_tokenizer, make_fake_clip_model)
        long_text = "word " * 500
        result = encoder.encode([long_text])
        assert result.data.shape[0] == 1
        assert torch.isfinite(result.data).all()
