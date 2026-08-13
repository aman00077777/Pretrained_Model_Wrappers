"""
tests/language/test_xlm.py

Note: the real class is named XLMREncoder (not XLMEncoder) — see the
findings summary for why that matters.
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from fusion.encoders.language.xlm import XLMREncoder
from fusion.encoders.modal_tensor import ModalTensor, Modality
from fusion.encoders.registry import ENCODER_REGISTRY

HIDDEN_SIZE = 32


class TestXLMREncoder:
    def _load(self, fake_tokenizer, make_fake_model) -> XLMREncoder:
        with patch(
            "fusion.encoders.language.xlm.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ), patch(
            "fusion.encoders.language.xlm.AutoModel.from_pretrained",
            return_value=make_fake_model(hidden_size=HIDDEN_SIZE),
        ):
            return XLMREncoder(config={"model_name": "xlm-roberta-base"})

    def test_registered_in_encoder_registry(self) -> None:
        assert "xlm" in ENCODER_REGISTRY
        assert ENCODER_REGISTRY.get("xlm") is XLMREncoder

    def test_wrapper_loads_tokenizer_and_model(self, fake_tokenizer, make_fake_model) -> None:
        with patch(
            "fusion.encoders.language.xlm.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ) as mock_tok, patch(
            "fusion.encoders.language.xlm.AutoModel.from_pretrained",
            return_value=make_fake_model(hidden_size=HIDDEN_SIZE),
        ) as mock_model:
            encoder = XLMREncoder(config={"model_name": "xlm-roberta-base"})

        mock_tok.assert_called_once()
        mock_model.assert_called_once()
        assert isinstance(encoder, XLMREncoder)
        assert encoder.get_output_dim() == HIDDEN_SIZE

    def test_encode_returns_modal_tensor_type(self, fake_tokenizer, make_fake_model) -> None:
        encoder = self._load(fake_tokenizer, make_fake_model)
        result = encoder.encode(["bonjour le monde"])
        assert isinstance(result, ModalTensor)
        assert result.modality == Modality.LANGUAGE

    def test_encode_output_shape_matches_expected_dim(self, fake_tokenizer, make_fake_model) -> None:
        encoder = self._load(fake_tokenizer, make_fake_model)
        result = encoder.encode(["une seule phrase"])
        assert result.data.shape[-1] == encoder.get_output_dim()

    def test_encode_list_of_texts_returns_batch_dim(self, fake_tokenizer, make_fake_model) -> None:
        encoder = self._load(fake_tokenizer, make_fake_model)
        texts = ["hello world", "hola mundo", "\u4f60\u597d\u4e16\u754c"]
        result = encoder.encode(texts)
        assert result.data.shape[0] == len(texts)

    def test_encode_empty_string_handled_gracefully(self, fake_tokenizer, make_fake_model) -> None:
        encoder = self._load(fake_tokenizer, make_fake_model)
        result = encoder.encode([""])
        assert result.data.shape[0] == 1
        assert torch.isfinite(result.data).all()

    def test_encode_empty_list_raises_value_error(self, fake_tokenizer, make_fake_model) -> None:
        encoder = self._load(fake_tokenizer, make_fake_model)
        try:
            encoder.encode([])
            assert False, "expected ValueError for an empty batch"
        except ValueError:
            pass

    def test_encode_long_text_truncates_without_error(self, fake_tokenizer, make_fake_model) -> None:
        encoder = self._load(fake_tokenizer, make_fake_model)
        long_text = "word " * 2000
        result = encoder.encode([long_text])
        assert result.data.shape[0] == 1
        assert torch.isfinite(result.data).all()
