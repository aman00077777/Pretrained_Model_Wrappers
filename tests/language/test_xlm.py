"""
tests/language/test_xlm.py

Test cases (per the Phase 6 roadmap spec):
  - test_wrapper_loads_tokenizer_and_model
  - test_encode_returns_modal_tensor_type
  - test_encode_output_shape_matches_expected_dim
  - test_encode_list_of_texts_returns_batch_dim
  - test_encode_empty_string_handled_gracefully
  - test_encode_long_text_truncates_without_error
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from fusion.constants import Modality
from fusion.encoders.language.xlm import XLMEncoder
from fusion.types import ModalTensor

HIDDEN_SIZE = 32


class TestXLMEncoder:
    def _load(self, fake_tokenizer, make_fake_model, **kwargs) -> XLMEncoder:
        with patch(
            "fusion.encoders.language.xlm.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ), patch(
            "fusion.encoders.language.xlm.AutoModel.from_pretrained",
            return_value=make_fake_model(hidden_size=HIDDEN_SIZE, has_pooler=True),
        ):
            return XLMEncoder.from_pretrained("xlm-roberta-base", **kwargs)

    def test_wrapper_loads_tokenizer_and_model(self, fake_tokenizer, make_fake_model) -> None:
        """from_pretrained() must call both the tokenizer and model loaders."""
        with patch(
            "fusion.encoders.language.xlm.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ) as mock_tok, patch(
            "fusion.encoders.language.xlm.AutoModel.from_pretrained",
            return_value=make_fake_model(hidden_size=HIDDEN_SIZE),
        ) as mock_model:
            encoder = XLMEncoder.from_pretrained("xlm-roberta-base")

        mock_tok.assert_called_once()
        mock_model.assert_called_once()
        assert isinstance(encoder, XLMEncoder)
        assert encoder.get_output_dim() == HIDDEN_SIZE

    def test_encode_returns_modal_tensor_type(self, fake_tokenizer, make_fake_model) -> None:
        """encode() must return a ModalTensor tagged as LANGUAGE."""
        encoder = self._load(fake_tokenizer, make_fake_model)
        result = encoder.encode(["bonjour le monde"])
        assert isinstance(result, ModalTensor)
        assert result.modality == Modality.LANGUAGE

    def test_encode_output_shape_matches_expected_dim(self, fake_tokenizer, make_fake_model) -> None:
        """The pooled embedding's last dim must equal get_output_dim()."""
        encoder = self._load(fake_tokenizer, make_fake_model)
        result = encoder.encode(["une seule phrase"])
        assert result.tensor.shape[-1] == encoder.get_output_dim()

    def test_encode_list_of_texts_returns_batch_dim(self, fake_tokenizer, make_fake_model) -> None:
        """A list of N texts (mixed languages) must produce a batch dim of N."""
        encoder = self._load(fake_tokenizer, make_fake_model)
        texts = ["hello world", "hola mundo", "\u4f60\u597d\u4e16\u754c"]
        result = encoder.encode(texts)
        assert result.tensor.shape[0] == len(texts)

    def test_encode_empty_string_handled_gracefully(self, fake_tokenizer, make_fake_model) -> None:
        """An empty string must not raise and must still produce one row."""
        encoder = self._load(fake_tokenizer, make_fake_model)
        result = encoder.encode([""])
        assert result.tensor.shape[0] == 1
        assert torch.isfinite(result.tensor).all()

    def test_encode_long_text_truncates_without_error(self, fake_tokenizer, make_fake_model) -> None:
        """Text far longer than max_length must truncate, not error."""
        encoder = self._load(fake_tokenizer, make_fake_model)
        long_text = "word " * 2000
        result = encoder.encode([long_text], max_length=16)
        assert result.tensor.shape[0] == 1
        assert torch.isfinite(result.tensor).all()
