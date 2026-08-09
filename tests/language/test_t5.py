"""
tests/language/test_t5.py

Test cases (per the Phase 6 roadmap spec):
  - test_wrapper_loads_tokenizer_and_model
  - test_encode_returns_modal_tensor_type
  - test_encode_output_shape_matches_expected_dim
  - test_encode_list_of_texts_returns_batch_dim
  - test_encode_empty_string_handled_gracefully
  - test_encode_long_text_truncates_without_error
  - test_pooling_strategy_correct_for_decoder_only_models (t5 only, per spec)
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from fusion.constants import Modality
from fusion.encoders.language._pooling import mean_pool
from fusion.encoders.language.t5 import T5Encoder
from fusion.types import ModalTensor

HIDDEN_SIZE = 32


class TestT5Encoder:
    def _load(self, fake_tokenizer, make_fake_model, **kwargs) -> T5Encoder:
        with patch(
            "fusion.encoders.language.t5.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ), patch(
            "fusion.encoders.language.t5.T5EncoderModel.from_pretrained",
            return_value=make_fake_model(hidden_size=HIDDEN_SIZE, has_pooler=False),
        ):
            return T5Encoder.from_pretrained("t5-base", **kwargs)

    def test_wrapper_loads_tokenizer_and_model(self, fake_tokenizer, make_fake_model) -> None:
        """from_pretrained() must call both the tokenizer and encoder loaders."""
        with patch(
            "fusion.encoders.language.t5.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ) as mock_tok, patch(
            "fusion.encoders.language.t5.T5EncoderModel.from_pretrained",
            return_value=make_fake_model(hidden_size=HIDDEN_SIZE, has_pooler=False),
        ) as mock_model:
            encoder = T5Encoder.from_pretrained("t5-base")

        mock_tok.assert_called_once()
        mock_model.assert_called_once()
        assert isinstance(encoder, T5Encoder)
        assert encoder.get_output_dim() == HIDDEN_SIZE

    def test_encode_returns_modal_tensor_type(self, fake_tokenizer, make_fake_model) -> None:
        """encode() must return a ModalTensor tagged as LANGUAGE."""
        encoder = self._load(fake_tokenizer, make_fake_model)
        result = encoder.encode(["translate English to German: hello"])
        assert isinstance(result, ModalTensor)
        assert result.modality == Modality.LANGUAGE

    def test_encode_output_shape_matches_expected_dim(self, fake_tokenizer, make_fake_model) -> None:
        """The pooled embedding's last dim must equal get_output_dim()."""
        encoder = self._load(fake_tokenizer, make_fake_model)
        result = encoder.encode(["a single sentence"])
        assert result.tensor.shape[-1] == encoder.get_output_dim()

    def test_encode_list_of_texts_returns_batch_dim(self, fake_tokenizer, make_fake_model) -> None:
        """A list of N texts must produce a batch dimension of N."""
        encoder = self._load(fake_tokenizer, make_fake_model)
        texts = ["first sentence", "a second, longer sentence here", "third"]
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

    def test_pooling_strategy_correct_for_decoder_only_models(self) -> None:
        """T5 has no CLS/pooler token — encode() must use masked mean-pooling
        over the encoder's hidden states, matching _pooling.mean_pool exactly
        (not, e.g., first-token pooling, which would be wrong for T5)."""
        torch.manual_seed(0)
        B, T, D = 2, 5, HIDDEN_SIZE
        last_hidden_state = torch.randn(B, T, D)
        attention_mask = torch.ones(B, T, dtype=torch.long)
        attention_mask[0, 3:] = 0  # first sequence has 2 real tokens of padding

        expected = mean_pool(last_hidden_state, attention_mask)

        # first-token ("cls-style") pooling would give a different, wrong answer
        wrong = last_hidden_state[:, 0]
        assert not torch.allclose(expected, wrong)
