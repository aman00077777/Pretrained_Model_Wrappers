"""
tests/language/test_t5.py

Includes test_pooling_strategy_correct_for_decoder_only_models (t5, per
the roadmap spec) — real t5.py uses masked_mean_pooling since T5 has no
CLS/pooler token; this proves it's actually wired in, not just imported.
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from fusion.encoders.language.pooling import masked_mean_pooling
from fusion.encoders.language.t5 import T5Encoder
from fusion.encoders.modal_tensor import ModalTensor, Modality
from fusion.encoders.registry import ENCODER_REGISTRY

HIDDEN_SIZE = 32


class TestT5Encoder:
    def _load(self, fake_tokenizer, make_fake_model) -> T5Encoder:
        with patch(
            "fusion.encoders.language.t5.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ), patch(
            "fusion.encoders.language.t5.T5EncoderModel.from_pretrained",
            return_value=make_fake_model(hidden_size=HIDDEN_SIZE),
        ):
            return T5Encoder(config={"model_name": "t5-small"})

    def test_registered_in_encoder_registry(self) -> None:
        assert "t5" in ENCODER_REGISTRY
        assert ENCODER_REGISTRY.get("t5") is T5Encoder

    def test_wrapper_loads_tokenizer_and_model(self, fake_tokenizer, make_fake_model) -> None:
        """__init__ must load via T5EncoderModel specifically (encoder-only,
        no decoder) — not the full T5ForConditionalGeneration."""
        with patch(
            "fusion.encoders.language.t5.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ) as mock_tok, patch(
            "fusion.encoders.language.t5.T5EncoderModel.from_pretrained",
            return_value=make_fake_model(hidden_size=HIDDEN_SIZE),
        ) as mock_model:
            encoder = T5Encoder(config={"model_name": "t5-small"})

        mock_tok.assert_called_once()
        mock_model.assert_called_once()
        assert isinstance(encoder, T5Encoder)
        assert encoder.get_output_dim() == HIDDEN_SIZE

    def test_encode_returns_modal_tensor_type(self, fake_tokenizer, make_fake_model) -> None:
        encoder = self._load(fake_tokenizer, make_fake_model)
        result = encoder.encode(["translate English to German: hello"])
        assert isinstance(result, ModalTensor)
        assert result.modality == Modality.LANGUAGE

    def test_encode_output_shape_matches_expected_dim(self, fake_tokenizer, make_fake_model) -> None:
        encoder = self._load(fake_tokenizer, make_fake_model)
        result = encoder.encode(["a single sentence"])
        assert result.data.shape[-1] == encoder.get_output_dim()

    def test_encode_list_of_texts_returns_batch_dim(self, fake_tokenizer, make_fake_model) -> None:
        encoder = self._load(fake_tokenizer, make_fake_model)
        texts = ["first sentence", "a second, longer sentence here", "third"]
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

    def test_pooling_strategy_correct_for_decoder_only_models(self) -> None:
        """T5 has no CLS/pooler token — must use masked mean-pooling over the
        encoder's hidden states, not first-token pooling."""
        torch.manual_seed(0)
        B, T, D = 2, 5, HIDDEN_SIZE
        last_hidden_state = torch.randn(B, T, D)
        attention_mask = torch.ones(B, T, dtype=torch.long)
        attention_mask[0, 3:] = 0

        expected = masked_mean_pooling(last_hidden_state, attention_mask)
        wrong = last_hidden_state[:, 0]
        assert not torch.allclose(expected, wrong)
