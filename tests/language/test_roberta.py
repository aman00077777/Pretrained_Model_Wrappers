"""
tests/language/test_roberta.py

Test cases (per the Phase 6 roadmap spec) plus registry + empty-list checks,
matching test_bert.py's pattern.
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from fusion.encoders.language.roberta import RoBERTaEncoder
from fusion.encoders.modal_tensor import ModalTensor, Modality
from fusion.encoders.registry import ENCODER_REGISTRY

HIDDEN_SIZE = 32


class TestRoBERTaEncoder:
    def _load(self, fake_tokenizer, make_fake_model) -> RoBERTaEncoder:
        with patch(
            "fusion.encoders.language.roberta.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ), patch(
            "fusion.encoders.language.roberta.AutoModel.from_pretrained",
            return_value=make_fake_model(hidden_size=HIDDEN_SIZE),
        ):
            return RoBERTaEncoder(config={"model_name": "roberta-base"})

    def test_registered_in_encoder_registry(self) -> None:
        assert "roberta" in ENCODER_REGISTRY
        assert ENCODER_REGISTRY.get("roberta") is RoBERTaEncoder

    def test_wrapper_loads_tokenizer_and_model(self, fake_tokenizer, make_fake_model) -> None:
        with patch(
            "fusion.encoders.language.roberta.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ) as mock_tok, patch(
            "fusion.encoders.language.roberta.AutoModel.from_pretrained",
            return_value=make_fake_model(hidden_size=HIDDEN_SIZE),
        ) as mock_model:
            encoder = RoBERTaEncoder(config={"model_name": "roberta-base"})

        mock_tok.assert_called_once()
        mock_model.assert_called_once()
        assert isinstance(encoder, RoBERTaEncoder)
        assert encoder.get_output_dim() == HIDDEN_SIZE

    def test_encode_returns_modal_tensor_type(self, fake_tokenizer, make_fake_model) -> None:
        encoder = self._load(fake_tokenizer, make_fake_model)
        result = encoder.encode(["hello world"])
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

    def test_token_type_ids_not_passed_to_model(self, fake_tokenizer, make_fake_model) -> None:
        """RoBERTa doesn't use token_type_ids — the real code explicitly pops
        it before the forward pass; confirm that doesn't crash even though
        the fake tokenizer (like a real one) includes it in its output."""
        encoder = self._load(fake_tokenizer, make_fake_model)
        result = encoder.encode(["some text"])  # would error if popped incorrectly
        assert result.data.shape[0] == 1

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
