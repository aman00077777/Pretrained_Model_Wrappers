"""
tests/language/test_gpt2.py

Named to match the roadmap's test-file table (test_gpt2.py). The module
under test is fusion/encoders/language/gpt.py (GPTEncoder) — see that
file's real name vs. the checklist's "gpt2.py" for why.

Includes test_pooling_strategy_correct_for_decoder_only_models (gpt2,
per the roadmap spec).
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from fusion.encoders.language.gpt import GPTEncoder
from fusion.encoders.language.pooling import last_token_pooling
from fusion.encoders.modal_tensor import ModalTensor, Modality
from fusion.encoders.registry import ENCODER_REGISTRY

HIDDEN_SIZE = 32


class TestGPTEncoder:
    def _load(self, fake_tokenizer, make_fake_model) -> GPTEncoder:
        with patch(
            "fusion.encoders.language.gpt.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ), patch(
            "fusion.encoders.language.gpt.AutoModel.from_pretrained",
            return_value=make_fake_model(hidden_size=HIDDEN_SIZE),
        ):
            return GPTEncoder(config={"model_name": "gpt2"})

    def test_registered_in_encoder_registry(self) -> None:
        assert "gpt" in ENCODER_REGISTRY
        assert ENCODER_REGISTRY.get("gpt") is GPTEncoder

    def test_wrapper_loads_tokenizer_and_model(self, fake_tokenizer, make_fake_model) -> None:
        """__init__ must load both, and set a pad token since GPT-2's
        tokenizer starts without one."""
        fake_tokenizer.pad_token = None
        with patch(
            "fusion.encoders.language.gpt.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ) as mock_tok, patch(
            "fusion.encoders.language.gpt.AutoModel.from_pretrained",
            return_value=make_fake_model(hidden_size=HIDDEN_SIZE),
        ) as mock_model:
            encoder = GPTEncoder(config={"model_name": "gpt2"})

        mock_tok.assert_called_once()
        mock_model.assert_called_once()
        assert isinstance(encoder, GPTEncoder)
        assert encoder.get_output_dim() == HIDDEN_SIZE
        assert fake_tokenizer.pad_token == fake_tokenizer.eos_token

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
        """GPT-2 is causal — must pool each sequence's LAST non-padded token,
        not the first (there's no CLS token)."""
        torch.manual_seed(0)
        B, T, D = 2, 5, HIDDEN_SIZE
        last_hidden_state = torch.randn(B, T, D)
        attention_mask = torch.ones(B, T, dtype=torch.long)
        attention_mask[0, 3:] = 0  # sequence 0: 3 real tokens, 2 padding

        result = last_token_pooling(last_hidden_state, attention_mask)

        assert torch.allclose(result[0], last_hidden_state[0, 2])
        assert torch.allclose(result[1], last_hidden_state[1, T - 1])
        assert not torch.allclose(result[0], last_hidden_state[0, 0])
