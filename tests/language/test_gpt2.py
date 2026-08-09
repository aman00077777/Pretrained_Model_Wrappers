"""
tests/language/test_gpt2.py

Named to match the roadmap's test-file table (test_gpt2.py). The module
under test is ``fusion/encoders/language/gpt.py`` (GPTEncoder) — see that
file's docstring for why it's gpt.py and not gpt2.py.

Test cases (per the Phase 6 roadmap spec):
  - test_wrapper_loads_tokenizer_and_model
  - test_encode_returns_modal_tensor_type
  - test_encode_output_shape_matches_expected_dim
  - test_encode_list_of_texts_returns_batch_dim
  - test_encode_empty_string_handled_gracefully
  - test_encode_long_text_truncates_without_error
  - test_pooling_strategy_correct_for_decoder_only_models (gpt2 only, per spec)
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from fusion.constants import Modality
from fusion.encoders.language._pooling import last_token_pool
from fusion.encoders.language.gpt import GPTEncoder
from fusion.types import ModalTensor

HIDDEN_SIZE = 32


class TestGPTEncoder:
    def _load(self, fake_tokenizer, make_fake_model, **kwargs) -> GPTEncoder:
        with patch(
            "fusion.encoders.language.gpt.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ), patch(
            "fusion.encoders.language.gpt.AutoModel.from_pretrained",
            return_value=make_fake_model(hidden_size=HIDDEN_SIZE, has_pooler=False),
        ):
            return GPTEncoder.from_pretrained("gpt2", **kwargs)

    def test_wrapper_loads_tokenizer_and_model(self, fake_tokenizer, make_fake_model) -> None:
        """from_pretrained() must call both loaders and set a pad token."""
        fake_tokenizer.pad_token = None  # GPT-2 tokenizers start with none
        with patch(
            "fusion.encoders.language.gpt.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ) as mock_tok, patch(
            "fusion.encoders.language.gpt.AutoModel.from_pretrained",
            return_value=make_fake_model(hidden_size=HIDDEN_SIZE, has_pooler=False),
        ) as mock_model:
            encoder = GPTEncoder.from_pretrained("gpt2")

        mock_tok.assert_called_once()
        mock_model.assert_called_once()
        assert isinstance(encoder, GPTEncoder)
        assert encoder.get_output_dim() == HIDDEN_SIZE
        assert fake_tokenizer.pad_token == fake_tokenizer.eos_token

    def test_encode_returns_modal_tensor_type(self, fake_tokenizer, make_fake_model) -> None:
        """encode() must return a ModalTensor tagged as LANGUAGE."""
        encoder = self._load(fake_tokenizer, make_fake_model)
        result = encoder.encode(["hello world"])
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
        """GPT-2 is causal decoder-only — encode() must pool from each
        sequence's LAST non-padded token, not the first (there's no CLS
        token, and the last position is the only one that has seen the
        whole sequence)."""
        torch.manual_seed(0)
        B, T, D = 2, 5, HIDDEN_SIZE
        last_hidden_state = torch.randn(B, T, D)
        attention_mask = torch.ones(B, T, dtype=torch.long)
        attention_mask[0, 3:] = 0  # first sequence: 3 real tokens, 2 padding

        result = last_token_pool(last_hidden_state, attention_mask)

        # sequence 0's last real token is at index 2 (0-indexed); sequence 1's
        # last real token is at index T-1 (fully unpadded)
        assert torch.allclose(result[0], last_hidden_state[0, 2])
        assert torch.allclose(result[1], last_hidden_state[1, T - 1])
        # and it must differ from naive first-token pooling
        assert not torch.allclose(result[0], last_hidden_state[0, 0])
