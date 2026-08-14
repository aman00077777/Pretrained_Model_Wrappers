"""
tests/language/test_bert.py

Test cases (per the Phase 6 roadmap spec):
  - test_wrapper_loads_tokenizer_and_model
  - test_encode_returns_modal_tensor_type
  - test_encode_output_shape_matches_expected_dim
  - test_encode_list_of_texts_returns_batch_dim
  - test_encode_empty_string_handled_gracefully
  - test_encode_long_text_truncates_without_error

Plus two checks the rule update specifically requires:
  - test_registered_in_encoder_registry (rule #1/#4 — decorator + registry)
  - test_encode_empty_list_raises_value_error (real bert.py's own guard)
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from fusion.encoders.language.bert import BERTEncoder
from fusion.encoders.modal_tensor import ModalTensor, Modality
from fusion.encoders.registry import ENCODER_REGISTRY

HIDDEN_SIZE = 32


class TestBERTEncoder:
    def _load(self, fake_tokenizer, make_fake_model, **config_overrides) -> BERTEncoder:
        config = {"model_name": "bert-base-uncased", **config_overrides}
        with patch(
            "fusion.encoders.language.bert.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ), patch(
            "fusion.encoders.language.bert.AutoModel.from_pretrained",
            return_value=make_fake_model(hidden_size=HIDDEN_SIZE),
        ):
            return BERTEncoder(config=config)

    def test_registered_in_encoder_registry(self) -> None:
        """@ENCODER_REGISTRY.register("bert") must make the class discoverable
        by name — this is what the FUSION config/training engine relies on."""
        assert "bert" in ENCODER_REGISTRY
        assert ENCODER_REGISTRY.get("bert") is BERTEncoder

    def test_wrapper_loads_tokenizer_and_model(self, fake_tokenizer, make_fake_model) -> None:
        """__init__(config) must call both the tokenizer and model loaders."""
        with patch(
            "fusion.encoders.language.bert.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ) as mock_tok, patch(
            "fusion.encoders.language.bert.AutoModel.from_pretrained",
            return_value=make_fake_model(hidden_size=HIDDEN_SIZE),
        ) as mock_model:
            encoder = BERTEncoder(config={"model_name": "bert-base-uncased"})

        mock_tok.assert_called_once()
        mock_model.assert_called_once()
        assert isinstance(encoder, BERTEncoder)
        assert encoder.get_output_dim() == HIDDEN_SIZE

    def test_default_model_name_used_when_config_omitted(self, fake_tokenizer, make_fake_model) -> None:
        """No config (or no model_name key) must fall back to DEFAULT_MODEL,
        not raise a KeyError."""
        with patch(
            "fusion.encoders.language.bert.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ), patch(
            "fusion.encoders.language.bert.AutoModel.from_pretrained",
            return_value=make_fake_model(hidden_size=HIDDEN_SIZE),
        ) as mock_model:
            BERTEncoder()

        mock_model.assert_called_once_with("bert-base-uncased")

    def test_encode_returns_modal_tensor_type(self, fake_tokenizer, make_fake_model) -> None:
        """encode() must return a ModalTensor tagged as LANGUAGE, with .data set."""
        encoder = self._load(fake_tokenizer, make_fake_model)
        result = encoder.encode(["hello world"])
        assert isinstance(result, ModalTensor)
        assert result.modality == Modality.LANGUAGE
        assert result.attention_mask is not None

    def test_encode_output_shape_matches_expected_dim(self, fake_tokenizer, make_fake_model) -> None:
        """The pooled embedding's last dim must equal get_output_dim()."""
        encoder = self._load(fake_tokenizer, make_fake_model)
        result = encoder.encode(["a single sentence"])
        assert result.data.shape[-1] == encoder.get_output_dim()
        assert result.embedding.shape[-1] == encoder.get_output_dim()

    def test_encode_list_of_texts_returns_batch_dim(self, fake_tokenizer, make_fake_model) -> None:
        """A list of N texts must produce a batch dimension of N."""
        encoder = self._load(fake_tokenizer, make_fake_model)
        texts = ["first sentence", "a second, longer sentence here", "third"]
        result = encoder.encode(texts)
        assert result.data.shape[0] == len(texts)

    def test_encode_single_string_auto_wraps(self, fake_tokenizer, make_fake_model) -> None:
        """A bare string (not a list) must be handled the same as a length-1 list."""
        encoder = self._load(fake_tokenizer, make_fake_model)
        result = encoder.encode("just one string")
        assert result.data.shape[0] == 1

    def test_encode_empty_string_handled_gracefully(self, fake_tokenizer, make_fake_model) -> None:
        """An empty string (one example whose text is "") must not raise."""
        encoder = self._load(fake_tokenizer, make_fake_model)
        result = encoder.encode([""])
        assert result.data.shape[0] == 1
        assert torch.isfinite(result.data).all()

    def test_encode_empty_list_raises_value_error(self, fake_tokenizer, make_fake_model) -> None:
        """An empty BATCH (zero examples) must raise — distinct from [""]."""
        encoder = self._load(fake_tokenizer, make_fake_model)
        try:
            encoder.encode([])
            assert False, "expected ValueError for an empty batch"
        except ValueError:
            pass

    def test_encode_long_text_truncates_without_error(self, fake_tokenizer, make_fake_model) -> None:
        """Text far longer than 512 tokens must truncate, not error."""
        encoder = self._load(fake_tokenizer, make_fake_model)
        long_text = "word " * 2000
        result = encoder.encode([long_text])
        assert result.data.shape[0] == 1
        assert torch.isfinite(result.data).all()
