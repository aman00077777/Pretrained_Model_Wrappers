"""
tests/test_loader.py

Unit tests for fusion/models/pretrained/loader.py.

Test cases (per Aman's spec):
  - test_load_known_name_returns_correct_wrapper_class
  - test_load_unknown_name_raises_key_error
  - test_load_pretrained_passes_cache_dir_to_underlying_loader
  - test_load_pretrained_passes_kwargs_through
  - test_registry_contains_all_expected_friendly_names
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

# ---------------------------------------------------------------------------
# Local imports (adjust if running outside the installed package)
# ---------------------------------------------------------------------------
from fusion.models.pretrained.loader import (
    PRETRAINED_REGISTRY,
    PretrainedModelError,
    load_pretrained,
)
from fusion.encoders.base import BaseEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_encoder() -> MagicMock:
    """Return a MagicMock that passes isinstance(x, BaseEncoder) checks."""
    mock = MagicMock(spec=BaseEncoder)
    return mock


@contextmanager
def _patch_registry(name: str, hf_id: str, wrapper_cls):
    """Context manager: temporarily inject (hf_id, wrapper_cls) under *name*."""
    original_store = dict(PRETRAINED_REGISTRY._store)

    def factory():
        return (hf_id, wrapper_cls)

    PRETRAINED_REGISTRY._store[name] = factory
    try:
        yield
    finally:
        PRETRAINED_REGISTRY._store.clear()
        PRETRAINED_REGISTRY._store.update(original_store)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadKnownName:
    """load_pretrained() with a registered name calls the correct wrapper."""

    def test_load_known_name_returns_correct_wrapper_class(self):
        """Registry look-up resolves to the expected wrapper and returns it."""
        mock_encoder = _make_mock_encoder()
        MockWrapper = MagicMock()
        MockWrapper.from_pretrained.return_value = mock_encoder
        MockWrapper.__name__ = "MockWrapper"

        with _patch_registry("test-model", "org/test-model", MockWrapper):
            result = load_pretrained("test-model")

        MockWrapper.from_pretrained.assert_called_once_with("org/test-model")
        assert result is mock_encoder

    def test_load_pretrained_passes_cache_dir_to_underlying_loader(self):
        """cache_dir is forwarded to wrapper_cls.from_pretrained."""
        mock_encoder = _make_mock_encoder()
        MockWrapper = MagicMock()
        MockWrapper.from_pretrained.return_value = mock_encoder
        MockWrapper.__name__ = "MockWrapper"

        with _patch_registry("test-model", "org/test-model", MockWrapper):
            load_pretrained("test-model", cache_dir="/tmp/test_cache")

        MockWrapper.from_pretrained.assert_called_once_with(
            "org/test-model", cache_dir="/tmp/test_cache"
        )

    def test_load_pretrained_passes_kwargs_through(self):
        """Extra **kwargs are forwarded verbatim to from_pretrained."""
        import torch

        mock_encoder = _make_mock_encoder()
        MockWrapper = MagicMock()
        MockWrapper.from_pretrained.return_value = mock_encoder
        MockWrapper.__name__ = "MockWrapper"

        with _patch_registry("test-model", "org/test-model", MockWrapper):
            load_pretrained(
                "test-model",
                cache_dir="/tmp/cache",
                torch_dtype=torch.float16,
                device_map="cpu",
            )

        MockWrapper.from_pretrained.assert_called_once_with(
            "org/test-model",
            cache_dir="/tmp/cache",
            torch_dtype=torch.float16,
            device_map="cpu",
        )


class TestLoadUnknownName:
    """load_pretrained() with an unregistered name raises PretrainedModelError."""

    def test_load_unknown_name_raises_key_error(self):
        with pytest.raises(PretrainedModelError) as exc_info:
            load_pretrained("this-name-does-not-exist-xyz")

        err = exc_info.value
        assert "this-name-does-not-exist-xyz" in str(err)
        assert err.details is not None
        assert err.details["name"] == "this-name-does-not-exist-xyz"
        assert isinstance(err.details["available"], list)

    def test_error_message_lists_available_names(self):
        """The raised exception must mention the available friendly names."""
        with pytest.raises(PretrainedModelError) as exc_info:
            load_pretrained("__nonexistent__")

        error_str = str(exc_info.value)
        # At least one known name should appear in the message
        assert any(
            name in error_str
            for name in ("clip", "bert-base", "dinov2-base", "wav2vec2-base")
        )


class TestRegistryContents:
    """PRETRAINED_REGISTRY must contain every expected friendly name."""

    EXPECTED_NAMES = [
        # Vision
        "clip", "clip-large",
        "dinov2-base", "dinov2-large",
        "vit-base",
        "efficientnet-b0",
        "resnet-50",
        "convnext-base",
        "swin-base",
        # Language
        "bert-base", "bert-large",
        "roberta-base",
        "clip-text",
        "t5-base",
        "gpt2",
        "xlm-roberta-base",
        # Audio
        "wav2vec2-base",
        "hubert-base",
        "whisper-base",
        "clap-base",
    ]

    def test_registry_contains_all_expected_friendly_names(self):
        registered = set(PRETRAINED_REGISTRY.list())
        missing = [n for n in self.EXPECTED_NAMES if n not in registered]
        assert missing == [], (
            f"Missing friendly names in PRETRAINED_REGISTRY: {missing}"
        )

    def test_registry_factories_return_hf_id_and_class_tuple(self):
        """Each factory in the registry must return a (str, type) tuple."""
        for name in PRETRAINED_REGISTRY.list():
            factory = PRETRAINED_REGISTRY.get(name)
            result = factory()
            assert isinstance(result, tuple) and len(result) == 2, (
                f"Factory for '{name}' must return a 2-tuple"
            )
            hf_id, cls = result
            assert isinstance(hf_id, str) and hf_id, (
                f"Factory for '{name}' returned a non-string HF id: {hf_id!r}"
            )
            assert isinstance(cls, type), (
                f"Factory for '{name}' returned a non-class: {cls!r}"
            )


class TestLoadPretrainedErrorPropagation:
    """When from_pretrained itself throws, load_pretrained wraps in PretrainedModelError."""

    def test_from_pretrained_exception_is_wrapped(self):
        MockWrapper = MagicMock()
        MockWrapper.from_pretrained.side_effect = RuntimeError("network timeout")
        MockWrapper.__name__ = "MockWrapper"

        with _patch_registry("test-model", "org/test-model", MockWrapper):
            with pytest.raises(PretrainedModelError) as exc_info:
                load_pretrained("test-model")

        assert "network timeout" in str(exc_info.value)
        assert exc_info.value.details["error_type"] == "RuntimeError"
