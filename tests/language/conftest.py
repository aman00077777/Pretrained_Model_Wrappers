"""
Shared test fixtures for the language/ wrapper test suite.

All 8 encoders reach the network via Hugging Face's ``from_pretrained``
calls, made directly inside each encoder's ``__init__`` (the real Phase 2
pattern — there is no separate ``from_pretrained`` classmethod, only
``BaseEncoder.from_config()``, which just does ``cls(config)``). This
sandbox has no access to huggingface.co, and even with access,
downloading real weights per test run is slow and non-deterministic for
CI — so every test here mocks the underlying *transformers* library
calls instead, one layer deeper than ``tests/test_loader.py``'s existing
mocking (which mocks the wrapper's own loading entry point).

Fakes below are real objects (plain classes / real ``nn.Module``s, not
``MagicMock``s) so each encoder's actual forward-pass and pooling code
executes for real against small, fast, controlled tensors. These tests
check real tensor math, not just "was this mock called."
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch
import torch.nn as nn
from transformers import BatchEncoding


class FakeConfig:
    """Duck-typed stand-in for a Hugging Face model config object."""

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeTokenizer:
    """Minimal fake tokenizer/processor — real small tensors, zero network
    access. Accepts text positionally (``AutoTokenizer``-style) or via a
    ``text=`` keyword (``CLIPProcessor``-style) so the same fake covers both
    call conventions used across the real encoder files."""

    def __init__(self) -> None:
        self.pad_token: Optional[str] = "<pad>"
        self.eos_token: str = "<eos>"

    def __call__(
        self,
        texts: Optional[List[str]] = None,
        text: Optional[List[str]] = None,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str = "pt",
    ) -> BatchEncoding:
        actual_texts = texts if texts is not None else text
        batch_size = len(actual_texts)
        # Empty strings still produce a length-1 sequence (a single
        # pad/special-token slot) so batches never collapse to 0 width.
        lengths = [min(max(len(t.split()), 1) + 2, max_length) for t in actual_texts]
        seq_len = max(lengths) if lengths else 1
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
        token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        for i, length in enumerate(lengths):
            input_ids[i, :length] = torch.randint(1, 1000, (length,))
            attention_mask[i, :length] = 1
        return BatchEncoding(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )


class FakeEncoderModel(nn.Module):
    """Fake HF encoder (BERT/RoBERTa/XLM-R/T5/GPT-2 shaped) — a real
    ``nn.Module`` producing correctly-shaped, randomly-initialised outputs
    instead of running real pretrained weights.

    Args:
        hidden_size: Fake hidden dimensionality.
    """

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # A real trainable parameter, so .parameters() / freeze() have
        # something real to iterate over.
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.config = FakeConfig(hidden_size=hidden_size, d_model=hidden_size, n_embd=hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **_kwargs):
        B, T = input_ids.shape
        last_hidden_state = self.proj(torch.randn(B, T, self.hidden_size))
        return SimpleNamespace(last_hidden_state=last_hidden_state)


class FakeCLIPTextTower(nn.Module):
    """Fake stand-in for ``CLIPModel.text_model`` — produces a pooler_output."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **_kwargs):
        B = input_ids.shape[0]
        return SimpleNamespace(pooler_output=self.proj(torch.randn(B, self.hidden_size)))


class FakeCLIPFullModel(nn.Module):
    """Fake stand-in for the full ``CLIPModel`` — clip_text.py only uses
    ``.text_model``, ``.text_projection``, and ``.config.projection_dim``
    off of it, so that's all this provides."""

    def __init__(self, text_hidden_size: int = 32, projection_dim: int = 32) -> None:
        super().__init__()
        self.text_model = FakeCLIPTextTower(text_hidden_size)
        self.text_projection = nn.Linear(text_hidden_size, projection_dim, bias=False)
        self.config = FakeConfig(projection_dim=projection_dim)


# ---------------------------------------------------------------------------
# Pytest fixtures — the actual thing test files use (fixture injection works
# regardless of --import-mode, unlike `from conftest import ...`).
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_tokenizer():
    """A fresh FakeTokenizer instance for one test."""
    return FakeTokenizer()


@pytest.fixture
def make_fake_model():
    """Factory fixture: call with kwargs to build a FakeEncoderModel."""

    def _make(hidden_size: int = 32) -> FakeEncoderModel:
        return FakeEncoderModel(hidden_size=hidden_size)

    return _make


@pytest.fixture
def make_fake_clip_model():
    """Factory fixture: call with kwargs to build a FakeCLIPFullModel."""

    def _make(text_hidden_size: int = 32, projection_dim: int = 32) -> FakeCLIPFullModel:
        return FakeCLIPFullModel(text_hidden_size=text_hidden_size, projection_dim=projection_dim)

    return _make
