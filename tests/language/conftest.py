"""
Shared test fixtures for the language/ wrapper test suite.

All 6 wrappers reach the network via Hugging Face's ``from_pretrained``
classmethods. This sandbox has no access to huggingface.co, and even
with access, downloading real weights per test run is slow and
non-deterministic for CI — so every test here mocks the HF
``from_pretrained`` call points, one layer deeper than
``tests/test_loader.py``'s existing mocking (which mocks the wrapper's
own ``from_pretrained``; these mock the *transformers* library calls
inside each wrapper).

FakeTokenizer / FakeEncoderModel below are real objects (a plain class
and a real ``nn.Module``, not ``MagicMock``s) so each wrapper's actual
forward-pass and pooling code executes for real against small, fast,
controlled tensors. These tests check real tensor math, not just
"was this mock called."
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
    """Minimal fake tokenizer — real small tensors, zero network access."""

    def __init__(self) -> None:
        self.pad_token: Optional[str] = "<pad>"
        self.eos_token: str = "<eos>"

    def __call__(
        self,
        texts: List[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str = "pt",
    ) -> BatchEncoding:
        batch_size = len(texts)
        # Empty strings still produce a length-1 sequence (a single
        # pad/special-token slot) so batches never collapse to 0 width.
        lengths = [min(max(len(t.split()), 1) + 2, max_length) for t in texts]
        seq_len = max(lengths) if lengths else 1
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
        for i, length in enumerate(lengths):
            input_ids[i, :length] = torch.randint(1, 1000, (length,))
            attention_mask[i, :length] = 1
        return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})


class FakeEncoderModel(nn.Module):
    """Fake HF encoder — a real ``nn.Module`` producing correctly-shaped,
    randomly-initialised outputs instead of running real pretrained weights.

    Args:
        hidden_size: Fake hidden dimensionality.
        has_pooler: If True, output includes a ``pooler_output`` (BERT/
            RoBERTa/XLM-style).
        has_text_embeds: If True, output includes ``text_embeds`` instead
            (CLIP-style).
        projection_dim: Only used when ``has_text_embeds`` is True.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        has_pooler: bool = True,
        has_text_embeds: bool = False,
        projection_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.has_pooler = has_pooler
        self.has_text_embeds = has_text_embeds
        proj_dim = projection_dim or hidden_size
        # A real trainable parameter, so .parameters() / freeze() have
        # something real to iterate over and .device resolves correctly.
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.config = FakeConfig(
            hidden_size=hidden_size,
            d_model=hidden_size,
            n_embd=hidden_size,
            projection_dim=proj_dim,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **_kwargs):
        B, T = input_ids.shape
        last_hidden_state = self.proj(torch.randn(B, T, self.hidden_size))
        out = {"last_hidden_state": last_hidden_state}
        if self.has_pooler:
            out["pooler_output"] = torch.randn(B, self.hidden_size)
        if self.has_text_embeds:
            out["text_embeds"] = torch.randn(B, self.config.projection_dim)
        return SimpleNamespace(**out)


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

    def _make(
        hidden_size: int = 32,
        has_pooler: bool = True,
        has_text_embeds: bool = False,
        projection_dim: Optional[int] = None,
    ) -> FakeEncoderModel:
        return FakeEncoderModel(
            hidden_size=hidden_size,
            has_pooler=has_pooler,
            has_text_embeds=has_text_embeds,
            projection_dim=projection_dim,
        )

    return _make
