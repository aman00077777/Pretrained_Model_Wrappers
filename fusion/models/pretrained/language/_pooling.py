"""
Shared pooling helpers for the language/ sub-module.

Not one of the 6 assigned wrapper files — reusable logic so BERT-style,
T5-style, and GPT-style pooling aren't each hand-rolled per wrapper.

Note for Arya (audio/clap.py): CLAP's text tower is BERT/RoBERTa-shaped,
so ``mean_pool`` here should be directly reusable for CLAP's
``encode_text`` rather than reimplementing masked mean-pooling — see the
roadmap's coordination note under your clap.py task.

Usage:
    from fusion.encoders.language._pooling import mean_pool, last_token_pool
"""

from __future__ import annotations

import torch


def mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Masked mean-pool token embeddings into one vector per sequence.

    Padding positions (``attention_mask == 0``) are excluded from the
    average, so pooled results don't depend on batch-relative padding
    length. Used by T5 (which has no CLS/pooler token at all) and
    available as an alternative strategy for BERT/RoBERTa/XLM.

    Args:
        last_hidden_state: Token embeddings, shape ``(B, T, D)``.
        attention_mask: Bool or 0/1 mask, shape ``(B, T)``; 1/True = real
            token, 0/False = padding.

    Returns:
        Pooled embeddings of shape ``(B, D)``.
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (B, T, 1)
    summed = (last_hidden_state * mask).sum(dim=1)  # (B, D)
    counts = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
    return summed / counts  # (B, D)


def last_token_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Pool by taking each sequence's last non-padded token embedding.

    Standard choice for decoder-only causal models (GPT-2) which have no
    CLS token and where later positions have seen the whole sequence.

    Args:
        last_hidden_state: Token embeddings, shape ``(B, T, D)``.
        attention_mask: Bool or 0/1 mask, shape ``(B, T)``; 1/True = real
            token, 0/False = padding. Assumes right-padding (real tokens
            first, padding after) — the usual HF tokenizer default.

    Returns:
        Pooled embeddings of shape ``(B, D)``.
    """
    seq_lengths = attention_mask.to(torch.long).sum(dim=1)  # (B,)
    last_idx = (seq_lengths - 1).clamp(min=0)  # (B,)
    batch_idx = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[batch_idx, last_idx]  # (B, D)
