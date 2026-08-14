"""Shared pooling utilities for language encoders (Sec. 7)."""
import torch


def masked_mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token hidden states, ignoring padding tokens.

    hidden_states: [B, L, H]
    attention_mask: [B, L]
    """
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = torch.sum(hidden_states * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def last_token_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Select the last non-padding token's hidden state per sequence (used by GPT)."""
    last_indices = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.size(0)
    return hidden_states[torch.arange(batch_size), last_indices]
