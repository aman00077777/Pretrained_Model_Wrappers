"""
fusion/encoders/base.py — STUB. See fusion/__init__.py for why this exists.

``BaseEncoder`` is imported by loader.py (``from fusion.encoders.base
import BaseEncoder``) and converter.py, and its contract is only
partially visible from how those two files use it:

  * ``load_pretrained()`` expects ``wrapper_cls.from_pretrained(hf_model_id,
    cache_dir=..., **kwargs)`` to return a ``BaseEncoder`` instance.
  * ``converter.py``'s docstring says a converted encoder "satisfies
    ``encode()``, ``get_output_dim()``, ``freeze()``, ``unfreeze()``, and
    ``freeze_except_last_n()``".

Everything below is inferred from just those two call sites — it is a
best-effort contract, not a confirmed one. In particular,
``freeze_except_last_n``'s exact semantics ("last n" of what — layers?
parameter tensors?) is unspecified anywhere I could find; this
implementation freezes all but the last *n* top-level named parameters,
clearly marked below so it's easy to correct once the real definition
turns up.
"""

from __future__ import annotations

import abc
from typing import Any, Optional

import torch.nn as nn

from fusion.types import ModalTensor


class BaseEncoder(nn.Module, abc.ABC):
    """Common interface every FUSION encoder (custom or pretrained) implements.

    Args:
        output_dim (int): Dimensionality of the pooled embedding this
            encoder produces.
    """

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError(f"`output_dim` must be positive, got {output_dim}.")
        self._output_dim = output_dim

    @classmethod
    @abc.abstractmethod
    def from_pretrained(
        cls, model_name_or_path: str, cache_dir: Optional[str] = None, **kwargs: Any
    ) -> "BaseEncoder":
        """Construct this encoder by loading a pretrained checkpoint."""
        raise NotImplementedError

    @abc.abstractmethod
    def encode(self, *args: Any, **kwargs: Any) -> ModalTensor:
        """Run the encoder and return a pooled :class:`ModalTensor`."""
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """Return the dimensionality of this encoder's pooled output."""
        return self._output_dim

    def freeze(self) -> None:
        """Disable gradients for every parameter in this encoder."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Enable gradients for every parameter in this encoder."""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_except_last_n(self, n: int) -> None:
        """Freeze all parameters except the last *n* (by iteration order).

        Best-guess semantics — see the module docstring. "Last n" here
        means the last *n* parameter tensors returned by
        ``self.named_parameters()``, not the last *n* transformer layers.

        Args:
            n (int): Number of trailing parameter tensors to leave trainable.
        """
        if n < 0:
            raise ValueError(f"`n` must be >= 0, got {n}.")
        names = [name for name, _ in self.named_parameters()]
        trainable = set(names[len(names) - n :]) if n > 0 else set()
        for name, param in self.named_parameters():
            param.requires_grad = name in trainable
