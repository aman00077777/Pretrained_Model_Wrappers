"""
Task H1 — BaseEncoder

Foundation class every Phase 2 encoder (vision / language / audio) must
inherit from. Suyash and Aryan depend on this file, so its contract must
stay stable once merged to develop.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch.nn as nn

from fusion.encoders.modal_tensor import ModalTensor
from fusion.utils.logging import get_logger

logger = get_logger(__name__)


class BaseEncoder(nn.Module, ABC):
    """
    Every subclass must:
      - implement encode(inputs) -> ModalTensor
      - implement get_output_dim() -> int
      - support freeze / unfreeze / freeze_except_last_n
      - support from_config()-based construction
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        logger.info(
            "event=initialization_started encoder=%s",
            self.__class__.__name__,
        )

    @abstractmethod
    def encode(self, inputs) -> ModalTensor:
        """Run the forward pass and return a ModalTensor."""
        raise NotImplementedError

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return the dimensionality of the pooled embedding."""
        raise NotImplementedError

    def freeze(self) -> None:
        """Make every parameter non-trainable."""
        for p in self.parameters():
            p.requires_grad = False
        logger.info("event=encoder_frozen encoder=%s", self.__class__.__name__)

    def unfreeze(self) -> None:
        """Make every parameter trainable."""
        for p in self.parameters():
            p.requires_grad = True
        logger.info("event=encoder_unfrozen encoder=%s", self.__class__.__name__)

    def freeze_except_last_n(self, n: int) -> None:
        """
        1. Freeze everything.
        2. Get the immediate child modules.
        3. Select the last n of them.
        4. Make only their parameters trainable.

        n < 0            -> ValueError
        n == 0           -> everything stays frozen
        n >= child count -> all children become trainable
        """
        if n < 0:
            raise ValueError(
                f"event=invalid_freeze_n encoder={self.__class__.__name__} n={n}"
            )

        self.freeze()

        children: List[nn.Module] = list(self.children())
        if n == 0 or not children:
            return

        for child in children[-n:]:
            for p in child.parameters():
                p.requires_grad = True

        logger.info(
            "event=freeze_except_last_n encoder=%s n=%d child_count=%d",
            self.__class__.__name__, n, len(children),
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(config)
