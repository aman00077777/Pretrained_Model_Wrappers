"""
fusion/types.py — STUB. See fusion/__init__.py for why this exists.

``ModalTensor`` is the return type every wrapper's ``encode()`` is
supposed to produce per the Phase 6 roadmap doc, but it wasn't defined
anywhere in either repo. This is a minimal, best-guess shape: the pooled
embedding tensor plus which modality produced it. Replace with the real
definition if one exists elsewhere — the field name assumed by the
language wrappers here is ``.tensor``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fusion.constants import Modality


@dataclass
class ModalTensor:
    """A pooled embedding tensor tagged with the modality that produced it.

    Args:
        tensor (torch.Tensor): Pooled embedding, shape ``(B, output_dim)``.
        modality (Modality): Which modality produced this embedding.
    """

    tensor: torch.Tensor
    modality: Modality

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape
