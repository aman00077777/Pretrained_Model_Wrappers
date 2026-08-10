"""
XLM (XLM-RoBERTa) multilingual language encoder wrapper.

Usage:
    from fusion.encoders.language import XLMEncoder

Note: loader.py registers the friendly name ``"xlm-roberta-base"`` against
this class, pointing at the HF id ``"xlm-roberta-base"`` — i.e. XLM-R, not
the older ``XLMModel`` / ``XLMWithLMHeadModel`` architecture. XLM-R is
RoBERTa-shaped, so this mirrors RoBERTaEncoder's pooling exactly.
"""

from __future__ import annotations

from typing import Any, List, Optional

from transformers import AutoModel, AutoTokenizer

from fusion.constants import Modality
from fusion.encoders.base import BaseEncoder
from fusion.encoders.language._pooling import mean_pool
from fusion.types import ModalTensor


class XLMEncoder(BaseEncoder):
    """Wraps a Hugging Face XLM-RoBERTa model as a FUSION language encoder.

    Args:
        model: Loaded Hugging Face XLM-RoBERTa model.
        tokenizer: The matching tokenizer.
        pooling (str): ``"cls"`` (pooler_output, default) or ``"mean"``.
    """

    def __init__(self, model: Any, tokenizer: Any, pooling: str = "cls") -> None:
        if pooling not in ("cls", "mean"):
            raise ValueError(f"`pooling` must be 'cls' or 'mean', got {pooling!r}.")
        super().__init__(output_dim=model.config.hidden_size)
        self.model = model
        self.tokenizer = tokenizer
        self.pooling = pooling

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "xlm-roberta-base",
        cache_dir: Optional[str] = None,
        pooling: str = "cls",
        **kwargs: Any,
    ) -> "XLMEncoder":
        """Load a pretrained XLM-RoBERTa checkpoint + tokenizer.

        Args:
            model_name_or_path: HF Hub id or local path, e.g.
                ``"xlm-roberta-base"``.
            cache_dir: Optional local cache directory for downloaded weights.
            pooling: ``"cls"`` or ``"mean"``.
            **kwargs: Forwarded to ``AutoModel.from_pretrained``.

        Returns:
            An initialised XLMEncoder.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_name_or_path, cache_dir=cache_dir, **kwargs)
        return cls(model=model, tokenizer=tokenizer, pooling=pooling)

    def encode(self, texts: List[str], max_length: int = 512) -> ModalTensor:
        """Encode a batch of raw multilingual text into pooled embeddings.

        Args:
            texts: List of strings, any of the ~100 languages XLM-R covers.
            max_length: Truncation length passed to the tokenizer.

        Returns:
            ModalTensor wrapping a ``(B, output_dim)`` embedding tensor.
        """
        device = next(self.model.parameters()).device
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        ).to(device)
        outputs = self.model(**batch)

        if self.pooling == "cls":
            pooler_output = getattr(outputs, "pooler_output", None)
            pooled = (
                pooler_output if pooler_output is not None else outputs.last_hidden_state[:, 0]
            )
        else:
            pooled = mean_pool(outputs.last_hidden_state, batch["attention_mask"])

        return ModalTensor(tensor=pooled, modality=Modality.LANGUAGE)
