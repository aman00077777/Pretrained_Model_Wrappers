"""
BERT language encoder wrapper.

Usage:
    from fusion.encoders.language import BERTEncoder
"""

from __future__ import annotations

from typing import Any, List, Optional

from transformers import AutoModel, AutoTokenizer

from fusion.constants import Modality
from fusion.encoders.base import BaseEncoder
from fusion.encoders.language._pooling import mean_pool
from fusion.types import ModalTensor


class BERTEncoder(BaseEncoder):
    """Wraps a Hugging Face BERT model as a FUSION language encoder.

    Args:
        model: Loaded Hugging Face BERT model (e.g. ``BertModel``).
        tokenizer: The matching tokenizer.
        pooling (str): ``"cls"`` (BERT's pooler_output, default) or
            ``"mean"`` (masked mean over token embeddings).
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
        model_name_or_path: str = "bert-base-uncased",
        cache_dir: Optional[str] = None,
        pooling: str = "cls",
        **kwargs: Any,
    ) -> "BERTEncoder":
        """Load a pretrained BERT checkpoint + tokenizer from Hugging Face.

        Args:
            model_name_or_path: HF Hub id or local path, e.g.
                ``"bert-base-uncased"``, ``"bert-large-uncased"``.
            cache_dir: Optional local cache directory for downloaded weights.
            pooling: ``"cls"`` or ``"mean"`` — see the class docstring.
            **kwargs: Forwarded to ``AutoModel.from_pretrained``.

        Returns:
            An initialised BERTEncoder.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_name_or_path, cache_dir=cache_dir, **kwargs)
        return cls(model=model, tokenizer=tokenizer, pooling=pooling)

    def encode(self, texts: List[str], max_length: int = 512) -> ModalTensor:
        """Encode a batch of raw text into pooled embeddings.

        Args:
            texts: List of strings. An empty string is handled the same
                as any other input (tokenizes to special tokens only).
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
