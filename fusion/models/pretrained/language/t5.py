"""
T5 language encoder wrapper.

Usage:
    from fusion.encoders.language import T5Encoder

DEVIATION FROM BaseEncoder's BERT/RoBERTa-style default (per the
roadmap's instruction to flag these to Aman Sharma): T5 is an
encoder-decoder model with no CLS token and no pooler at all, so there
is no "cls" pooling option here — only masked mean-pooling over the
*encoder's* hidden states. This wrapper loads ``T5EncoderModel`` (the
encoder stack only), not the full ``T5ForConditionalGeneration`` —
there's no decoder involved in producing a text embedding.
"""

from __future__ import annotations

from typing import Any, List, Optional

from transformers import AutoTokenizer, T5EncoderModel

from fusion.constants import Modality
from fusion.encoders.base import BaseEncoder
from fusion.encoders.language._pooling import mean_pool
from fusion.types import ModalTensor


class T5Encoder(BaseEncoder):
    """Wraps a Hugging Face T5 encoder stack as a FUSION language encoder.

    Args:
        model: Loaded ``T5EncoderModel`` (encoder-only, no decoder).
        tokenizer: The matching tokenizer.
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        super().__init__(output_dim=model.config.d_model)
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "t5-base",
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> "T5Encoder":
        """Load a pretrained T5 encoder + tokenizer from Hugging Face.

        Args:
            model_name_or_path: HF Hub id or local path, e.g. ``"t5-base"``.
            cache_dir: Optional local cache directory for downloaded weights.
            **kwargs: Forwarded to ``T5EncoderModel.from_pretrained``.

        Returns:
            An initialised T5Encoder.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        model = T5EncoderModel.from_pretrained(model_name_or_path, cache_dir=cache_dir, **kwargs)
        return cls(model=model, tokenizer=tokenizer)

    def encode(self, texts: List[str], max_length: int = 512) -> ModalTensor:
        """Encode a batch of raw text into pooled embeddings.

        Args:
            texts: List of strings.
            max_length: Truncation length passed to the tokenizer.

        Returns:
            ModalTensor wrapping a ``(B, output_dim)`` embedding tensor,
            pooled by masked mean over the encoder's token embeddings
            (T5 has no CLS/pooler token to use instead).
        """
        device = next(self.model.parameters()).device
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        ).to(device)
        outputs = self.model(**batch)
        pooled = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
        return ModalTensor(tensor=pooled, modality=Modality.LANGUAGE)
