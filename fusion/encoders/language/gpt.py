"""
GPT-2 language encoder wrapper.

Usage:
    from fusion.encoders.language import GPTEncoder

File name note: loader.py's registry imports this from
``fusion.encoders.language.gpt`` (``gpt.py``), not ``gpt2.py`` — the
roadmap doc's own file-naming table says ``gpt2.py``. Named to match
loader.py, since that's the file GitHub search / the actual import
statement will look for.

DEVIATION FROM BaseEncoder's BERT/RoBERTa-style default (per the
roadmap's instruction to flag these to Aman Sharma): GPT-2 is causal
decoder-only, so it has no CLS token and, unlike encoder models, later
positions have seen the whole sequence — pooling here takes each
sequence's *last non-padded token* instead. GPT-2's tokenizer also has
no pad token by default; one is added (aliased to EOS) the first time
it's needed.
"""

from __future__ import annotations

from typing import Any, List, Optional

from transformers import AutoModel, AutoTokenizer

from fusion.constants import Modality
from fusion.encoders.base import BaseEncoder
from fusion.encoders.language._pooling import last_token_pool
from fusion.types import ModalTensor


class GPTEncoder(BaseEncoder):
    """Wraps a Hugging Face GPT-2 model as a FUSION language encoder.

    Args:
        model: Loaded Hugging Face GPT-2 model.
        tokenizer: The matching tokenizer (pad token added if missing).
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        hidden_size = getattr(model.config, "hidden_size", None) or model.config.n_embd
        super().__init__(output_dim=hidden_size)
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "gpt2",
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> "GPTEncoder":
        """Load a pretrained GPT-2 checkpoint + tokenizer from Hugging Face.

        Args:
            model_name_or_path: HF Hub id or local path, e.g. ``"gpt2"``.
            cache_dir: Optional local cache directory for downloaded weights.
            **kwargs: Forwarded to ``AutoModel.from_pretrained``.

        Returns:
            An initialised GPTEncoder.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_name_or_path, cache_dir=cache_dir, **kwargs)
        return cls(model=model, tokenizer=tokenizer)

    def encode(self, texts: List[str], max_length: int = 512) -> ModalTensor:
        """Encode a batch of raw text into pooled embeddings.

        Args:
            texts: List of strings.
            max_length: Truncation length passed to the tokenizer.

        Returns:
            ModalTensor wrapping a ``(B, output_dim)`` embedding tensor,
            pooled from each sequence's last non-padded token.
        """
        device = next(self.model.parameters()).device
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        ).to(device)
        outputs = self.model(**batch)
        pooled = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        return ModalTensor(tensor=pooled, modality=Modality.LANGUAGE)
