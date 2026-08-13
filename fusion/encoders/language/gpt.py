"""Task S5 — GPT encoder. Uses last non-padding token as sequence representation."""
import logging
import time

from transformers import AutoModel, AutoTokenizer

from fusion.encoders.base import BaseEncoder
from fusion.encoders.modal_tensor import ModalTensor, Modality
from fusion.encoders.registry import ENCODER_REGISTRY
from fusion.encoders.language.pooling import last_token_pooling
from fusion.utils.logging import get_logger, log_event

logger = get_logger(__name__)

DEFAULT_MODEL = "gpt2"


@ENCODER_REGISTRY.register("gpt")
class GPTEncoder(BaseEncoder):
    def __init__(self, config: dict = None):
        config = config or {}
        config.setdefault("model_name", DEFAULT_MODEL)
        super().__init__(config)
        model_name = config["model_name"]
        log_event(logger, logging.INFO, "model_loading", encoder="gpt", model=model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._output_dim = self.model.config.hidden_size
        log_event(logger, logging.INFO, "initialization_complete", encoder="gpt",
                   output_dim=self._output_dim)

    def get_output_dim(self) -> int:
        return self._output_dim

    def encode(self, inputs) -> ModalTensor:
        if isinstance(inputs, str):
            inputs = [inputs]
        if not inputs:
            log_event(logger, logging.ERROR, "encoding_failed", encoder="gpt",
                       error_type="ValueError")
            raise ValueError("encode() received an empty batch")

        start = time.time()
        log_event(logger, logging.DEBUG, "encode_started", encoder="gpt", batch_size=len(inputs))

        try:
            tokenized = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
            log_event(logger, logging.DEBUG, "preprocessing_complete", encoder="gpt")

            outputs = self.model(
                input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"]
            )
            log_event(logger, logging.DEBUG, "forward_complete", encoder="gpt")

            pooled = last_token_pooling(outputs.last_hidden_state, tokenized["attention_mask"])
        except Exception as exc:
            log_event(logger, logging.ERROR, "encoding_failed", encoder="gpt",
                       error_type=type(exc).__name__)
            raise

        duration_ms = int((time.time() - start) * 1000)
        log_event(logger, logging.DEBUG, "encode_complete", encoder="gpt",
                   output_shape=list(pooled.shape), duration_ms=duration_ms)

        return ModalTensor(
            data=pooled,
            modality=Modality.LANGUAGE,
            attention_mask=tokenized["attention_mask"],
        )
