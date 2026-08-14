"""Task S1 — CLIP Text encoder."""
import logging
import time

import torch
from transformers import CLIPModel, CLIPProcessor

from fusion.encoders.base import BaseEncoder
from fusion.encoders.modal_tensor import ModalTensor, Modality
from fusion.encoders.registry import ENCODER_REGISTRY
from fusion.utils.logging import get_logger, log_event

logger = get_logger(__name__)

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


@ENCODER_REGISTRY.register("clip_text")
class CLIPTextEncoder(BaseEncoder):
    def __init__(self, config: dict = None):
        config = config or {}
        config.setdefault("model_name", DEFAULT_MODEL)
        super().__init__(config)
        model_name = config["model_name"]
        log_event(logger, logging.INFO, "model_loading", encoder="clip_text", model=model_name)
        full_model = CLIPModel.from_pretrained(model_name)
        self.text_model = full_model.text_model
        self.text_projection = full_model.text_projection
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self._output_dim = full_model.config.projection_dim
        log_event(logger, logging.INFO, "initialization_complete", encoder="clip_text",
                   output_dim=self._output_dim)

    def get_output_dim(self) -> int:
        return self._output_dim

    def encode(self, inputs) -> ModalTensor:
        if isinstance(inputs, str):
            inputs = [inputs]
        if not inputs:
            log_event(logger, logging.ERROR, "encoding_failed", encoder="clip_text",
                       error_type="ValueError")
            raise ValueError("encode() received an empty batch")

        start = time.time()
        log_event(logger, logging.DEBUG, "encode_started", encoder="clip_text", batch_size=len(inputs))

        try:
            processed = self.processor(text=inputs, return_tensors="pt", padding=True, truncation=True)
            log_event(logger, logging.DEBUG, "preprocessing_complete", encoder="clip_text")

            outputs = self.text_model(
                input_ids=processed["input_ids"],
                attention_mask=processed["attention_mask"],
            )
            pooled = self.text_projection(outputs.pooler_output)
            log_event(logger, logging.DEBUG, "forward_complete", encoder="clip_text")
        except Exception as exc:
            log_event(logger, logging.ERROR, "encoding_failed", encoder="clip_text",
                       error_type=type(exc).__name__)
            raise

        duration_ms = int((time.time() - start) * 1000)
        log_event(logger, logging.DEBUG, "encode_complete", encoder="clip_text",
                   output_shape=list(pooled.shape), duration_ms=duration_ms)

        return ModalTensor(
            data=pooled,
            modality=Modality.LANGUAGE,
            attention_mask=processed["attention_mask"],
        )
