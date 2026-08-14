"""Task S8 — Custom Language Encoder. Wraps an arbitrary nn.Module; not auto-registered."""
import logging
import time

import torch.nn as nn

from fusion.encoders.base import BaseEncoder
from fusion.encoders.modal_tensor import ModalTensor, Modality
from fusion.utils.logging import get_logger, log_event

logger = get_logger(__name__)


class CustomLanguageEncoder(BaseEncoder):
    def __init__(self, module: nn.Module, output_dim: int):
        super().__init__(config={"model_name": "custom"})
        self.module = module
        self._output_dim = output_dim
        log_event(logger, logging.INFO, "initialization_complete", encoder="custom_language",
                   output_dim=self._output_dim)

    def get_output_dim(self) -> int:
        return self._output_dim

    def encode(self, inputs) -> ModalTensor:
        if inputs is None or len(inputs) == 0:
            log_event(logger, logging.ERROR, "encoding_failed", encoder="custom_language",
                       error_type="ValueError")
            raise ValueError("encode() received an empty batch")

        start = time.time()
        log_event(logger, logging.DEBUG, "encode_started", encoder="custom_language",
                   batch_size=inputs.shape[0])
        try:
            output = self.module(inputs)
            log_event(logger, logging.DEBUG, "forward_complete", encoder="custom_language")
        except Exception as exc:
            log_event(logger, logging.ERROR, "encoding_failed", encoder="custom_language",
                       error_type=type(exc).__name__)
            raise

        duration_ms = int((time.time() - start) * 1000)
        log_event(logger, logging.DEBUG, "encode_complete", encoder="custom_language",
                   output_shape=list(output.shape), duration_ms=duration_ms)

        return ModalTensor(data=output, modality=Modality.LANGUAGE)
