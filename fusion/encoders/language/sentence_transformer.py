"""Task S6 — Sentence Transformer encoder."""
import logging
import time

from sentence_transformers import SentenceTransformer

from fusion.encoders.base import BaseEncoder
from fusion.encoders.modal_tensor import ModalTensor, Modality
from fusion.encoders.registry import ENCODER_REGISTRY
from fusion.utils.logging import get_logger, log_event

logger = get_logger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@ENCODER_REGISTRY.register("sentence_transformer")
class SentenceTransformerEncoder(BaseEncoder):
    def __init__(self, config: dict = None):
        config = config or {}
        config.setdefault("model_name", DEFAULT_MODEL)
        super().__init__(config)
        model_name = config["model_name"]
        log_event(logger, logging.INFO, "model_loading", encoder="sentence_transformer", model=model_name)
        self.model = SentenceTransformer(model_name)
        self._output_dim = self.model.get_sentence_embedding_dimension()
        log_event(logger, logging.INFO, "initialization_complete", encoder="sentence_transformer",
                   output_dim=self._output_dim)

    def get_output_dim(self) -> int:
        return self._output_dim

    def encode(self, inputs) -> ModalTensor:
        if isinstance(inputs, str):
            inputs = [inputs]
        if not inputs:
            log_event(logger, logging.ERROR, "encoding_failed", encoder="sentence_transformer",
                       error_type="ValueError")
            raise ValueError("encode() received an empty batch")

        start = time.time()
        log_event(logger, logging.DEBUG, "encode_started", encoder="sentence_transformer",
                   batch_size=len(inputs))

        try:
            embeddings = self.model.encode(inputs, convert_to_tensor=True)
            log_event(logger, logging.DEBUG, "forward_complete", encoder="sentence_transformer")
        except Exception as exc:
            log_event(logger, logging.ERROR, "encoding_failed", encoder="sentence_transformer",
                       error_type=type(exc).__name__)
            raise

        duration_ms = int((time.time() - start) * 1000)
        log_event(logger, logging.DEBUG, "encode_complete", encoder="sentence_transformer",
                   output_shape=list(embeddings.shape), duration_ms=duration_ms)

        return ModalTensor(data=embeddings, modality=Modality.LANGUAGE)
