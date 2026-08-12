"""DINOv2 vision encoder wrapper.

Usage:
    from fusion.models.pretrained.vision.dinov2 import DINOv2Encoder
"""

from __future__ import annotations

import inspect
import time
from typing import Any, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

from fusion.constants import Modality
from fusion.encoders.base import BaseEncoder
from fusion.encoders.modal_tensor import ModalTensor
from fusion.exceptions import DataError, EncoderError
from fusion.utils.logging import get_logger

logger = get_logger(__name__)


class DINOv2Encoder(BaseEncoder):
    """Wraps Hugging Face's DINOv2 vision model as a FUSION vision encoder."""

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Load pretrained model and processor."""
        logger.info("event=model_loading encoder=dinov2 model=%s", model_name)
        try:
            processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
            model = Dinov2Model.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)
        except Exception as e:
            logger.error("event=model_loading_failed encoder=dinov2 model=%s error=%s", model_name, str(e))
            raise EncoderError(f"Failed to load DINOv2 model '{model_name}': {e}") from e

        self._output_dim = model.config.hidden_size

        sig = inspect.signature(BaseEncoder.__init__)
        if "output_dim" in sig.parameters:
            super().__init__(output_dim=self._output_dim)
        else:
            super().__init__(config=kwargs.get("config", {}))

        self.processor = processor
        self.model = model

        logger.info("event=initialization_complete encoder=dinov2 output_dim=%d", self._output_dim)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> DINOv2Encoder:
        """Load pretrained model and processor."""
        return cls(model_name=model_name_or_path, cache_dir=cache_dir, **kwargs)

    def encode(
        self,
        images: Union[Image.Image, List[Image.Image], torch.Tensor],
    ) -> ModalTensor:
        """Encode image input(s) into pooled embeddings.

        Args:
            images: A single PIL Image, list of PIL Images, or preprocessed torch Tensor.

        Returns:
            ModalTensor with modality=Modality.VISION and shape (batch_size, embedding_dim).
        """
        start_time = time.time()

        if isinstance(images, torch.Tensor):
            if images.dim() != 4:
                raise DataError(f"Tensor input must have shape (B, C, H, W), got shape {tuple(images.shape)}")
            if images.shape[0] == 0:
                raise DataError("Received an empty tensor batch.")
            pixel_values = images
            batch_size = pixel_values.shape[0]
        else:
            img_list = images if isinstance(images, list) else [images]
            batch_size = len(img_list)

            if batch_size == 0:
                raise DataError("Received an empty batch of images.")

            for idx, img in enumerate(img_list):
                if not isinstance(img, Image.Image):
                    raise DataError(f"Input at index {idx} is not a PIL Image, got type {type(img)}")
                if img.width == 0 or img.height == 0:
                    raise DataError(f"Invalid image dimensions at index {idx}: {img.width}x{img.height}")

            try:
                processed = self.processor(images=img_list, return_tensors="pt")
                pixel_values = processed["pixel_values"]
            except Exception as e:
                logger.error("event=preprocessing_failed encoder=dinov2 error=%s", str(e))
                raise DataError(f"Failed to preprocess images: {e}") from e

        device = next(self.model.parameters()).device
        pixel_values = pixel_values.to(device)

        logger.debug("event=encode_started encoder=dinov2 batch_size=%d", batch_size)
        logger.debug("event=preprocessing_complete encoder=dinov2")

        try:
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
            logger.debug("event=forward_complete encoder=dinov2")
        except Exception as e:
            logger.error("event=encoding_failed encoder=dinov2 error_type=%s", type(e).__name__)
            raise EncoderError(f"Failed model forward pass: {e}") from e

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state
            if pooled.dim() == 3:
                pooled = pooled[:, 0]

        if pooled.dim() == 4:
            pooled = pooled.flatten(1)

        duration_ms = (time.time() - start_time) * 1000
        logger.debug(
            "event=encode_complete encoder=dinov2 output_shape=%s duration_ms=%.2f",
            list(pooled.shape),
            duration_ms,
        )

        return ModalTensor(data=pooled, modality=Modality.VISION)

    def get_output_dim(self) -> int:
        """Return the embedding dimension."""
        return self._output_dim
