"""
align.py

Defines ALIGNWrapper: wraps HuggingFace's ALIGN model
(kakaobrain/align-base) as a FUSION multimodal encoder.

Follows the same BaseEncoder pattern as CLIPWrapper:
    encode_image / encode_text / encode / get_output_dim

Expected config:
    {"model_name": "kakaobrain/align-base"}
"""

from typing import Any, Dict, List, Union

import torch
from PIL import Image
from transformers import AlignModel, AlignProcessor

from fusion.constants import Modality
from fusion.encoders.base import BaseEncoder
from fusion.encoders.modal_tensor import ModalTensor
from fusion.encoders.registry import register_encoder


@register_encoder("align")
class ALIGNWrapper(BaseEncoder):
    """FUSION-native wrapper around HuggingFace's AlignModel."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        self.model_name = self.config.get(
            "model_name",
            "kakaobrain/align-base",
        )

        self.model = AlignModel.from_pretrained(self.model_name)
        self.processor = AlignProcessor.from_pretrained(self.model_name)

        self.model.eval()

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------

    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
    ) -> ModalTensor:
        """Encode one or more PIL images into ALIGN vision embeddings.

        Args:
            images: A single ``PIL.Image.Image`` or a list of them.

        Returns:
            ModalTensor of shape (B, projection_dim) tagged as VISION.
        """
        if isinstance(images, Image.Image):
            images = [images]

        inputs = self.processor(images=images, return_tensors="pt")

        with torch.no_grad():
            embeds = self.model.get_image_features(**inputs)

        return ModalTensor(data=embeds, modality=Modality.VISION)

    # ------------------------------------------------------------------
    # Language
    # ------------------------------------------------------------------

    def encode_text(
        self,
        texts: Union[str, List[str]],
    ) -> ModalTensor:
        """Encode one or more strings into ALIGN text embeddings.

        Args:
            texts: A single string or a list of strings.

        Returns:
            ModalTensor of shape (B, projection_dim) tagged as LANGUAGE.
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        with torch.no_grad():
            embeds = self.model.get_text_features(**inputs)

        return ModalTensor(data=embeds, modality=Modality.LANGUAGE)

    # ------------------------------------------------------------------
    # Generic encode
    # ------------------------------------------------------------------

    def encode(self, inputs) -> ModalTensor:
        """Dispatch based on input type.

        Args:
            inputs: ``PIL.Image.Image`` / list of images → vision path.
                    ``str`` / list of strings → text path.

        Raises:
            TypeError: If the input type is not recognised.
        """
        item = inputs[0] if isinstance(inputs, (list, tuple)) else inputs

        if isinstance(item, str):
            return self.encode_text(inputs)
        elif isinstance(item, (Image.Image, torch.Tensor)):
            return self.encode_image(inputs)
        else:
            raise TypeError(
                f"ALIGNWrapper.encode() got an unsupported input type: "
                f"{type(item)}. Expected str / List[str] for text, or "
                "PIL.Image.Image (or a list) for images."
            )

    def get_output_dim(self) -> int:
        return self.model.config.projection_dim
