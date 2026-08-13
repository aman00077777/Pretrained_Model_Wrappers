"""
align.py

Defines ALIGNWrapper: wraps HuggingFace's ALIGN model
(kakaobrain/align-base) as a FUSION multimodal encoder.

Mirrors the same encode_image / encode_text / compute_similarity /
encode / get_output_dim pattern established in clip.py.

Expected config:
    {"model_name": "kakaobrain/align-base"}

The model_name is optional and defaults to:
    "kakaobrain/align-base"
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
        """
        Args:
            config:
                Dictionary optionally containing "model_name",
                a HuggingFace ALIGN checkpoint ID.

                Defaults to:
                    "kakaobrain/align-base"
        """
        super().__init__(config)

        self.model_name = self.config.get(
            "model_name",
            "kakaobrain/align-base",
        )

        self.model = AlignModel.from_pretrained(
            self.model_name
        )

        self.processor = AlignProcessor.from_pretrained(
            self.model_name
        )

        self.model.eval()

    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
    ) -> ModalTensor:
        """
        Encode image inputs using ALIGN's vision tower.

        Args:
            images:
                A single PIL image or a list of PIL images.

        Returns:
            ModalTensor with modality=Modality.VISION containing
            image embeddings of shape (B, D).
        """
        if isinstance(images, Image.Image):
            images = [images]

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        )

        with torch.no_grad():
            embeds = self.model.get_image_features(**inputs)

        return ModalTensor(
            data=embeds,
            modality=Modality.VISION,
        )

    def encode_text(
        self,
        texts: Union[str, List[str]],
    ) -> ModalTensor:
        """
        Encode text inputs using ALIGN's text tower.

        Args:
            texts:
                A single string or a list of strings.

        Returns:
            ModalTensor with modality=Modality.LANGUAGE containing
            text embeddings of shape (B, D).
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

        return ModalTensor(
            data=embeds,
            modality=Modality.LANGUAGE,
        )

    def compute_similarity(
        self,
        images: Union[Image.Image, List[Image.Image]],
        texts: Union[str, List[str]],
    ) -> torch.Tensor:
        """
        Compute cosine similarity between image and text embeddings.

        Args:
            images:
                A single PIL image or a list of PIL images.

            texts:
                A single string or a list of strings.

        Returns:
            Tensor of shape (B_images, B_texts) containing the cosine
            similarity between every image/text pair.
        """
        # Direct calls only -- never route through self.encode().
        image_embeds = self.encode_image(images).embedding
        text_embeds = self.encode_text(texts).embedding

        image_embeds = image_embeds / image_embeds.norm(
            p=2,
            dim=-1,
            keepdim=True,
        )

        text_embeds = text_embeds / text_embeds.norm(
            p=2,
            dim=-1,
            keepdim=True,
        )

        return image_embeds @ text_embeds.T

    def encode(self, inputs) -> ModalTensor:
        """
        BaseEncoder contract entry point.

        Dispatches to encode_image() or encode_text() based on
        the input data type.

        Args:
            inputs:
                str / List[str] for text, or
                PIL.Image.Image / torch.Tensor for images.

                A list of either type is also supported.

        Returns:
            ModalTensor returned by encode_image() or encode_text().

        Raises:
            TypeError:
                If the input's type is not recognized.

        Note:
            A list of strings that are actually image file paths
            cannot be distinguished from text based on type alone.
        """
        item = (
            inputs[0]
            if isinstance(inputs, (list, tuple))
            else inputs
        )

        if isinstance(item, str):
            return self.encode_text(inputs)

        elif isinstance(item, (Image.Image, torch.Tensor)):
            return self.encode_image(inputs)

        else:
            raise TypeError(
                f"ALIGNWrapper.encode() got an unsupported input type: "
                f"{type(item)}. "
                "Expected str / List[str] for text, or "
                "PIL.Image.Image / torch.Tensor "
                "(or a list of either) for images."
            )

    def get_output_dim(self) -> int:
        """
        Return the dimensionality of ALIGN's projected embeddings.

        ALIGN's image and text towers project into a shared embedding
        space, so one dimension applies to both.

        Returns:
            int: ALIGN projection dimension.
        """
        return self.model.config.projection_dim